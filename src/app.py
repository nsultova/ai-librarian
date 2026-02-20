"""
app.py — FastAPI web server for the AI Librarian.

Request lifecycle:
    GET  /          → renders the main page with library + filter state
    POST /upload    → ingests a PDF or EPUB into the vector database
    POST /search    → runs a RAG query and renders the answer

All heavy work (file parsing, embedding, LLM calls) is dispatched to a
thread pool via run_in_threadpool so the async event loop stays free to
handle other requests during long-running ingestion or queries.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from src.config import MAX_UPLOAD_BYTES, UPLOAD_DIR, ensure_dirs
from src.ingest import chunk_file
from src.rag import query_library
from src.vector import add_documents_to_db, get_vector_db



# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.

    Code before `yield` runs on startup (before first request).
    Code after `yield` runs on shutdown (after last request).

    Used here to create required data directories on startup.
    Kept separate from config.py to avoid filesystem side effects
    at import time — config.py may be imported by tests or CLI tools
    that do not need directories created.
    """
    ensure_dirs()
    yield
    # shutdown logic would go here if needed


logger = logging.getLogger(__name__)

#app = FastAPI(title="AI Librarian")
app = FastAPI(title="AI Librarian", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_library_state() -> Dict[str, Any]:
    """
    Single ChromaDB scan returning both the structured book list and
    the available filter options for the search form.

    Called on every page render. Kept as one function (rather than two)
    to avoid scanning the full collection twice per request — db.get()
    fetches all stored metadata regardless of how much is read from it.

    Returns:
        {
            "library":        list of book dicts sorted by title,
            "filter_options": {"authors": [...], "file_types": [...]}
        }

    On any database error, returns empty defaults so the page still renders.
    """
    try:
        result = get_vector_db().get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        books: Dict[str, Dict] = {}
        authors: set = set()
        file_types: set = set()

        for meta in metadatas:
            # source_file is the deduplication key — title alone is
            # unreliable because two files could share the same title.
            key = str(meta.get("source_file") or meta.get("title") or "unknown")

            if key not in books:
                books[key] = {
                    "title":       meta.get("title", "Unknown"),
                    "author":      meta.get("author", "Unknown"),
                    "file_type":   meta.get("file_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "page_count":  meta.get("page_count"),
                    "chapters":    set(),
                    "chunk_count": 0,
                }

            if meta.get("chapter"):
                books[key]["chapters"].add(meta["chapter"])

            books[key]["chunk_count"] += 1

            if meta.get("author") and meta["author"] != "Unknown":
                authors.add(meta["author"])
            if meta.get("file_type"):
                file_types.add(meta["file_type"])

        book_list = sorted(books.values(), key=lambda b: b["title"].lower())
        for book in book_list:
            book["chapters"] = sorted(book["chapters"])

        return {
            "library": book_list,
            "filter_options": {
                "authors":    sorted(authors),
                "file_types": sorted(file_types),
            },
        }

    except Exception as e:
        logger.error(f"Failed to load library state: {e}")
        return {
            "library": [],
            "filter_options": {"authors": [], "file_types": []},
        }


def _build_template_context(request: Request, state: Dict, **kwargs) -> Dict:
    """
    Build the base template context dict shared by all page-rendering routes.

    Accepts extra keyword arguments merged in for route-specific values
    (query, answer, sources, active_filters, prefill_title, prefill_chapter).
    """
    return {
        "request":        request,
        "library":        state["library"],
        "filter_options": state["filter_options"],
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    prefill_title:   Optional[str] = None,
    prefill_chapter: Optional[str] = None,
) -> HTMLResponse:
    """
    Render the main page.

    prefill_title and prefill_chapter come from sidebar chapter links —
    they pre-filter the search form so the user's next query is scoped
    to that book/chapter without extra interaction.
    """
    state = await run_in_threadpool(_get_library_state)
    context = _build_template_context(
        request, state,
        prefill_title=prefill_title,
        prefill_chapter=prefill_chapter,
    )
    return templates.TemplateResponse("index.html", context)


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
) -> HTMLResponse:
    """
    Ingest a PDF or EPUB into the vector database.

    Pipeline:
        1. Validate filename and file size
        2. Save to disk (data/uploads/)
        3. Parse, clean, and chunk the document  (run_in_threadpool)
        4. Embed chunks and store in ChromaDB    (run_in_threadpool)
        5. Re-render the main page with a status message

    Steps 3 and 4 are dispatched to a thread pool because they are
    synchronous and CPU/IO-heavy (embedding model inference can take
    several minutes for large files). Running them directly in an async
    handler would block the event loop and freeze all other requests.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith((".pdf", ".epub")):
        raise HTTPException(status_code=400, detail="Only PDF and EPUB files are supported.")

    # Read into memory once — used for size check and writing to disk.
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds the {max_mb}MB limit.")

    file_location = UPLOAD_DIR / safe_name
    with open(file_location, "wb") as f:
        f.write(content)

    chunks = await run_in_threadpool(chunk_file, str(file_location))
    await run_in_threadpool(add_documents_to_db, chunks)

    state = await run_in_threadpool(_get_library_state)
    context = _build_template_context(
        request, state,
        message=f"Successfully ingested: {safe_name}",
        prefill_title=None,
        prefill_chapter=None,
    )
    return templates.TemplateResponse("index.html", context)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query:     str            = Form(...),
    author:    Optional[str]  = Form(None),
    file_type: Optional[str]  = Form(None),
    title:     Optional[str]  = Form(None),
    chapter:   Optional[str]  = Form(None),
) -> HTMLResponse:
    """
    Run a RAG query against the vector database and render the answer.

    Filter fields (author, file_type, title, chapter) narrow the ChromaDB
    similarity search to chunks matching all provided values. "all" is the
    sentinel value from the <select> dropdowns meaning no filter applied.

    query_library() is dispatched to a thread pool because it is
    synchronous — it calls the embedding model (to embed the query) and
    the Ollama LLM (to generate the answer), both of which are blocking.
    """
    filters: Dict[str, str] = {}
    if author    and author    != "all": filters["author"]    = author
    if file_type and file_type != "all": filters["file_type"] = file_type
    if title     and title     != "all": filters["title"]     = title
    if chapter   and chapter.strip():    filters["chapter"]   = chapter.strip()

    answer, sources = await run_in_threadpool(
        query_library, query, filters or None
    )

    state = await run_in_threadpool(_get_library_state)
    context = _build_template_context(
        request, state,
        query=query,
        answer=answer,
        sources=sources,
        active_filters=filters,
        prefill_title=title,
        prefill_chapter=chapter,
    )
    return templates.TemplateResponse("index.html", context)


# ---------------------------------------------------------------------------
# JSON / debug endpoints
# ---------------------------------------------------------------------------

@app.get("/library")
async def get_library() -> List[Dict]:
    """Return the structured book list as JSON. Useful for debugging ingestion."""
    state = await run_in_threadpool(_get_library_state)
    return state["library"]


@app.get("/api/filters")
async def get_available_filters() -> Dict:
    """Return available filter options as JSON."""
    state = await run_in_threadpool(_get_library_state)
    return state["filter_options"]


@app.get("/debug/search")
async def debug_search(q: str, k: int = 5) -> List[Dict]:
    """
    Run a raw similarity search and return chunk previews with metadata.
    Use this to verify that retrieved chunks are semantically relevant
    and that metadata (title, author, chapter) looks correct after ingestion.

    Example:
        curl "http://localhost:8000/debug/search?q=what+is+the+first+chapter+about"
    """
    db = get_vector_db()
    docs = await run_in_threadpool(db.similarity_search, q, k)
    return [
        {
            "rank":     i + 1,
            "metadata": doc.metadata,
            "preview":  doc.page_content[:300],
            "length":   len(doc.page_content),
        }
        for i, doc in enumerate(docs)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)