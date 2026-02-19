import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, List, Any

from src.config import UPLOAD_DIR
from src.ingest import chunk_file
from src.vector import add_documents_to_db, get_vector_db
from src.rag import query_library

logger = logging.getLogger(__name__)

app = FastAPI(title="My Private Library")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_filter_options() -> Dict[str, List[str]]:
    try:
        db = get_vector_db()
        result = db._collection.get(include=["metadatas"])

        if not result or not result["metadatas"]:
            return {"authors": [], "file_types": [], "titles": []}

        authors, file_types, titles = set(), set(), set()
        for meta in result["metadatas"]:
            if meta.get("author") and meta["author"] != "Unknown":
                authors.add(meta["author"])
            if meta.get("file_type"):
                file_types.add(meta["file_type"])
            if meta.get("title"):
                titles.add(meta["title"])

        return {
            "authors": sorted(authors),
            "file_types": sorted(file_types),
            "titles": sorted(titles),
        }
    except Exception as e:
        logger.error(f"Failed to fetch filter options: {e}")
        return {"authors": [], "file_types": [], "titles": []}


def _aggregate_library() -> List[Dict[str, Any]]:
    """
    Build a structured book list from DB metadata.
    Deduplicates by source_file, collects unique chapter titles per book.
    Direct metadata query — no embedding/retrieval involved.
    """
    try:
        db = get_vector_db()
        result = db._collection.get(include=["metadatas"])

        if not result or not result["metadatas"]:
            return []

        books: Dict[str, Dict[str, Any]] = {}

        for meta in result["metadatas"]:
            key = str(meta.get("source_file") or meta.get("title") or "unknown")

            if key not in books:
                books[key] = {
                    "title": meta.get("title", "Unknown"),
                    "author": meta.get("author", "Unknown"),
                    "file_type": meta.get("file_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "page_count": meta.get("page_count"),
                    # "publisher": meta.get("publisher", ""),
                    # "publication_date": meta.get("publication_date", ""),
                    # "language": meta.get("language", ""),
                    "chapters": set(),
                    "chunk_count": 0,
                }

            if meta.get("chapter"):
                books[key]["chapters"].add(meta["chapter"])

            books[key]["chunk_count"] += 1

        for book in books.values():
            book["chapters"] = sorted(book["chapters"])

        return sorted(books.values(), key=lambda b: b["title"].lower())

    except Exception as e:
        logger.error(f"Failed to aggregate library: {e}")
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    prefill_title: Optional[str] = None,
    prefill_chapter: Optional[str] = None,
):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "filter_options": _get_filter_options(),
            "library": _aggregate_library(),
            "prefill_title": prefill_title,
            "prefill_chapter": prefill_chapter,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith((".pdf", ".epub")):
        raise HTTPException(status_code=400, detail="Only PDF and EPUB files are supported")

    file_location = UPLOAD_DIR / safe_name
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = chunk_file(str(file_location))
    add_documents_to_db(chunks)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": f"Successfully ingested: {safe_name}",
            "filter_options": _get_filter_options(),
            "library": _aggregate_library(),
            "prefill_title": None,
            "prefill_chapter": None,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    author: Optional[str] = Form(None),
    file_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    chapter: Optional[str] = Form(None),
):
    filters = {}
    if author and author != "all":
        filters["author"] = author
    if file_type and file_type != "all":
        filters["file_type"] = file_type
    if title and title != "all":
        filters["title"] = title
    if chapter and chapter.strip():
        filters["chapter"] = chapter.strip()

    answer, sources = query_library(query, filters if filters else None)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
            "sources": sources,
            "active_filters": filters,
            "filter_options": _get_filter_options(),
            "library": _aggregate_library(),
            "prefill_title": title,
            "prefill_chapter": chapter,
        },
    )


# ---------------------------------------------------------------------------
# JSON / debug
# ---------------------------------------------------------------------------

@app.get("/api/filters")
async def get_available_filters():
    return _get_filter_options()


@app.get("/library")
async def get_library():
    return _aggregate_library()


@app.get("/debug/search")
async def debug_search(q: str, k: int = 5):
    db = get_vector_db()
    docs = db.similarity_search(q, k=k)
    return [
        {
            "rank": i + 1,
            "metadata": doc.metadata,
            "preview": doc.page_content[:300],
            "length": len(doc.page_content),
        }
        for i, doc in enumerate(docs)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
