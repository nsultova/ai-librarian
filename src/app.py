import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, List

from src.config import UPLOAD_DIR
from src.ingest import chunk_file
from src.vector import add_documents_to_db, get_vector_db
from src.rag import query_library


logger = logging.getLogger(__name__)

app = FastAPI(title="My Private Library")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _get_filter_options() -> Dict[str, List[str]]:
    """
    Pull unique metadata values from the DB for populating filter dropdowns.
    Returns empty lists if the DB is empty or unavailable.
    
    Note:
    Called every time a page is rendered, which means every request hits the DB
    — fine for local PoC but not beyond
    
    """
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "filter_options": _get_filter_options()},
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_name = Path(file.filename).name
    if not safe_name.endswith((".pdf", ".epub")):
        raise HTTPException(
            status_code=400, detail="Only PDF and EPUB files are supported"
        )

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
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    author: Optional[str] = Form(None),
    file_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    filters = {}
    if author and author != "all":
        filters["author"] = author
    if file_type and file_type != "all":
        filters["file_type"] = file_type
    if title and title != "all":
        filters["title"] = title

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
        },
    )


@app.get("/api/filters")
async def get_available_filters():
    """JSON endpoint — kept for debugging/external use."""
    return _get_filter_options()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)