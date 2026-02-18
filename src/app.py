import shutil
import json
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

from src.config import UPLOAD_DIR
from src.ingest import chunk_file 
from src.vector import add_documents_to_db, get_vector_db
from src.rag import query_library


logger = logging.getLogger(__name__)

app = FastAPI(title="My Private Library")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # 1. sanitize and validate first — before touching the filesystem
    safe_name = Path(file.filename).name
    if not safe_name.endswith((".pdf", ".epub")):
        raise HTTPException(status_code=400, detail="Only PDF and EPUB files are supported")
    
    # 2. now safe to construct the path and write
    file_location = UPLOAD_DIR / safe_name
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. process
    chunks = chunk_file(str(file_location))
    add_documents_to_db(chunks)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"Successfully ingested {safe_name}"
    })

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    answer, sources = query_library(query)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": query,
        "answer": answer,
        "sources": sources
    })
    
@app.get("/api/filters")
async def get_available_filters():
    """Return available filter options from the database."""
    try:
        db = get_vector_db()
        collection = db._collection
        
        # Get all unique metadata values
        all_metadata = collection.get(include=['metadatas'])
        
        if not all_metadata or not all_metadata['metadatas']:
            return {"authors": [], "file_types": [], "titles": []}
        
        authors = set()
        file_types = set()
        titles = set()
        
        for meta in all_metadata['metadatas']:
            if meta.get('author') and meta['author'] != 'Unknown':
                authors.add(meta['author'])
            if meta.get('file_type'):
                file_types.add(meta['file_type'])
            if meta.get('title'):
                titles.add(meta['title'])
        
        return {
            "authors": sorted(list(authors)),
            "file_types": sorted(list(file_types)),
            "titles": sorted(list(titles))
        }
    except Exception as e:
        logger.error(f"Failed to get filters: {e}")
        return {"authors": [], "file_types": [], "titles": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)