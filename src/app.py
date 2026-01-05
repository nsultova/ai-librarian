import shutil
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import UPLOAD_DIR
from src.ingest import chunk_file 
from src.vector import add_documents_to_db
from src.rag import query_library

app = FastAPI(title="My Private Library")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
        
    file_location = UPLOAD_DIR / file.filename
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = chunk_file(str(file_location))
    add_documents_to_db(chunks)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": f"Successfully ingested {file.filename}"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)