import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = DATA_DIR / "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# SOTA Model for Embeddings (Small, Fast, Local)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# make sure ollama is installed
LLM_MODEL = "llama3.2" 