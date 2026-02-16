import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = DATA_DIR / "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# LLM_MODEL = "llama3.2" 

# Model Configuration
# Embedding: NVIDIA's flagship model for high-quality semantic search
EMBEDDING_MODEL_NAME = "nvidia/NV-Embed-v2"

# LLM: Mistral 7B via Ollama - balanced performance for local deployment
LLM_MODEL = "mistral:7b"

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5  # Number of chunks to retrieve
