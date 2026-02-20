from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = DATA_DIR / "chroma_db"

def ensure_dirs() -> None:
    """Create required data directories if they do not exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# LLM_MODEL = "llama3.2" 

# Model Configuration
# Embedding: NVIDIA's flagship model for high-quality semantic search
# EMBEDDING_MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"  # Only 120MB!

# LLM: Mistral 7B via Ollama - balanced performance for local deployment
LLM_MODEL = "mistral:7b"

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5  # Number of chunks to retrieve
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
