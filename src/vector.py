from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DB_DIR, EMBEDDING_MODEL_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def get_vector_db() -> Chroma:
#     embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
#     return Chroma(
#         persist_directory=str(DB_DIR),
#         embedding_function=embedding_function
#     )


def get_vector_db() -> Chroma:
    """
    Get or create the ChromaDB vector database.
    
    Returns:
        Chroma: Vector database instance with configured embeddings
    """
    logger.info(f"Initializing vector database with model: {EMBEDDING_MODEL_NAME}")
    
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            'device': 'cpu'  # Use 'cuda' if GPU available
            #'trust_remote_code': True  # Required for NVIDIA models
        },
        encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
    )
    
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}  # Explicit similarity metric
    )


def add_documents_to_db(chunks: List[Document]) -> None:
    """
    Add document chunks to the vector database.
    
    Args:
        chunks: List of Document objects to add
    """
    if not chunks:
        logger.warning("No chunks provided to add to database")
        return
    
    try:
        db = get_vector_db()
        db.add_documents(chunks)
        logger.info(f"Successfully added {len(chunks)} chunks to Vector DB.")
    except Exception as e:
        logger.error(f"Failed to add documents to database: {e}")
        raise