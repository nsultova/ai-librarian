import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DB_DIR, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level singleton. None until first call to get_vector_db().
# Managed manually so reset_database() can invalidate it safely.
# (lru_cache was not used here because it cannot be cleared selectively
#  — reset_database() destroys the underlying ChromaDB collection, which
#  would leave a cached but broken Chroma object on subsequent calls.)
_db_instance: Chroma | None = None


def _create_db() -> Chroma:
    """
    Instantiate the Chroma vector store with the configured embedding model.

    Separated from get_vector_db() so the construction logic can be read
    and tested independently from the caching logic.

    Embedding notes:
    - normalize_embeddings=True is required for cosine similarity to work
      correctly. Cosine similarity assumes unit vectors; without this, dot
      products are not comparable across embeddings of different magnitudes.
    - e5-small-v2 is a 120MB instruction-tuned model. It expects queries
      prefixed with "query: " and documents with "passage: " at training
      time, but LangChain's HuggingFaceEmbeddings handles this automatically.
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_vector_db() -> Chroma:
    """
    Return the shared Chroma instance, creating it on first call.

    This is a manual singleton pattern. The instance is stored in the
    module-level _db_instance variable and reused across all calls to
    avoid reloading the embedding model (which is expensive).

    The instance is invalidated (set to None) by reset_database(), which
    forces re-creation on the next call.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = _create_db()
    return _db_instance


def add_documents_to_db(chunks: List[Document]) -> None:
    """
    Embed and store document chunks in the vector database.

    Embedding is handled internally by the Chroma instance — each chunk's
    page_content is converted to a vector using the configured embedding
    model, then stored alongside the raw text and metadata.

    Args:
        chunks: Document objects produced by ingest.chunk_file().
                Each must have scalar-only metadata values (str, int, float)
                because ChromaDB does not support nested types per chunk.
    """
    if not chunks:
        logger.warning("add_documents_to_db called with empty chunk list")
        return

    logger.info(f"Embedding and storing {len(chunks)} chunks...")
    logger.info("This may take several minutes for large documents.")

    try:
        get_vector_db().add_documents(chunks)
        logger.info(f"Successfully stored {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Failed to add documents to database: {e}")
        raise


def reset_database() -> None:
    """
    Delete all documents from the vector database and invalidate the
    cached instance.

    The global _db_instance is set to None after deletion so the next
    call to get_vector_db() constructs a fresh Chroma object against the
    newly created (empty) collection, rather than reusing a reference to
    the deleted one.
    """
    global _db_instance

    try:
        get_vector_db().delete_collection()
        logger.info("Vector database collection deleted.")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise
    finally:
        # Always clear the instance, even if deletion raised —
        # a broken collection is not recoverable without re-initialization.
        _db_instance = None
        logger.info("Cached DB instance cleared. Next call will reinitialize.")