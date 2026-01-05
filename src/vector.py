from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DB_DIR, EMBEDDING_MODEL_NAME

def get_vector_db() -> Chroma:
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function
    )

def add_documents_to_db(chunks: List[Document]) -> None:
    db = get_vector_db()
    db.add_documents(chunks)
    print(f"Successfully added {len(chunks)} chunks to Vector DB.")