from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import DB_DIR, EMBEDDING_MODEL_NAME

def get_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_function
    )

def add_documents_to_db(chunks):
    db = get_vector_db()
    # Adding documents automatically generates embeddings
    db.add_documents(chunks)
    print("Documents added to Vector DB.")