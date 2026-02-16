import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_embedding_model():
    """Test if embedding model loads and generates embeddings."""
    print("="*60)
    print("TESTING EMBEDDING MODEL")
    print("="*60)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import EMBEDDING_MODEL_NAME
        
        print(f"\nModel: {EMBEDDING_MODEL_NAME}")
        print("Loading model (this may take a while on first run)...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding generation
        test_texts = [
            "This is a test document about artificial intelligence.",
            "RAG systems combine retrieval and generation.",
            "The weather is sunny today."
        ]
        
        print("\nGenerating embeddings for test texts...")
        embedded = embeddings.embed_documents(test_texts)
        
        print(f"Successfully generated {len(embedded)} embeddings")
        print(f"Embedding dimension: {len(embedded[0])}")
        
        # Test similarity
        query = "Tell me about AI systems"
        query_embedding = embeddings.embed_query(query)
        
        print(f"Query embedding dimension: {len(query_embedding)}")
        print("\n EMBEDDING MODEL TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n  EMBEDDING MODEL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("AI LIBRARIAN MODEL VERIFICATION")
    print("="*60)
    
    results = {
        "Embedding Model": test_embedding_model()
    }

if __name__ == "__main__":
    sys.exit(main())