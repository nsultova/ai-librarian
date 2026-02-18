import sys
from pathlib import Path
import argparse
import traceback
from src.config import LLM_MODEL
from src.rag import PROMPT_TEMPLATE


def test_embedding_model():
    """Test if embedding model loads and generates embeddings."""
    print("="*60)
    print("TESTING EMBEDDING MODEL")
    print("="*60)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.config import EMBEDDING_MODEL_NAME
        
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
        traceback.print_exc()
        return False
    
    
def test_rag_pipeline():
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from src.config import LLM_MODEL
        from src.rag import PROMPT_TEMPLATE

        # 1. check prompt template renders correctly
        print("\nChecking prompt template...")
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        rendered = prompt.format(
            context="The sky is blue because of Rayleigh scattering.",
            question="Why is the sky blue?"
        )
        assert "{context}" not in rendered, "context variable was not injected"
        assert "{question}" not in rendered, "question variable was not injected"
        print("Prompt template renders correctly")

        # 2. check ollama is reachable and responds
        print(f"\nCalling Ollama with model: {LLM_MODEL}")
        llm = ChatOllama(model=LLM_MODEL)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": "The sky is blue because of Rayleigh scattering.",
            "question": "Why is the sky blue?"
        })

        assert isinstance(answer, str), "answer is not a string"
        assert len(answer.strip()) > 0, "answer is empty"

        print(f"Response received ({len(answer)} chars)")
        print(f"Preview: {answer[:200]}")
        print("\nRAG PIPELINE TEST PASSED")
        return True

    except ConnectionRefusedError:
        print("\nFAILED: Ollama is not running. Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


    
def test_metadata_retrieval():
    """Test if metadata filtering works."""
    print("=" * 60)
    print("TESTING METADATA RETRIEVAL")
    print("=" * 60)
    
    try:
        from src.vector import get_vector_db
        
        db = get_vector_db()
        
        # Check if we have any documents
        collection = db._collection
        count = collection.count()
        print(f"\nTotal documents in DB: {count}")
        
        if count == 0:
            print("No documents found. Upload some files first.")
            return False
        
        # Sample a few docs to see their metadata
        results = db.similarity_search("test", k=3)
        
        print("\nSample metadata from retrieved chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Metadata: {doc.metadata}")
            print(f"Content preview: {doc.page_content[:100]}...")
        
        # Test filtering (if you have metadata)
        if results and 'author' in results[0].metadata:
            author = results[0].metadata['author']
            print(f"\nTesting filter by author: {author}")
            
            filtered_results = db.similarity_search(
                "test",
                k=3,
                filter={"author": author}
            )
            print(f"Found {len(filtered_results)} chunks by {author}")
        
        print("\n✓ METADATA RETRIEVAL TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ METADATA RETRIEVAL TEST FAILED: {e}")
        traceback.print_exc()
        return False

TESTS = {
    "embedding": test_embedding_model,
    "rag": test_rag_pipeline,
    "metadata": test_metadata_retrieval
}


def main():
    parser = argparse.ArgumentParser(description="AI Librarian model tests")
    parser.add_argument(
        "tests",
        nargs="*",
        choices=list(TESTS.keys()),
        help=f"Tests to run. Available: {', '.join(TESTS.keys())}. Runs all if omitted."
    )
    args = parser.parse_args()

    to_run = {name: TESTS[name] for name in (args.tests or TESTS.keys())}

    print("\n" + "=" * 60)
    print("AI LIBRARIAN MODEL VERIFICATION")
    print("=" * 60)

    results = {name: fn() for name, fn in to_run.items()}

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()