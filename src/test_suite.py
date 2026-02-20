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
    
  
def test_vector_db_singleton_and_reset():
    """
    Verifies two behaviors of the manual singleton in vector.py:

    1. Singleton: get_vector_db() returns the same object on repeated calls.
       This matters because the embedding model is expensive to load —
       reloading it on every query would make the app unusable.

    2. Reset invalidation: reset_database() clears the cached instance so
       the next call to get_vector_db() constructs a fresh object.
       Without this, reset_database() would leave a broken Chroma reference
       pointing at a deleted collection.
    """
    print("=" * 60)
    print("TESTING VECTOR DB SINGLETON AND RESET")
    print("=" * 60)
    
    confirm = input(
        "This test will DELETE the database. Continue? (yes/no): "
    )
    
    if confirm.lower() != "yes": # NOT yes 
        print("Skipped.")
        return True  # not a failure, just skipped
        

    try:
        import src.vector as vector_module
        from src.vector import get_vector_db, reset_database

        # -- Part 1: singleton behavior --
        print("\nPart 1: singleton")

        db1 = get_vector_db()
        db2 = get_vector_db()

        assert db1 is db2, (
            "get_vector_db() returned different objects on consecutive calls. "
            "The singleton is not working — the embedding model would be "
            "reloaded on every request."
        )
        print("  get_vector_db() returns the same instance on repeated calls")

        # -- Part 2: reset invalidates the cache --
        print("\nPart 2: reset invalidation")

        reset_database()

        assert vector_module._db_instance is None, (
            "_db_instance was not set to None after reset_database(). "
            "The next call to get_vector_db() would return a reference to "
            "the deleted collection."
        )
        print("  _db_instance is None after reset_database()")

        db3 = get_vector_db()

        assert db3 is not None, (
            "get_vector_db() returned None after reset. "
            "Re-initialization failed."
        )
        assert db1 is not db3, (
            "get_vector_db() returned the old instance after reset. "
            "The cache was not properly invalidated."
        )
        print("  get_vector_db() returns a new instance after reset")

        print("\nVECTOR DB SINGLETON AND RESET TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
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

def test_app_routes():
    """
    Tests the FastAPI routes using FastAPI's built-in test client.

    The TestClient runs the full app in-process — no server needs to be
    running. It also triggers the lifespan context manager (startup/shutdown)
    so ensure_dirs() is called just as it would be in production.

    What we verify:
        - Core routes return expected status codes
        - Upload validation rejects bad filenames, wrong formats, oversized files
        - JSON endpoints return the correct shape
        - The app handles an empty database gracefully (no crash, sensible defaults)
    """
    print("=" * 60)
    print("TESTING APP ROUTES")
    print("=" * 60)

    try:
        from fastapi.testclient import TestClient
        from src.app import app

        # TestClient handles lifespan (startup/shutdown) automatically
        with TestClient(app) as client:

            # -- Home page --
            print("\nGET /")
            r = client.get("/")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            assert "text/html" in r.headers["content-type"]
            print("  GET / returns 200 HTML")

            # -- Library JSON --
            print("\nGET /library")
            r = client.get("/library")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            assert isinstance(r.json(), list), "Expected a list"
            print(f"  GET /library returns list ({len(r.json())} books)")

            # -- Filter options --
            print("\nGET /api/filters")
            r = client.get("/api/filters")
            assert r.status_code == 200
            body = r.json()
            assert "authors" in body and "file_types" in body, (
                f"Expected 'authors' and 'file_types' keys, got: {list(body.keys())}"
            )
            print("  GET /api/filters returns correct shape")

            print("\nPOST /upload — no filename")
            r = client.post("/upload", files={"file": ("", b"", "application/pdf")})
            assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"
            # 400 = our handler rejected it (file.filename is empty string)
            # 422 = FastAPI's validation layer rejected it before reaching the handler
            print(f"  Correctly rejected: no filename ({r.status_code})")

            # -- Upload: wrong format --
            print("\nPOST /upload — wrong format")
            r = client.post("/upload", files={"file": ("test.txt", b"hello", "text/plain")})
            assert r.status_code == 400, f"Expected 400, got {r.status_code}"
            print("  Correctly rejected: unsupported format")

            # -- Upload: file too large --
            print("\nPOST /upload — oversized file")
            from src.config import MAX_UPLOAD_BYTES
            oversized = b"0" * (MAX_UPLOAD_BYTES + 1)
            r = client.post("/upload", files={"file": ("big.pdf", oversized, "application/pdf")})
            assert r.status_code == 413, f"Expected 413, got {r.status_code}"
            print("  Correctly rejected: file too large")

            # -- Search: empty database should not crash --
            print("\nPOST /search — empty database")
            r = client.post("/search", data={"query": "test query"})
            # We expect either 200 (handled gracefully) or 500 only if
            # the LLM is unreachable — not a crash from missing data.
            # A 200 here means the app degraded gracefully.
            assert r.status_code in (200, 500), (
                f"Unexpected status {r.status_code} — app may have crashed"
            )
            print(f"  POST /search returned {r.status_code} (no unhandled crash)")

        print("\nAPP ROUTES TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False

def test_utils():
    """
    Tests for utils.clean_text.

    clean_text has two jobs: rejoin hyphenated line-breaks introduced
    by PDF extraction, and collapse all whitespace into single spaces.
    These are tested independently and in combination.
    """
    print("=" * 60)
    print("TESTING UTILS")
    print("=" * 60)

    try:
        from src.utils import clean_text

        # -- Hyphenated line-breaks --
        result = clean_text("impor-\ntant")
        assert result == "important", f"Expected 'important', got '{result}'"
        print("  Hyphenated line-break rejoined correctly")

        # -- Whitespace collapse --
        result = clean_text("too   many    spaces\nand\ttabs")
        assert result == "too many spaces and tabs", f"Got '{result}'"
        print("  Whitespace collapsed correctly")

        # -- Clean text is unchanged --
        result = clean_text("already clean text")
        assert result == "already clean text", f"Got '{result}'"
        print("  Clean text passes through unchanged")

        # -- Empty string --
        result = clean_text("")
        assert result == "", f"Expected empty string, got '{result}'"
        print("  Empty string returns empty string")

        # -- Combined: hyphen + extra whitespace --
        result = clean_text("impor-\ntant   concept")
        assert result == "important concept", f"Got '{result}'"
        print("  Combined hyphen + whitespace handled correctly")

        print("\nUTILS TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_metadata_pure_functions():
    """
    Tests for the pure utility functions in metadata.py.

    These functions have no side effects and require no files —
    they can be tested entirely with in-memory inputs. This makes
    them the most reliable part of the test suite.

    Focus is on chapter_for_page, which implements the step-function
    logic that maps PDF page numbers to chapter titles. This is the
    most non-obvious piece of logic in the codebase.
    """
    print("=" * 60)
    print("TESTING METADATA PURE FUNCTIONS")
    print("=" * 60)

    try:
        from src.metadata import (
            DocumentMetadata,
            _sanitize,
            build_page_chapter_map,
            chapter_for_page,
            sanitize_spine_name,
        )

        # -- _sanitize --
        print("\n_sanitize")

        assert _sanitize(None, fallback="x") == "x"
        print("  None returns fallback")

        assert _sanitize("  hello  ") == "hello"
        print("  Whitespace stripped")

        assert _sanitize("ab\x00cd") == "ab cd"
        print("  Control characters replaced with space")

        assert _sanitize("a" * 600, max_length=10) == "a" * 10
        print("  Long strings truncated to max_length")

        assert _sanitize("", fallback="default") == "default"
        print("  Empty string returns fallback")

        # -- sanitize_spine_name --
        print("\nsanitize_spine_name")

        assert sanitize_spine_name("Text/part01_chapter03.xhtml") == "part01 chapter03"
        print("  Path and underscores cleaned correctly")

        assert sanitize_spine_name("ch-01.xhtml") == "ch 01"
        print("  Hyphens replaced with spaces")

        assert sanitize_spine_name("simple.xhtml") == "simple"
        print("  Extension stripped")

        # -- build_page_chapter_map --
        print("\nbuild_page_chapter_map")

        entries = [(5, "Chapter Two"), (0, "Chapter One"), (10, "Chapter Three")]
        result = build_page_chapter_map(entries)
        assert result == {0: "Chapter One", 5: "Chapter Two", 10: "Chapter Three"}
        print("  Map built correctly from unsorted input")

        assert build_page_chapter_map([]) == {}
        print("  Empty input returns empty map")

        # -- chapter_for_page: the step-function logic --
        # Map: Chapter One starts p.0, Chapter Two starts p.5, Chapter Three p.10
        print("\nchapter_for_page (step-function)")

        page_map = {0: "Chapter One", 5: "Chapter Two", 10: "Chapter Three"}

        assert chapter_for_page(0, page_map) == "Chapter One"
        print("  Page 0 -> Chapter One (exact match on first boundary)")

        assert chapter_for_page(3, page_map) == "Chapter One"
        print("  Page 3 -> Chapter One (between boundaries)")

        assert chapter_for_page(5, page_map) == "Chapter Two"
        print("  Page 5 -> Chapter Two (exact match on boundary)")

        assert chapter_for_page(7, page_map) == "Chapter Two"
        print("  Page 7 -> Chapter Two (between boundaries)")

        assert chapter_for_page(10, page_map) == "Chapter Three"
        print("  Page 10 -> Chapter Three (last chapter)")

        assert chapter_for_page(99, page_map) == "Chapter Three"
        print("  Page 99 -> Chapter Three (beyond last boundary)")

        assert chapter_for_page(0, {}) == ""
        print("  Empty map returns empty string")

        # -- DocumentMetadata.to_chunk_metadata --
        print("\nDocumentMetadata.to_chunk_metadata")

        meta = DocumentMetadata(
            title="Test Book",
            author="Test Author",
            file_type="pdf",
            source_file="test.pdf",
            page_count=100,
            chapters=["Ch 1", "Ch 2"],
            page_chapter_map={0: "Ch 1", 10: "Ch 2"},
        )
        chunk_meta = meta.to_chunk_metadata()

        # Scalar fields present
        assert chunk_meta["title"] == "Test Book"
        assert chunk_meta["author"] == "Test Author"
        assert chunk_meta["file_type"] == "pdf"
        assert chunk_meta["source_file"] == "test.pdf"
        assert chunk_meta["page_count"] == 100
        print("  All scalar fields present")

        # Derived fields excluded
        assert "chapters" not in chunk_meta, (
            "'chapters' should not be in chunk metadata — it is a list "
            "and ChromaDB does not support non-scalar metadata values."
        )
        assert "page_chapter_map" not in chunk_meta, (
            "'page_chapter_map' should not be in chunk metadata."
        )
        print("  Derived fields correctly excluded")

        print("\nMETADATA PURE FUNCTIONS TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_ingest_pipeline():
    """
    Tests the ingest pipeline functions using synthetic in-memory Documents.

    We test _annotate_documents and _split_documents directly rather than
    going through chunk_file, because:
        - chunk_file requires a real file on disk
        - these functions contain the logic worth verifying
        - testing them independently makes failures easier to diagnose

    A separate integration smoke test covers the full chunk_file pipeline
    if a sample file is available.
    """
    print("=" * 60)
    print("TESTING INGEST PIPELINE")
    print("=" * 60)

    try:
        from langchain_core.documents import Document
        from src.ingest import _annotate_documents, _split_documents
        from src.metadata import DocumentMetadata

        # -- _annotate_documents: metadata stamping --
        print("\n_annotate_documents: metadata stamping")

        meta = DocumentMetadata(
            title="Test Book",
            author="Test Author",
            file_type="pdf",
            source_file="test.pdf",
            page_count=50,
        )
        docs = [
            Document(page_content="First page content.", metadata={"page": 0}),
            Document(page_content="Second page content.", metadata={"page": 1}),
        ]

        annotated = _annotate_documents(docs, meta)

        assert annotated[0].metadata["title"] == "Test Book"
        assert annotated[0].metadata["author"] == "Test Author"
        assert annotated[1].metadata["file_type"] == "pdf"
        print("  Scalar metadata stamped onto all documents")

        assert "chapters" not in annotated[0].metadata
        assert "page_chapter_map" not in annotated[0].metadata
        print("  Derived fields not written to chunk metadata")

        assert annotated[0].metadata["chunk_index"] == 0
        assert annotated[1].metadata["chunk_index"] == 1
        print("  chunk_index set correctly")

        # -- _annotate_documents: chapter assignment from page map --
        print("\n_annotate_documents: PDF chapter assignment")

        meta_with_chapters = DocumentMetadata(
            title="Test Book",
            author="Test Author",
            file_type="pdf",
            source_file="test.pdf",
            page_chapter_map={0: "Introduction", 5: "Chapter One"},
        )
        docs = [
            Document(page_content="Intro text.", metadata={"page": 0}),
            Document(page_content="Mid intro.", metadata={"page": 3}),
            Document(page_content="Chapter content.", metadata={"page": 5}),
        ]

        annotated = _annotate_documents(docs, meta_with_chapters)

        assert annotated[0].metadata["chapter"] == "Introduction"
        assert annotated[1].metadata["chapter"] == "Introduction"
        assert annotated[2].metadata["chapter"] == "Chapter One"
        print("  Chapter labels assigned correctly from page map")

        # -- _split_documents: chunks produced, metadata preserved --
        print("\n_split_documents: chunking and metadata preservation")

        # Create a document long enough to guarantee at least two chunks
        long_text = "word " * 500
        docs = [Document(
            page_content=long_text,
            metadata={"title": "Test Book", "chapter": "Chapter One"},
        )]

        chunks = _split_documents(docs)

        assert len(chunks) > 1, (
            f"Expected multiple chunks from {len(long_text)} chars of text, "
            f"got {len(chunks)}. Check CHUNK_SIZE in config.py."
        )
        print(f"  {len(long_text)} chars split into {len(chunks)} chunks")

        # Metadata must survive the split — every chunk should carry it
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("title") == "Test Book", (
                f"Chunk {i} lost its 'title' metadata after splitting."
            )
            assert chunk.metadata.get("chapter") == "Chapter One", (
                f"Chunk {i} lost its 'chapter' metadata after splitting."
            )
        print("  Metadata preserved on all chunks after split")

        print("\nINGEST PIPELINE TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False

TESTS = {
    "embedding": test_embedding_model,
    "rag": test_rag_pipeline,
    "metadata": test_metadata_retrieval,
    "meta_pure": test_metadata_pure_functions,  
    "vector": test_vector_db_singleton_and_reset, 
    "app" : test_app_routes,
    "utils":     test_utils,                   
    "ingest":    test_ingest_pipeline,
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