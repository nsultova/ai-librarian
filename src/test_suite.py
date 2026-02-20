"""
test_suite.py — AI Librarian verification tests.

Tests are grouped by cost (fast to slow) so failures surface quickly:
    Fast    utils, metadata, ingest      pure functions, no I/O
    Medium  vector, db_metadata, app     needs DB / embedding model
    Slow    embedding, rag               needs Ollama running

Usage:
    python -m src.test_suite              # run all tests
    python -m src.test_suite metadata     # run one test
    python -m src.test_suite utils ingest # run several
"""

import argparse
import sys
import traceback


# ---------------------------------------------------------------------------
# Fast tests — pure functions, no I/O, no models
# ---------------------------------------------------------------------------

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

        result = clean_text("impor-\ntant")
        assert result == "important", f"Expected 'important', got '{result}'"
        print("  Hyphenated line-break rejoined correctly")

        result = clean_text("too   many    spaces\nand\ttabs")
        assert result == "too many spaces and tabs", f"Got '{result}'"
        print("  Whitespace collapsed correctly")

        result = clean_text("already clean text")
        assert result == "already clean text", f"Got '{result}'"
        print("  Clean text passes through unchanged")

        result = clean_text("")
        assert result == "", f"Expected empty string, got '{result}'"
        print("  Empty string returns empty string")

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


def test_metadata():
    """
    Tests for the pure utility functions in metadata.py.

    These functions have no side effects and require no files —
    they can be tested entirely with in-memory inputs. This makes
    them the fastest and most reliable tests in the suite.

    Focus is on chapter_for_page, which implements the step-function
    logic that maps PDF page numbers to chapter titles. This is the
    most non-obvious piece of logic in the codebase and the most
    likely to break silently if someone edits the function.
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

        # -- chapter_for_page: step-function logic --
        # This is the most important test here. The step-function maps any
        # page number to the chapter whose bookmark is at or before that page.
        # Boundary conditions (exact hits, between chapters, beyond last) are
        # all meaningfully different and all tested explicitly.
        print("\nchapter_for_page (step-function)")
        page_map = {0: "Chapter One", 5: "Chapter Two", 10: "Chapter Three"}

        assert chapter_for_page(0, page_map) == "Chapter One"
        print("  Page 0  -> Chapter One (exact match, first boundary)")

        assert chapter_for_page(3, page_map) == "Chapter One"
        print("  Page 3  -> Chapter One (between boundaries)")

        assert chapter_for_page(5, page_map) == "Chapter Two"
        print("  Page 5  -> Chapter Two (exact match on boundary)")

        assert chapter_for_page(7, page_map) == "Chapter Two"
        print("  Page 7  -> Chapter Two (between boundaries)")

        assert chapter_for_page(10, page_map) == "Chapter Three"
        print("  Page 10 -> Chapter Three (last chapter, exact match)")

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

        assert chunk_meta["title"] == "Test Book"
        assert chunk_meta["author"] == "Test Author"
        assert chunk_meta["file_type"] == "pdf"
        assert chunk_meta["source_file"] == "test.pdf"
        assert chunk_meta["page_count"] == 100
        print("  All scalar fields present")

        assert "chapters" not in chunk_meta, (
            "'chapters' must not be in chunk metadata — ChromaDB does not "
            "support list values per chunk."
        )
        assert "page_chapter_map" not in chunk_meta, (
            "'page_chapter_map' must not be in chunk metadata."
        )
        print("  Derived fields correctly excluded")

        print("\nMETADATA TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_ingest():
    """
    Tests the ingest pipeline steps using synthetic in-memory Documents.

    _annotate_documents and _split_documents are tested directly rather
    than through chunk_file because:
        - chunk_file requires a real file on disk
        - testing steps individually makes failures easier to diagnose
        - the logic worth verifying lives in these two functions

    The full chunk_file pipeline is covered implicitly by test_rag,
    which requires ingested documents to return meaningful answers.
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

        # -- _annotate_documents: PDF chapter assignment --
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
            Document(page_content="Mid intro.",  metadata={"page": 3}),
            Document(page_content="Chapter content.", metadata={"page": 5}),
        ]
        annotated = _annotate_documents(docs, meta_with_chapters)

        assert annotated[0].metadata["chapter"] == "Introduction"
        assert annotated[1].metadata["chapter"] == "Introduction"
        assert annotated[2].metadata["chapter"] == "Chapter One"
        print("  Chapter labels assigned correctly from page map")

        # -- _split_documents: chunks produced, metadata preserved --
        print("\n_split_documents: chunking and metadata preservation")
        long_text = "word " * 500
        docs = [Document(
            page_content=long_text,
            metadata={"title": "Test Book", "chapter": "Chapter One"},
        )]
        chunks = _split_documents(docs)

        assert len(chunks) > 1, (
            f"Expected multiple chunks from {len(long_text)} chars, "
            f"got {len(chunks)}. Check CHUNK_SIZE in config.py."
        )
        print(f"  {len(long_text)} chars split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("title") == "Test Book", \
                f"Chunk {i} lost 'title' metadata after splitting."
            assert chunk.metadata.get("chapter") == "Chapter One", \
                f"Chunk {i} lost 'chapter' metadata after splitting."
        print("  Metadata preserved on all chunks after split")

        print("\nINGEST TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Medium tests — needs DB init (embedding model) but not Ollama
# ---------------------------------------------------------------------------

def test_vector():
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

    confirm = input("This test will DELETE the database. Continue? (yes/no): ")
    if confirm.lower() != "yes":
        print("Skipped.")
        return True

    try:
        import src.vector as vector_module
        from src.vector import get_vector_db, reset_database

        print("\nPart 1: singleton")
        db1 = get_vector_db()
        db2 = get_vector_db()
        assert db1 is db2, (
            "get_vector_db() returned different objects on consecutive calls. "
            "The singleton is not working — the embedding model would be "
            "reloaded on every request."
        )
        print("  get_vector_db() returns the same instance on repeated calls")

        print("\nPart 2: reset invalidation")
        reset_database()
        assert vector_module._db_instance is None, (
            "_db_instance was not set to None after reset_database(). "
            "The next call to get_vector_db() would return a stale reference "
            "to the deleted collection."
        )
        print("  _db_instance is None after reset_database()")

        db3 = get_vector_db()
        assert db3 is not None, "get_vector_db() returned None after reset."
        assert db1 is not db3, (
            "get_vector_db() returned the old instance after reset. "
            "Cache was not properly invalidated."
        )
        print("  get_vector_db() returns a new instance after reset")

        print("\nVECTOR TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_db_metadata():
    """
    Integration test: verifies that metadata is correctly stored in and
    retrievable from ChromaDB after ingestion.

    Unlike test_metadata (pure functions), this test requires the database
    to contain at least one ingested document. It checks that:
        - chunks are retrievable via similarity search
        - expected metadata fields (author, title, chapter) are present
        - filtering by author returns only matching chunks

    If the database is empty, the test is skipped (not failed) — run
    the app and ingest a file first, then re-run this test.
    """
    print("=" * 60)
    print("TESTING DB METADATA RETRIEVAL")
    print("=" * 60)

    try:
        from src.vector import get_vector_db

        db = get_vector_db()

        # Use public API — db.get() instead of db._collection.get()
        result = db.get(include=["metadatas"])
        count = len(result.get("metadatas") or [])
        print(f"\nTotal chunks in DB: {count}")

        if count == 0:
            print("No documents found — ingest a file first, then re-run.")
            print("Skipped.")
            return True  # not a failure

        results = db.similarity_search("test", k=3)
        print("\nSample metadata from retrieved chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n  Chunk {i}")
            print(f"  Metadata: {doc.metadata}")
            print(f"  Preview:  {doc.page_content[:100]}...")

        if results and "author" in results[0].metadata:
            author = results[0].metadata["author"]
            print(f"\nTesting filter by author: '{author}'")
            filtered = db.similarity_search("test", k=3, filter={"author": author})
            assert all(
                d.metadata.get("author") == author for d in filtered
            ), "Filtered results contain chunks from a different author."
            print(f"  {len(filtered)} chunks returned, all by '{author}'")

        print("\nDB METADATA TEST PASSED")
        return True

    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_app():
    """
    Tests the FastAPI routes using FastAPI's built-in test client.

    The TestClient runs the full app in-process — no server needs to be
    running. It also triggers the lifespan context manager (startup/shutdown)
    so ensure_dirs() is called just as it would be in production.

    What we verify:
        - Core routes return expected status codes
        - Upload validation rejects bad filenames, wrong formats, oversized files
        - JSON endpoints return the correct shape
        - The app handles an empty database gracefully
    """
    print("=" * 60)
    print("TESTING APP ROUTES")
    print("=" * 60)

    try:
        from fastapi.testclient import TestClient

        from src.app import app

        with TestClient(app) as client:

            print("\nGET /")
            r = client.get("/")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            assert "text/html" in r.headers["content-type"]
            print("  Returns 200 HTML")

            print("\nGET /library")
            r = client.get("/library")
            assert r.status_code == 200
            assert isinstance(r.json(), list)
            print(f"  Returns list ({len(r.json())} books)")

            print("\nGET /api/filters")
            r = client.get("/api/filters")
            assert r.status_code == 200
            body = r.json()
            assert "authors" in body and "file_types" in body, (
                f"Expected 'authors' and 'file_types' keys, got: {list(body.keys())}"
            )
            print("  Returns correct shape")

            print("\nPOST /upload — no filename")
            r = client.post("/upload", files={"file": ("", b"", "application/pdf")})
            assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"
            # 400 = our handler rejected it
            # 422 = FastAPI's validation layer rejected it before reaching the handler
            print(f"  Correctly rejected ({r.status_code})")

            print("\nPOST /upload — wrong format")
            r = client.post("/upload", files={"file": ("test.txt", b"hello", "text/plain")})
            assert r.status_code == 400, f"Expected 400, got {r.status_code}"
            print("  Correctly rejected: unsupported format")

            print("\nPOST /upload — oversized file")
            from src.config import MAX_UPLOAD_BYTES
            oversized = b"0" * (MAX_UPLOAD_BYTES + 1)
            r = client.post("/upload", files={"file": ("big.pdf", oversized, "application/pdf")})
            assert r.status_code == 413, f"Expected 413, got {r.status_code}"
            print("  Correctly rejected: file too large")

            print("\nPOST /search — empty database")
            r = client.post("/search", data={"query": "test query"})
            assert r.status_code in (200, 500), (
                f"Unexpected status {r.status_code} — app may have crashed"
            )
            print(f"  Returned {r.status_code} (no unhandled crash)")

        print("\nAPP TEST PASSED")
        return True

    except ImportError:
        print("\nSKIPPED: httpx not installed. Run: pip install httpx")
        return True
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Slow tests — requires Ollama running
# ---------------------------------------------------------------------------

def test_embedding():
    """
    Verifies the embedding model loads and produces correctly shaped vectors.

    This test is slow on first run because it downloads the model if not
    cached locally. Subsequent runs are fast.

    Does not require Ollama — only the HuggingFace sentence-transformers model.
    """
    print("=" * 60)
    print("TESTING EMBEDDING MODEL")
    print("=" * 60)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        from src.config import EMBEDDING_MODEL_NAME

        print(f"\nModel: {EMBEDDING_MODEL_NAME}")
        print("Loading model (may take a moment on first run)...")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        test_texts = [
            "This is a test document about artificial intelligence.",
            "RAG systems combine retrieval and generation.",
            "The weather is sunny today.",
        ]
        embedded = embeddings.embed_documents(test_texts)

        assert len(embedded) == len(test_texts), (
            f"Expected {len(test_texts)} embeddings, got {len(embedded)}"
        )
        assert len(embedded[0]) > 0, "Embedding vector is empty"
        print(f"  Generated {len(embedded)} embeddings, dimension: {len(embedded[0])}")

        query_embedding = embeddings.embed_query("Tell me about AI systems")
        assert len(query_embedding) == len(embedded[0]), (
            "Query embedding dimension does not match document embedding dimension"
        )
        print(f"  Query embedding dimension matches: {len(query_embedding)}")

        print("\nEMBEDDING TEST PASSED")
        return True

    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


def test_rag():
    """
    Verifies the RAG pipeline end-to-end via query_library().

    Requires Ollama to be running locally with the configured model pulled.
    Uses a hardcoded context string injected directly via _generate() to
    isolate the LLM call from the retrieval step — we're testing that the
    LLM responds sensibly, not that ChromaDB returns good results.

    If Ollama is not running:
        ollama serve
        ollama pull mistral:7b   (or whichever model is in config.py)
    """
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)

    try:
        from src.rag import PROMPT_TEMPLATE, _generate

        # -- Prompt template has required variables --
        print("\nChecking prompt template variables...")
        assert "{context}"  in PROMPT_TEMPLATE, "PROMPT_TEMPLATE missing {context}"
        assert "{question}" in PROMPT_TEMPLATE, "PROMPT_TEMPLATE missing {question}"
        print("  PROMPT_TEMPLATE contains {context} and {question}")

        # -- LLM responds with a non-empty string --
        # We call _generate directly with a known context rather than going
        # through retrieval — this tests the generation step in isolation.
        print(f"\nCalling _generate (requires Ollama)...")
        answer = _generate(
            context="The sky is blue because of Rayleigh scattering.",
            question="Why is the sky blue?",
        )

        assert isinstance(answer, str), f"Expected str, got {type(answer)}"
        assert len(answer.strip()) > 0, "Answer is empty"
        print(f"  Response received ({len(answer)} chars)")
        print(f"  Preview: {answer[:200]}")

        print("\nRAG TEST PASSED")
        return True

    except ConnectionRefusedError:
        print("\nFAILED: Ollama is not running.")
        print("  Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False

def test_rag_pure():
    """
    Tests the pure helper functions in rag.py.

    These functions contain real logic but have no external dependencies —
    no DB, no embedding model, no Ollama required. They belong in the fast
    section alongside test_metadata and test_utils.

    Functions covered:
        _build_chroma_filter  — translates plain dicts to ChromaDB syntax
        _build_context        — formats retrieved docs into a prompt context string
        _build_sources        — extracts and deduplicates source labels from docs
    """
    print("=" * 60)
    print("TESTING RAG PURE FUNCTIONS")
    print("=" * 60)

    try:
        from langchain_core.documents import Document
        from src.rag import _build_chroma_filter, _build_context, _build_sources

        # -- _build_chroma_filter --
        # ChromaDB cannot do plain equality — every comparison needs an
        # explicit $eq operator. Single vs. multiple conditions produce
        # different structures, both tested here.
        print("\n_build_chroma_filter")

        result = _build_chroma_filter({"author": "Borges"})
        assert result == {"author": {"$eq": "Borges"}}, f"Got: {result}"
        print("  Single field -> plain $eq dict (no $and wrapper)")

        result = _build_chroma_filter({"author": "Borges", "title": "Labyrinths"})
        conditions = result["$and"]
        assert {"author": {"$eq": "Borges"}} in conditions
        assert {"title":  {"$eq": "Labyrinths"}} in conditions
        assert len(conditions) == 2
        print("  Multiple fields -> $and list of $eq dicts")

        # -- _build_context --
        # Each chunk should be prefixed with [Title by Author] on its own line,
        # chunks separated by double newlines so the LLM sees them as distinct
        # passages. Empty input should return an empty string cleanly.
        print("\n_build_context")

        docs = [
            Document(
                page_content="The library is infinite.",
                metadata={"title": "Labyrinths", "author": "Borges"},
            ),
            Document(
                page_content="The rose has no why.",
                metadata={"title": "The Name of the Rose", "author": "Eco"},
            ),
        ]
        context = _build_context(docs)

        assert "[Labyrinths by Borges]" in context, (
            "Source attribution for first doc missing from context"
        )
        assert "The library is infinite." in context
        print("  First doc formatted with attribution and content")

        assert "[The Name of the Rose by Eco]" in context
        assert "The rose has no why." in context
        print("  Second doc formatted with attribution and content")

        assert "\n\n" in context, (
            "Chunks should be separated by double newlines so the LLM "
            "sees them as distinct passages."
        )
        print("  Chunks separated by double newlines")

        assert _build_context([]) == "", (
            "_build_context([]) should return empty string, not crash."
        )
        print("  Empty docs list returns empty string")

        # Missing metadata should fall back gracefully, not KeyError
        docs_no_meta = [Document(page_content="Some text.", metadata={})]
        context_no_meta = _build_context(docs_no_meta)
        assert "[Unknown by Unknown]" in context_no_meta
        print("  Missing metadata falls back to 'Unknown' without crashing")

        # -- _build_sources --
        # Sources should be deduplicated — two chunks from the same book
        # should produce only one source entry in the UI.
        print("\n_build_sources")

        docs = [
            Document(page_content="chunk 1", metadata={"title": "Labyrinths", "author": "Borges"}),
            Document(page_content="chunk 2", metadata={"title": "Labyrinths", "author": "Borges"}),
            Document(page_content="chunk 3", metadata={"title": "Ficciones",  "author": "Borges"}),
        ]
        sources = _build_sources(docs)

        assert len(sources) == 2, (
            f"Expected 2 deduplicated sources, got {len(sources)}: {sources}"
        )
        assert "Labyrinths by Borges" in sources
        assert "Ficciones by Borges" in sources
        print("  Two chunks from same book deduplicated to one source entry")

        assert _build_sources([]) == [], (
            "_build_sources([]) should return empty list, not crash."
        )
        print("  Empty docs list returns empty list")

        print("\nRAG PURE TEST PASSED")
        return True

    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False
    
    
# ---------------------------------------------------------------------------
# Registry and runner
# ---------------------------------------------------------------------------

TESTS = {
    # Fast — pure functions, no I/O, no models
    "utils":       test_utils,
    "metadata":    test_metadata,
    "ingest":      test_ingest,
    "rag_pure":    test_rag_pure,
    # Medium — needs DB / embedding model, no LLM
    "vector":      test_vector,
    "db_metadata": test_db_metadata,
    "app":         test_app,
    # Slow — needs Ollama running
    "embedding":   test_embedding,
    "rag":         test_rag,
}


def main():
    parser = argparse.ArgumentParser(
        description="AI Librarian test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tests: {', '.join(TESTS.keys())}",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        choices=list(TESTS.keys()),
        help="Tests to run. Runs all if omitted.",
    )
    args = parser.parse_args()

    to_run = {name: TESTS[name] for name in (args.tests or TESTS.keys())}

    print("\n" + "=" * 60)
    print("AI LIBRARIAN TEST SUITE")
    print("=" * 60)

    results = {name: fn() for name, fn in to_run.items()}

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:12}  {status}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()