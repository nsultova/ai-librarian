"""
ingest.py — document loading, cleaning, annotation, and chunking.

Pipeline (called from app.py via chunk_file):

    file on disk
        -> _load_documents()      load PDF or EPUB into raw Document list
        -> _annotate_documents()  attach file metadata + chapter labels to each doc
        -> _split_documents()     split docs into fixed-size overlapping chunks

The output is a list of LangChain Document objects ready to be embedded
and stored in ChromaDB by vector.add_documents_to_db().

Chunking strategy:
    Each chunk is CHUNK_SIZE characters with CHUNK_OVERLAP characters of
    overlap with its neighbors. Overlap ensures that sentences spanning a
    chunk boundary appear in both chunks — so a query matching that sentence
    will retrieve at least one chunk containing it in (hopefully) full context.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.metadata import (
    DocumentMetadata,
    extract_epub_chapter_title,
    get_file_metadata,
    sanitize_spine_name,
    chapter_for_page,
)
from src.utils import clean_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal pipeline steps
# ---------------------------------------------------------------------------

def _load_documents(file_path: str) -> List[Document]:
    """
    Load a PDF or EPUB file into a list of raw Document objects.

    PDF:  PyPDFLoader produces one Document per page. Each document's
          metadata includes 'page' (0-indexed int), set by LangChain.

    EPUB: load_epub() produces one Document per spine item (chapter).
          Each document's metadata includes 'chapter' (str), set by us.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        logger.info("Loading PDF...")
        return PyPDFLoader(file_path).load()

    if ext == ".epub":
        logger.info("Loading EPUB...")
        return _load_epub(file_path)

    raise ValueError(
        f"Unsupported file format: '{ext}'. Supported formats: .pdf, .epub"
    )


def _load_epub(file_path: str) -> List[Document]:
    """
    Load an EPUB using ebooklib + BeautifulSoup.

    EPUB files are ZIP archives containing HTML spine items — one per
    chapter or section. We iterate the spine in order, parse each item's
    HTML, extract a human-readable chapter title, and convert to plain text.

    Nav/TOC items (containing a <nav> tag) are skipped — they are structural
    metadata, not readable content.

    Chapter title resolution order:
        1. First heading tag found (h1 > h2 > h3 > h4)
        2. <title> tag of the HTML item
        3. Sanitized internal spine filename (e.g. 'Text/ch03.xhtml' -> 'ch03')
    """
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(file_path)
    docs = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        if soup.find("nav"):
            continue  # skip structural nav/TOC items

        chapter_title = (
            extract_epub_chapter_title(soup)
            or sanitize_spine_name(item.get_name())
        )

        text = clean_text(soup.get_text(separator="\n"))
        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source":  file_path,
                "chapter": chapter_title,
            },
        ))

    logger.info(f"Loaded {len(docs)} chapters from EPUB")
    return docs


def _annotate_documents(
    docs: List[Document],
    meta: DocumentMetadata,
) -> List[Document]:
    """
    Stamp each Document with file-level metadata and chapter labels.

    Two things happen here:

    1. Scalar metadata (title, author, file_type, etc.) from meta.to_chunk_metadata()
       is written to every document. This is what is needed for ChromaDB filtering to work —
       every chunk carries its parents documents metadata so queries can be scoped by author,
       title, chapter, etc.

    2. For PDFs: chapter labels are assigned using the outline's step-function map
       (page_chapter_map). PyPDFLoader sets doc.metadata['page'] as a 0-indexed int.
       chapter_for_page() walks the map to find which chapter that page belongs to.
       EPUBs already have chapter set in _load_epub(), so this step is unnecessary for them.

    chunk_index is set to preserve the original document order after splitting,
    since RecursiveCharacterTextSplitter does not guarantee order across docs.
    """
    chunk_metadata = meta.to_chunk_metadata()

    for idx, doc in enumerate(docs):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update(chunk_metadata)
        doc.metadata["chunk_index"] = idx

        if meta.page_chapter_map and "page" in doc.metadata:
            chapter = chapter_for_page(
                int(doc.metadata["page"]),
                meta.page_chapter_map,
            )
            if chapter:
                doc.metadata["chapter"] = chapter

    logger.info(f"Annotated {len(docs)} documents with metadata")
    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    """
    Split annotated documents into fixed-size overlapping chunks.

    RecursiveCharacterTextSplitter tries each separator in order,
    preferring to split at paragraph breaks before sentence breaks
    before word breaks — preserving as much semantic unit integrity
    as possible within the chunk size constraint.

    Separator priority:
        "\\n\\n"  paragraph break  (most preferred)
        "\\n"     line break
        ". "      sentence end
        " "       word boundary
        ""        character boundary (last resort)

    Overlap (CHUNK_OVERLAP chars) ensures context continuity — sentences
    near chunk boundaries appear in both neighboring chunks, so a query
    matching that sentence will retrieve a chunk with full surrounding context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        f"Split {len(docs)} documents into {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_file(file_path: str) -> List[Document]:
    """
    Full ingestion pipeline for a single PDF or EPUB file.

    Orchestrates three steps:
        1. Load   — read file into raw Document list
        2. Annotate — clean text, attach metadata and chapter labels
        3. Split  — chunk into overlapping fixed-size segments

    Returns a list of Document objects ready to be embedded and stored
    in ChromaDB by vector.add_documents_to_db().

    Raises:
        ValueError if the file format is not supported.
        Any exception from the underlying loaders propagates — callers
        (app.py) are responsible for handling ingestion errors.
    """
    logger.info(f"Starting ingestion: {file_path}")

    meta   = get_file_metadata(file_path)
    logger.info(f"Metadata: title='{meta.title}', author='{meta.author}', type={meta.file_type}")

    docs   = _load_documents(file_path)
    docs   = _annotate_documents(docs, meta)
    chunks = _split_documents(docs)

    logger.info(f"Ingestion complete: {len(chunks)} chunks from '{meta.title}'")
    return chunks