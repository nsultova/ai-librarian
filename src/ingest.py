import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.metadata import get_file_metadata, extract_epub_chapter_title, _sanitize_spine_name
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_txt(txt: str) -> str:
    # Rejoin hyphenated line-breaks: impor-\ntant -> important
    txt = re.sub(r"-\n", "", txt)
    # Collapse whitespace
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def load_epub(file_path: str) -> List[Document]:
    """
    Load an EPUB using ebooklib + BeautifulSoup.
    Each spine item (chapter) becomes one Document.
    Chapter title is extracted from HTML headings where possible,
    falling back to a sanitized version of the internal spine filename.
    """
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(file_path)
    docs = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Drop nav/TOC items — structural, not content
        if soup.find("nav"):
            continue

        # Prefer a heading-derived title; fall back to sanitized spine name
        chapter_title = extract_epub_chapter_title(soup)
        if not chapter_title:
            chapter_title = _sanitize_spine_name(item.get_name())

        text = soup.get_text(separator="\n")
        text = clean_txt(text)
        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source": file_path,
                "chapter": chapter_title,
            },
        ))

    logger.info(f"Loaded {len(docs)} chapters from EPUB")
    return docs


def chunk_file(file_path: str) -> List[Document]:
    logger.info(f"Starting ingestion: {file_path}")
    ext = file_path.rsplit(".", 1)[-1].lower()

    logger.info(f"Loading {ext.upper()} file...")
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
    elif ext == "epub":
        raw_docs = load_epub(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info("Extracting metadata...")
    file_metadata = get_file_metadata(file_path)
    logger.info(f"Extracted metadata: {file_metadata}")

    # chapters from PDF outline are stored at file level, not per-chunk
    # We pop it here to avoid storing a list in ChromaDB (it only accepts scalar values)
    # Extract chapters list → store separately in memory
    # Remove it from metadata → keep only scalars for ChromaDB
    # If the key doesn't exist (e.g., EPUB file), use the fallback ([] or None)
    
    pdf_chapters = file_metadata.pop("chapters", [])
    file_metadata.pop("has_outline", None)

    logger.info("Cleaning text...")
    for idx, doc in enumerate(raw_docs):
        doc.page_content = clean_txt(doc.page_content)
        doc.metadata.update(file_metadata)
        doc.metadata["chunk_index"] = idx

        # For PDFs: annotate chunk with nearest chapter from outline if available
        # (page_number comes from PyPDFLoader automatically as metadata['page'])

    logger.info(f"Cleaned {len(raw_docs)} sections")

    logger.info(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)
    logger.info(f"Created {len(chunks)} chunks from {file_metadata.get('title', 'Unknown')}")

    return chunks