from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import logging

import PyPDF2
from ebooklib import epub

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

def _sanitize(value: Any, max_length: int = 512, fallback: str = "") -> str:
    """
    Coerce to string, strip whitespace and control characters, truncate.
    Returns fallback if result is empty.
    """
    if value is None:
        return fallback
    text = str(value)
    # Remove control characters (keep newlines only if intentional — strip here)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:max_length]
    return text or fallback


def _sanitize_list(items: List[Any], max_length: int = 256) -> List[str]:
    return [s for s in (_sanitize(i, max_length) for i in items) if s]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _walk_pdf_outline(outline: list) -> List[str]:
    """
    Recursively walk the PDF outline/bookmark tree and collect titles.
    PyPDF2 outline entries are either Destination objects or nested lists.
    """
    titles = []
    for item in outline:
        if isinstance(item, list):
            titles.extend(_walk_pdf_outline(item))
        else:
            try:
                title = _sanitize(item.title, max_length=256)
                if title:
                    titles.append(title)
            except AttributeError:
                pass
    return titles


def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata and chapter outline from a PDF file."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata or {}

            title = _sanitize(info.get("/Title"), fallback=Path(file_path).stem)
            author = _sanitize(info.get("/Author"), fallback="Unknown")

            chapters = _walk_pdf_outline(reader.outline) if reader.outline else []

            return {
                "title": title,
                "author": author,
                "page_count": len(reader.pages),
                "chapters": chapters,        # list of chapter title strings
                "has_outline": bool(chapters),
            }
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata from {file_path}: {e}")
        return _fallback_metadata(file_path)


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

def _epub_first(book: epub.EpubBook, dc_tag: str, fallback: str = "") -> str:
    items = book.get_metadata("DC", dc_tag)
    if items:
        return _sanitize(items[0][0], fallback=fallback)
    return fallback


def extract_epub_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from an EPUB file."""
    try:
        book = epub.read_epub(file_path)

        return {
            "title": _epub_first(book, "title", fallback=Path(file_path).stem),
            "author": _epub_first(book, "creator", fallback="Unknown"),
            "language": _epub_first(book, "language", fallback="unknown"),
            "publisher": _epub_first(book, "publisher", fallback=""),
            "publication_date": _epub_first(book, "date", fallback=""),
        }
    except Exception as e:
        logger.warning(f"Failed to extract EPUB metadata from {file_path}: {e}")
        return _fallback_metadata(file_path)


def extract_epub_chapter_title(soup) -> Optional[str]:
    """
    Extract a human-readable chapter title from a parsed EPUB spine item.
    Tries headings first, then <title> tag, then sanitizes the filename fallback.
    Returns None if nothing useful is found — caller should fall back to spine name.
    """
    for tag in ("h1", "h2", "h3", "h4"):
        heading = soup.find(tag)
        if heading:
            title = _sanitize(heading.get_text(), max_length=256)
            if title:
                return title

    title_tag = soup.find("title")
    if title_tag:
        title = _sanitize(title_tag.get_text(), max_length=256)
        if title:
            return title

    return None


def _sanitize_spine_name(raw: str) -> str:
    """
    Turn an internal spine path like 'Text/part01_chapter03.xhtml'
    into something readable like 'part01 chapter03'.
    """
    name = Path(raw).stem                        # drop extension
    name = re.sub(r"[_\-/\\]+", " ", name)      # separators -> spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name or raw


# ---------------------------------------------------------------------------
# Fallback + entry point
# ---------------------------------------------------------------------------

def _fallback_metadata(file_path: str) -> Dict[str, Any]:
    return {
        "title": Path(file_path).stem,
        "author": "Unknown",
        "source_file": Path(file_path).name,
    }


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Main entry point. Returns a flat metadata dict for a given file."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        metadata = extract_pdf_metadata(file_path)
    elif ext == ".epub":
        metadata = extract_epub_metadata(file_path)
    else:
        metadata = _fallback_metadata(file_path)

    # Common fields always present
    metadata["file_type"] = ext.lstrip(".")
    metadata["source_file"] = Path(file_path).name

    return metadata