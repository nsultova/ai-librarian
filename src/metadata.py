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

def _walk_pdf_outline(outline: list, reader: "PyPDF2.PdfReader") -> List[tuple]:
    """
    Recursively walk the PDF outline/bookmark tree.
    Returns list of (page_index, title) tuples, sorted by page order.
    PyPDF2 outline entries are either Destination objects or nested lists.
    """
    entries = []
    for item in outline:
        if isinstance(item, list):
            entries.extend(_walk_pdf_outline(item, reader))
        else:
            try:
                title = _sanitize(item.title, max_length=256)
                if not title:
                    continue
                page_idx = reader.get_destination_page_number(item)
                entries.append((page_idx, title))
            except Exception:
                pass
    return entries


def build_page_chapter_map(outline_entries: List[tuple]) -> Dict[int, str]:
    """
    Given sorted (page_idx, title) pairs, build a map of:
        page_number -> chapter title

    Each page belongs to the chapter whose bookmark is at or before that page.
    This is a step function — chapter changes at each bookmark's page.
    """
    if not outline_entries:
        return {}

    sorted_entries = sorted(outline_entries, key=lambda x: x[0])
    return {page: title for page, title in sorted_entries}


def chapter_for_page(page_idx: int, page_chapter_map: Dict[int, str]) -> str:
    """
    Return the chapter title for a given page index using the step-function map.
    Finds the largest bookmark page_idx <= page_idx.
    """
    if not page_chapter_map:
        return ""
    # Walk backwards through sorted keys to find the last chapter start <= page
    for bookmark_page in sorted(page_chapter_map.keys(), reverse=True):
        if page_idx >= bookmark_page:
            return page_chapter_map[bookmark_page]
    return ""


def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata and chapter outline from a PDF file."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata or {}

            title = _sanitize(info.get("/Title"), fallback=Path(file_path).stem)
            author = _sanitize(info.get("/Author"), fallback="Unknown")

            outline_entries = _walk_pdf_outline(reader.outline, reader) if reader.outline else []
            chapter_titles = [t for _, t in sorted(outline_entries, key=lambda x: x[0])]
            page_chapter_map = build_page_chapter_map(outline_entries)

            return {
                "title": title,
                "author": author,
                "page_count": len(reader.pages),
                "chapters": chapter_titles,          # ordered list for /library
                "page_chapter_map": page_chapter_map, # page_idx -> chapter, used at ingest
                "has_outline": bool(outline_entries),
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
