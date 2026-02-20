"""
metadata.py — document metadata extraction and structured representation.

Handles PDF (via PyPDF2) and EPUB (via ebooklib) formats.
Provides DocumentMetadata, a dataclass that separates ChromaDB-safe scalar
fields from derived fields used only during ingestion.

Public interface:
    get_file_metadata(file_path)      -> DocumentMetadata
    extract_epub_chapter_title(soup)  -> Optional[str]
    sanitize_spine_name(raw)          -> str
    chapter_for_page(idx, map)        -> str
    build_page_chapter_map(entries)   -> Dict[int, str]
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2
from ebooklib import epub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """
    Structured metadata extracted from an ingested document.

    Separates two categories of fields:
        Scalar fields  — safe to store per-chunk in ChromaDB (str, int, float).
                         These become filterable metadata on every chunk.
        Derived fields — used during ingestion processing but NOT stored
                         per-chunk. chapters is used by /library aggregation
                         (reconstructed from chunk metadata at query time).
                         page_chapter_map is used to annotate PDF chunks
                         during ingest, then discarded.

    to_chunk_metadata() makes the ChromaDB-safe contract explicit, replacing
    the previous pattern of popping non-scalar keys from a plain dict —
    which was fragile because adding a new field to get_file_metadata()
    could silently break ingestion.
    """
    # -- Scalar: stored on every chunk in ChromaDB --
    title:       str
    author:      str
    file_type:   str
    source_file: str
    page_count:  Optional[int] = None

    # -- Derived: used during ingest, not stored per-chunk --
    chapters:         List[str]       = field(default_factory=list)
    page_chapter_map: Dict[int, str]  = field(default_factory=dict)

    def to_chunk_metadata(self) -> Dict:
        """
        Return only the scalar fields safe for ChromaDB chunk metadata.
        Called once per document during ingestion to stamp every chunk.
        """
        return {
            "title":       self.title,
            "author":      self.author,
            "file_type":   self.file_type,
            "source_file": self.source_file,
            "page_count":  self.page_count,
        }



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


def sanitize_spine_name(raw: str) -> str:
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
    """
    Minimal metadata for files where extraction failed or format is unsupported.
    """
    return {
        "title":    Path(file_path).stem,
        "author":   "Unknown",
        "page_count": None,
        "chapters": [],
        "page_chapter_map": {},
    }
    
def get_file_metadata(file_path: str) -> DocumentMetadata:
    """Main entry point. Returns structured metadata for a given file."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        raw = extract_pdf_metadata(file_path)
    elif ext == ".epub":
        raw = extract_epub_metadata(file_path)
    else:
        raw = _fallback_metadata(file_path)

    return DocumentMetadata(
        title=raw.get("title", Path(file_path).stem),
        author=raw.get("author", "Unknown"),
        file_type=ext.lstrip("."),
        source_file=Path(file_path).name,
        page_count=raw.get("page_count"),
        chapters=raw.get("chapters", []),
        page_chapter_map=raw.get("page_chapter_map", {}),
    )