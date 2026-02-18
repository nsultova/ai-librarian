from pathlib import Path
from typing import Dict, Any
import PyPDF2
from ebooklib import epub
import logging

logger = logging.getLogger(__name__)


"""
Test extraction - super helpful to get a sanity check first

python -c "from src.metadata import get_file_metadata; import pprint; pprint.pprint(get_file_metadata('data/uploads/some_book.pdf'))" 
"""

def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from PDF file."""
    try:
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            info = pdf.metadata or {}
            
            return {
                'title': info.get('/Title', '').strip() or Path(file_path).stem,
                'author': info.get('/Author', '').strip() or 'Unknown',
                'page_count': len(pdf.pages)
            }
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")
        return _fallback_metadata(file_path)

def extract_epub_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from EPUB file."""
    try:
        book = epub.read_epub(file_path)
        
        return {
            'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else Path(file_path).stem,
            'author': book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else 'Unknown',
            'language': book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else 'unknown',
            'publisher': book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else '',
            'publication_date': book.get_metadata('DC', 'date')[0][0] if book.get_metadata('DC', 'date') else '',
        }
    except Exception as e:
        logger.warning(f"Failed to extract EPUB metadata: {e}")
        return _fallback_metadata(file_path)

def _fallback_metadata(file_path: str) -> Dict[str, Any]:
    """Fallback metadata when extraction fails."""
    return {
        'title': Path(file_path).stem,
        'author': 'Unknown',
        'source_file': Path(file_path).name
    }

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Main entry point for metadata extraction."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        metadata = extract_pdf_metadata(file_path)
    elif file_ext == '.epub':
        metadata = extract_epub_metadata(file_path)
    else:
        metadata = _fallback_metadata(file_path)
    
    # Add common fields
    metadata['file_type'] = file_ext[1:]  # remove dot
    metadata['source_file'] = Path(file_path).name
    
    return metadata