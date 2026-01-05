import os
from pypdf import PdfReader
from ebooklib import epub
import warnings


### TODO: introduce later
"""
AI Cleanup. pass the "messy" filename to a small LLM in metadata.py 
to format it cleanly (e.g., convert tolkien_lotr_1_final.pdf -> Lord of the Rings: Fellowship of the Ring).
"""

# suppress ebooklib warnings about future XML parsing
warnings.filterwarnings("ignore", category=UserWarning)

def get_metadata(file_path: str) -> dict:
    file_ext = file_path.split('.')[-1].lower()
    filename = os.path.basename(file_path)
    
    meta = {
        "title": filename,
        "author": "unknown",
        "source": filename
    }

    try:
        if file_ext == 'pdf':
            meta.update(_extract_pdf_meta(file_path))
        elif file_ext == 'epub':
            meta.update(_extract_epub_meta(file_path))
    except Exception as e:
        print(f"Could not extract metadata from {filename}: {e}")
    
    return meta

def _extract_pdf_meta(file_path: str) -> dict:
    reader = PdfReader(file_path)
    info = reader.metadata
    
    data = {}
    if info:
        # PDF metadata keys can be inconsistent, check for standard ones
        if info.title: data['title'] = info.title
        if info.author: data['author'] = info.author
    
    return data

def _extract_epub_meta(file_path: str) -> dict:
    book = epub.read_epub(file_path)
    data = {}
    
    # ebooklib returns lists of tuples, we need the first item
    # standard dublin core keys: 'title', 'creator' (author)
    titles = book.get_metadata('DC', 'title')
    if titles:
        data['title'] = titles[0][0]
        
    creators = book.get_metadata('DC', 'creator')
    if creators:
        data['author'] = creators[0][0]
        
    return data