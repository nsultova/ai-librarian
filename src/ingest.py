import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.metadata import get_file_metadata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_txt(txt:str) -> str:
    
    # remove things like impor-\ntant
    txt = re.sub(r'-\n', '', txt)
    # replace multiple newlines/tabs with single space
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def chunk_file(file_path:str) -> List[Document]:
    
    file_ext = file_path.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_ext == 'epub':
        loader = UnstructuredEPubLoader(file_path)
    else:
        raise ValueError(f"unsupported file format (for now): {file_ext}")
    
    raw_docs = loader.load()
    
    # meta_info = get_file_metadata(file_path)
    # print(f"Detected Metadata: {meta_info}")
        # Extract metadata once per file
    file_metadata = get_file_metadata(file_path)
    logger.info(f"Extracted metadata: {file_metadata}")
    
    # clean
    for idx, doc in enumerate(raw_docs):
        doc.page_content = clean_txt(doc.page_content)
        doc.metadata.update(file_metadata)
        doc.metadata['chunk_index'] = idx  # Track position
        # metadata for later retrieval visibility
        # doc.metadata['source'] = file_path.split('/')[-1]
        # doc.metadata.update(meta_info)
        
    # chunkchunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, # overlap ensures context isn't lost at cut-offs
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_docs)
    
    print(f"Processed {len(chunks)} chunks from {file_path}")
    # print(f"Processed {len(chunks)} chunks from {meta_info['title']}")
    return chunks
        
    

