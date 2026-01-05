import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_txt(txt:str) -> str:
    
    # remove things like impor-\ntant
    txt = re.sub(r'-\n', '', txt)
    # replase multiple newlines/tabs with single space
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def chunk_file(file_path:str) -> str:
    
    file_ext = file_path.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_ext == 'epub':
        loader = UnstructuredEPubLoader(file_path)
    else:
        raise ValueError(f"unsupported file format (for now): {file_ext}")
    
    raw_docs = loader.load()
    
    
    # clean
    for doc in raw_docs:
        doc.page_content = clean_txt(doc.page_content)
        # metadata for later retrieval visibility
        doc.metadata['source'] = file_path.split('/')[-1]
        
    # chunkchunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # overlap ensures context isn't lost at cut-offs
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_docs)
    
    print(f"Processed {len(chunks)} chunks from {file_path}")
    return chunks
        
    

