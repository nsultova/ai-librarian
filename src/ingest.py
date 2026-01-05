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
    pass


