from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.vector import get_vector_db
from src.config import LLM_MODEL, RETRIEVAL_K

PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}

If you cannot find the answer in the context, say "I couldn't find that in your library."
Answer concisely and cite relevant details from the context where useful.
"""

# # rag.py — single retrieval pass
# def query_library(question: str) -> Tuple[str, List[str]]:
#     db = get_vector_db()  # retrieve once
#     retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
    
#     docs: List[Document] = retriever.invoke(question)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     sources = list({d.metadata.get("source", "Unknown") for d in docs})
def query_library(question: str, filters: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    db = get_vector_db()
    
    search_kwargs: Dict[str, Any] = {"k": RETRIEVAL_K}
    if filters:
        search_kwargs["filter"] = filters
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    docs: List[Document] = retriever.invoke(question)
    
    # Enhanced context with metadata
    context_parts = []
    for doc in docs:
        meta = doc.metadata
        source_info = f"[{meta.get('title', 'Unknown')} by {meta.get('author', 'Unknown')}]"
        context_parts.append(f"{source_info}\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    sources = list({f"{d.metadata.get('title', 'Unknown')} by {d.metadata.get('author', 'Unknown')}" for d in docs})
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model=LLM_MODEL)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, sources




# def query_library(question: str) -> Tuple[str, List[str]]:
#     db = get_vector_db()
    
#     # .as_retriever() returns a VectorStoreRetriever which is a Runnable
#     retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

  
#     template = """Answer the question based only on the following context:
#     {context}

#     Question: {question}
    
#     If you don't know the answer based on the context, say "I couldn't find that in your library."
#     """
#     prompt = ChatPromptTemplate.from_template(template)
    
#     llm = ChatOllama(model=LLM_MODEL)

#     def format_docs(docs: List[Document]) -> str:
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     # retrieve docs separately first to get the src for UI
#     docs: List[Document] = retriever.invoke(question)
    
#     sources: List[str] = list({d.metadata.get('source', 'Unknown') for d in docs})
    
#     answer: str = rag_chain.invoke(question)
    
#     return answer, sources