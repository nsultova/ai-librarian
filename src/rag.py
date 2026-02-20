"""
rag.py — Retrieval-Augmented Generation pipeline.

The RAG pattern in three steps:


    1. Retrieve  — embed the user's question and find the most semantically
                   similar chunks in ChromaDB (vector similarity search).

    2. Augment   — format the retrieved chunks into a context string and
                   inject it into the prompt alongside the question.
                   The prompt instructs the model to base its answer on
                   this context — but this is a behavioral nudge, not a
                   technical constraint. The model still uses its training
                   knowledge for language and reasoning; the instruction
                   steers it toward the retrieved content for factual claims.

    3. Generate  — pass the augmented prompt to the LLM and return its answer.


This separation is what makes RAG useful: the retrieval step grounds the
LLM's response in your actual documents, reducing hallucination and making
answers verifiable via the returned source list.

Note on halluciantions:
    When relevant content is in the context window, the model is statistically
    more likely to draw from it than to confabulate. It does not eliminate
    hallucination — the model can still ignore or misread the context — which
    is why sources are returned alongside the answer for user verification.

Public interface:
    query_library(question, filters) -> (answer, sources)
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, RETRIEVAL_K
from src.vector import get_vector_db


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# The prompt template is parsed once at module load rather than on every
# query. ChatPromptTemplate.from_template parses the template string —
# doing this repeatedly is unnecessary work.
PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}

If you cannot find the answer in the context, say "I couldn't find that in your library."
Answer concisely and cite relevant details from the context where useful.
"""

_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ChatOllama is a lightweight HTTP wrapper around the local Ollama server.
# Instantiated once here rather than on every query call.
_llm = ChatOllama(model=LLM_MODEL)

# LangChain chain: prompt -> LLM -> parse output to plain string.
# The | operator composes Runnables — each step's output becomes the next
# step's input. StrOutputParser extracts the text content from the LLM's
# response object.
_chain = _prompt | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Internal pipeline steps
# ---------------------------------------------------------------------------

def _build_chroma_filter(filters: Dict[str, Any]) -> Dict:
    """
    Translate a plain {field: value} dict into ChromaDB's filter syntax.

    ChromaDB requires explicit comparison operators — it does not support
    plain equality like {"author": "Borges"}. Every comparison must use
    an operator like $eq, $gt, $in, etc.

    Single field:    {"author": "Borges"}
                  -> {"author": {"$eq": "Borges"}}

    Multiple fields: {"author": "Borges", "title": "Labyrinths"}
                  -> {"$and": [{"author": {"$eq": "Borges"}},
                               {"title":  {"$eq": "Labyrinths"}}]}

    The $and wrapper is only added for multiple conditions — ChromaDB
    rejects $and with a single-element list.
    """
    conditions = [{k: {"$eq": v}} for k, v in filters.items()]

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _retrieve(question: str, filters: Optional[Dict[str, Any]]) -> List[Document]:
    """
    Embed the question and retrieve the most similar chunks from ChromaDB.

    The question is converted to a vector using the same embedding model
    that was used during ingestion. ChromaDB then finds the k nearest
    vectors by cosine similarity — these are the chunks most likely to
    contain a relevant answer.

    filters, if provided, narrow the search to chunks whose metadata
    matches all filter conditions. This is how title/author/chapter
    scoping works — the similarity search only considers matching chunks.

    Returns up to RETRIEVAL_K documents (configured in config.py).
    """
    search_kwargs: Dict[str, Any] = {"k": RETRIEVAL_K}
    if filters:
        search_kwargs["filter"] = _build_chroma_filter(filters)

    retriever = get_vector_db().as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    return retriever.invoke(question)


def _build_context(docs: List[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Each chunk is prefixed with its source attribution in the form:
        [Title by Author]
        chunk text...

    Injecting attribution inline means the LLM can reference the source
    when composing its answer ("According to X in Y..."), and helps when
    chunks from multiple books are retrieved together — the model can
    distinguish which content came from where.

    Chunks are separated by double newlines so the LLM sees them as
    distinct passages rather than one continuous block of text.
    """
    parts = []
    for doc in docs:
        meta = doc.metadata
        title  = meta.get("title",  "Unknown")
        author = meta.get("author", "Unknown")
        parts.append(f"[{title} by {author}]\n{doc.page_content}")

    return "\n\n".join(parts)


def _build_sources(docs: List[Document]) -> List[str]:
    """
    Extract a deduplicated list of source labels from retrieved chunks.

    Used to show the user which books contributed to the answer.
    Deduplication via set: multiple chunks from the same book produce
    only one source entry in the UI.
    """
    return list({
        f"{d.metadata.get('title', 'Unknown')} by {d.metadata.get('author', 'Unknown')}"
        for d in docs
    })


def _generate(context: str, question: str) -> str:
    """
    Invoke the LLM chain with the augmented context and question.

    The chain is: prompt -> LLM -> StrOutputParser.
    The prompt instructs the model to answer only from the provided context,
    which grounds the response in the retrieved documents rather than the
    model's general training knowledge.
    """
    return _chain.invoke({"context": context, "question": question})


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def query_library(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """
    Run the full RAG pipeline for a user question.

    Steps:
        1. Retrieve  — find the most relevant chunks via similarity search
        2. Augment   — format chunks into a context string with attribution
        3. Generate  — call the LLM with the augmented prompt

    Args:
        question: The user's natural language question.
        filters:  Optional metadata filters to scope the search.
                  Keys must match chunk metadata fields (title, author,
                  chapter, file_type). All conditions are ANDed together.

    Returns:
        (answer, sources) where answer is the LLM's response string and
        sources is a deduplicated list of "Title by Author" strings.
    """
    docs    = _retrieve(question, filters)
    context = _build_context(docs)
    sources = _build_sources(docs)
    answer  = _generate(context, question)

    return answer, sources