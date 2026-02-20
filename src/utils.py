"""
utils.py — shared text-processing utilities.

Functions:
    clean_text(text) — normalize raw extracted text for embedding.

Kept separate from ingest.py so these helpers can be used by any module
without pulling in ingestion dependencies.
"""

import re


def clean_text(text: str) -> str:
    """
    Normalize raw extracted text for embedding.

    Two passes:
        1. Rejoin hyphenated line-breaks that PDF extraction introduces.
           e.g. "impor-\\ntant" -> "important"
           These are soft hyphens added by typesetters for line wrapping —
           they are not part of the word and break semantic similarity.

        2. Collapse all remaining whitespace (spaces, tabs, newlines)
           into single spaces. Embedding models treat whitespace tokens
           as noise; compacting them improves chunk quality.
    """
    text = re.sub(r"-\n", "", text)       # rejoin hyphenated line-breaks
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text.strip()