"""
tools/document_search.py

In-memory keyword retrieval engine for UniStudy AI.
Provides fast, transparent ranking of text chunks based on query terms.
"""

import re
import math
from typing import List, Dict, Any


def tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer with stopword filtering."""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with"
    }
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in raw_tokens if t not in stopwords and len(t) > 1]


def search_documents(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Scores and ranks chunks against a query using term-frequency matching.

    Args:
        query: User search query or question.
        chunks: List of chunk dictionaries created by `chunk_document()`.
        top_k: Maximum number of relevant chunks to return.

    Returns:
        List of matching chunk dictionaries sorted by descending relevance score,
        with an added 'score' field.
    """
    if not query.strip() or not chunks:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        # Fallback to raw split if all terms were common words
        query_tokens = [t.lower() for t in query.split() if len(t) > 1]

    scored_results = []

    for chunk in chunks:
        chunk_text = chunk["text"]
        chunk_tokens = tokenize(chunk_text)
        if not chunk_tokens:
            continue

        chunk_token_count = len(chunk_tokens)
        score = 0.0

        # Term frequency scoring with query term density weighting
        for q_term in set(query_tokens):
            match_count = chunk_tokens.count(q_term)
            if match_count > 0:
                # Term frequency normalized by chunk length
                tf = match_count / math.sqrt(chunk_token_count)
                score += tf * 2.0

        # Phrase bonus: direct exact-phrase matches gain a boost
        clean_q = query.strip().lower()
        if len(clean_q) > 3 and clean_q in chunk_text.lower():
            score += 3.0

        if score > 0.0:
            result_item = dict(chunk)
            result_item["score"] = round(score, 3)
            scored_results.append(result_item)

    # Sort descending by score
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]