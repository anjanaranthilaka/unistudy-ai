"""
utils/text_processing.py

Utilities for tokenizing, cleaning, and chunking documents
with configurable size and overlap while preserving source metadata.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Normalizes excessive whitespace and line breaks."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_document(
    document_data: Dict[str, Any],
    chunk_size_words: int = 300,
    chunk_overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits document pages into overlapping text chunks with source metadata.

    Args:
        document_data: The dictionary returned by `read_document()`.
        chunk_size_words: Target word count per chunk.
        chunk_overlap_words: Overlapping word count between consecutive chunks.

    Returns:
        List of chunk dictionaries:
        [
            {
                "chunk_id": int,
                "filename": str,
                "page_num": int,
                "text": str,
                "word_count": int
            },
            ...
        ]
    """
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be strictly smaller than chunk_size_words")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    filename = document_data.get("filename", "unknown_document")
    pages = document_data.get("pages", [])

    for page in pages:
        page_num = page.get("page_num", 1)
        raw_text = clean_text(page.get("text", ""))
        if not raw_text:
            continue

        words = raw_text.split()
        if not words:
            continue

        start_idx = 0
        total_words = len(words)

        while start_idx < total_words:
            end_idx = min(start_idx + chunk_size_words, total_words)
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_counter,
                "filename": filename,
                "page_num": page_num,
                "text": chunk_text,
                "word_count": len(chunk_words)
            })
            chunk_counter += 1

            if end_idx == total_words:
                break

            # Advance by step size (size - overlap)
            start_idx += (chunk_size_words - chunk_overlap_words)

    return chunks