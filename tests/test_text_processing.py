"""
tests/test_text_processing.py
"""

import pytest
from utils.text_processing import clean_text, chunk_document


def test_clean_text():
    raw = "Line 1\r\nLine 2\n\n\n\nLine 3   with   spaces"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "   " not in cleaned


def test_chunk_document_basic():
    doc_data = {
        "filename": "test.txt",
        "pages": [
            {
                "page_num": 1,
                "text": " ".join([f"word{i}" for i in range(100)])
            }
        ]
    }

    # 40-word chunks with 10-word overlap
    chunks = chunk_document(doc_data, chunk_size_words=40, chunk_overlap_words=10)

    assert len(chunks) > 1
    assert chunks[0]["page_num"] == 1
    assert chunks[0]["filename"] == "test.txt"
    assert chunks[0]["word_count"] == 40
    # Next chunk starts with overlap
    assert "word30" in chunks[1]["text"]


def test_invalid_overlap_raises():
    doc_data = {"filename": "test.txt", "pages": [{"page_num": 1, "text": "hello world"}]}
    with pytest.raises(ValueError):
        chunk_document(doc_data, chunk_size_words=50, chunk_overlap_words=50)