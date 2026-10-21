"""
tests/test_document_search.py
"""

from tools.document_search import tokenize, search_documents


def test_tokenize():
    tokens = tokenize("What is Gradient Descent in Machine Learning?")
    assert "gradient" in tokens
    assert "descent" in tokens
    assert "machine" in tokens
    # Stopword 'is' and 'in' should be filtered
    assert "is" not in tokens
    assert "in" not in tokens


def test_search_documents():
    chunks = [
        {
            "chunk_id": 1,
            "filename": "ml_lecture.pdf",
            "page_num": 1,
            "text": "Supervised learning algorithms predict outcomes using labeled datasets."
        },
        {
            "chunk_id": 2,
            "filename": "ml_lecture.pdf",
            "page_num": 2,
            "text": "Gradient descent is an optimization algorithm used to minimize the loss function."
        },
        {
            "chunk_id": 3,
            "filename": "history.txt",
            "page_num": 1,
            "text": "The industrial revolution began in Great Britain during the late 18th century."
        }
    ]

    results = search_documents("How does gradient descent minimize loss?", chunks, top_k=2)

    assert len(results) > 0
    # Top result must be chunk 2
    assert results[0]["chunk_id"] == 2
    assert results[0]["page_num"] == 2
    assert results[0]["score"] > 0


def test_search_empty_queries_or_chunks():
    assert search_documents("", []) == []
    assert search_documents("gradient descent", []) == []