"""
tests/test_document_reader.py

Unit tests for tools/document_reader.py.
Tests plain text, markdown, empty file handling, and invalid extensions.
"""

import io
import pytest
from tools.document_reader import read_document


def test_read_plain_text():
    content = "Introduction to Machine Learning\nLecture 1 Notes."
    file_stream = io.BytesIO(content.encode("utf-8"))

    result = read_document(file_stream, "notes.txt")

    assert result["error"] is None
    assert result["file_type"] == ".txt"
    assert result["total_pages"] == 1
    assert "Machine Learning" in result["full_text"]
    assert len(result["pages"]) == 1


def test_read_markdown():
    content = "# Deep Learning\n- Perceptrons\n- Backpropagation"
    file_stream = io.BytesIO(content.encode("utf-8"))

    result = read_document(file_stream, "deep_learning.md")

    assert result["error"] is None
    assert result["file_type"] == ".md"
    assert "Backpropagation" in result["full_text"]


def test_unsupported_file_extension():
    content = "Some data"
    file_stream = io.BytesIO(content.encode("utf-8"))

    result = read_document(file_stream, "presentation.pptx")

    assert result["error"] is not None
    assert "Unsupported file extension" in result["error"]


def test_empty_file_handling():
    content = "   \n\t  "
    file_stream = io.BytesIO(content.encode("utf-8"))

    result = read_document(file_stream, "empty.txt")

    assert result["error"] is not None
    assert "empty" in result["error"].lower()