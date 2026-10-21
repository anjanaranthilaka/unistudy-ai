"""
tools/document_reader.py

Handles reading and text extraction from PDF, TXT, and Markdown files.
Preserves page metadata for PDF files to enable accurate citations.
"""

import os
from typing import Dict, Any, List
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# 25 MB file size limit to prevent out-of-memory errors on local hardware
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


class DocumentReaderError(Exception):
    """Custom exception raised when document reading or extraction fails."""
    pass


def read_document(file_source: Any, filename: str) -> Dict[str, Any]:
    """
    Reads a document (PDF, TXT, or MD) and returns extracted text and metadata.

    Args:
        file_source: A file path (str) or a file-like byte stream (such as Streamlit's UploadedFile).
        filename: The original name of the file including extension (e.g. 'lecture1.pdf').

    Returns:
        dict: {
            "filename": str,
            "file_type": str,
            "total_pages": int,
            "pages": List[Dict[str, Any]], # [{"page_num": 1, "text": "..."}]
            "full_text": str,
            "is_scanned_warning": bool,
            "error": str or None
        }
    """
    result = {
        "filename": filename,
        "file_type": "",
        "total_pages": 1,
        "pages": [],
        "full_text": "",
        "is_scanned_warning": False,
        "error": None
    }

    ext = os.path.splitext(filename)[1].lower()
    result["file_type"] = ext

    # Validate file extension
    if ext not in [".pdf", ".txt", ".md"]:
        result["error"] = f"Unsupported file extension '{ext}'. Only .pdf, .txt, and .md are supported."
        return result

    try:
        if ext == ".pdf":
            _extract_pdf(file_source, result)
        else:
            _extract_plain_text(file_source, result)

    except PdfReadError as e:
        result["error"] = f"Corrupted or invalid PDF file: {str(e)}"
    except UnicodeDecodeError:
        result["error"] = "Encoding error: File is not valid UTF-8 text."
    except Exception as e:
        result["error"] = f"Unexpected error reading file: {str(e)}"

    return result


def _extract_plain_text(file_source: Any, result: Dict[str, Any]) -> None:
    """Helper to extract text from .txt and .md files."""
    if hasattr(file_source, "read"):
        content_bytes = file_source.read()
        # Reset pointer in case file object is reused
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        text = content_bytes.decode("utf-8", errors="replace")
    elif isinstance(file_source, str) and os.path.exists(file_source):
        with open(file_source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        result["error"] = "Invalid file source provided."
        return

    text = text.strip()
    if not text:
        result["error"] = "The uploaded file is empty."
        return

    result["total_pages"] = 1
    result["pages"] = [{"page_num": 1, "text": text}]
    result["full_text"] = text


def _extract_pdf(file_source: Any, result: Dict[str, Any]) -> None:
    """Helper to extract text page-by-page using pypdf."""
    reader = PdfReader(file_source)
    total_pages = len(reader.pages)

    if total_pages == 0:
        result["error"] = "The PDF contains no pages."
        return

    result["total_pages"] = total_pages
    extracted_pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ""
        clean_page_text = page_text.strip()

        if clean_page_text:
            extracted_pages.append({
                "page_num": page_num,
                "text": clean_page_text
            })
            full_text_parts.append(f"[Page {page_num}]\n{clean_page_text}")

    result["pages"] = extracted_pages
    result["full_text"] = "\n\n".join(full_text_parts)

    # Scanned PDF detection:
    # If the PDF has pages, but total extracted characters across all pages is under 50,
    # it is almost certainly scanned images of slides/pages without an embedded text layer.
    total_chars = sum(len(p["text"]) for p in extracted_pages)
    if total_pages > 0 and total_chars < 50:
        result["is_scanned_warning"] = True
        result["error"] = (
            "No readable text could be extracted. This document appears to be a scanned image "
            "or contains only graphics. An OCR (Optical Character Recognition) tool is required "
            "to read text from scanned PDFs."
        )