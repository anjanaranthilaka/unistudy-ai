"""
tools/summarizer.py

Summarization tool for UniStudy AI.
Accepts document text or retrieved passages and requests a structured summary from Ollama.
"""

from typing import Dict, Any, List
from llm.ollama_client import OllamaClient
from utils.prompts import SUMMARIZATION_SYSTEM_PROMPT, format_summarize_prompt


def summarize_document(
    text_or_chunks: Any,
    client: OllamaClient,
    max_words: int = 1500
) -> Dict[str, Any]:
    """
    Summarizes document content into structured study points.

    Args:
        text_or_chunks: Plain string of text OR a list of chunk dicts from chunk_document().
        client: Active OllamaClient instance.
        max_words: Truncation threshold to avoid overflowing small LLM context windows.

    Returns:
        dict: {"summary": str, "source_chunks_used": int, "error": str or None}
    """
    result = {"summary": "", "source_chunks_used": 0, "error": None}

    # Extract clean text from either a string or chunk list
    if isinstance(text_or_chunks, list):
        result["source_chunks_used"] = len(text_or_chunks)
        combined_text = "\n\n".join([c.get("text", "") for c in text_or_chunks])
    elif isinstance(text_or_chunks, str):
        combined_text = text_or_chunks.strip()
        result["source_chunks_used"] = 1 if combined_text else 0
    else:
        result["error"] = "Invalid input: expected text string or list of chunks."
        return result

    if not combined_text:
        result["error"] = "No document text was provided to summarize."
        return result

    # Prevent context overflow by capping word count
    words = combined_text.split()
    if len(words) > max_words:
        combined_text = " ".join(words[:max_words]) + "\n...[truncated for summary]"

    prompt = format_summarize_prompt(combined_text)

    try:
        summary_response = client.generate(
            prompt=prompt,
            system=SUMMARIZATION_SYSTEM_PROMPT,
            temperature=0.2
        )
        result["summary"] = summary_response
    except Exception as e:
        result["error"] = f"Summarization failed: {str(e)}"

    return result