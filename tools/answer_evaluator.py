"""
tools/answer_evaluator.py

Evaluates student answers against reference course material.
Provides constructive feedback, approximate scores, and suggested model answers.
"""

import json
from typing import Dict, Any, List
from llm.ollama_client import OllamaClient
from utils.prompts import EVALUATION_SYSTEM_PROMPT


def evaluate_answer(
    question: str,
    student_answer: str,
    reference_chunks: List[Dict[str, Any]],
    client: OllamaClient
) -> Dict[str, Any]:
    """
    Compares a student's answer to the lecture notes.

    Args:
        question: The study question being answered.
        student_answer: The student's response.
        reference_chunks: Relevant document chunks providing the source truth.
        client: Active OllamaClient instance.

    Returns:
        dict: {
            "score": str,
            "score_note": str,
            "strengths": str,
            "gaps": str,
            "model_answer": str,
            "error": str or None
        }
    """
    result = {
        "score": "N/A",
        "score_note": "Score is approximate and for revision guidance only.",
        "strengths": "",
        "gaps": "",
        "model_answer": "",
        "error": None
    }

    if not question.strip():
        result["error"] = "No question was provided to evaluate against."
        return result

    if not student_answer.strip():
        result["error"] = "No student answer was submitted for evaluation."
        return result

    ref_text = "\n\n".join([f"[{c.get('filename', 'Doc')}]: {c.get('text', '')}" for c in reference_chunks])
    if not ref_text.strip():
        ref_text = "No specific reference notes provided. Grade based on standard academic accuracy."

    prompt = (
        f"Study Question: {question}\n\n"
        f"Student Answer: {student_answer}\n\n"
        f"Reference Lecture Material:\n{ref_text}\n\n"
        "Evaluate the student's submission. Output valid JSON."
    )

    try:
        raw_json = client.generate(
            prompt=prompt,
            system=EVALUATION_SYSTEM_PROMPT,
            temperature=0.1,
            format_json=True
        )

        data = json.loads(raw_json)
        result["score"] = data.get("score", "Ungraded")
        result["score_note"] = data.get("score_note", result["score_note"])
        result["strengths"] = data.get("strengths", "No specific strengths identified.")
        result["gaps"] = data.get("gaps", "None noted.")
        result["model_answer"] = data.get("model_answer", "")

    except json.JSONDecodeError:
        result["error"] = "Evaluation output could not be parsed as structured JSON."
    except Exception as e:
        result["error"] = f"Evaluation failed: {str(e)}"

    return result