"""
tools/quiz_generator.py

Generates multiple-choice questions from document context using structured JSON mode.
Includes validation to handle malformed LLM outputs safely.
"""

import json
from typing import Dict, Any, List
from llm.ollama_client import OllamaClient
from utils.prompts import QUIZ_SYSTEM_PROMPT


def generate_quiz(
    chunks: List[Dict[str, Any]],
    client: OllamaClient,
    num_questions: int = 3
) -> Dict[str, Any]:
    """
    Generates multiple-choice quiz questions based strictly on the provided chunks.

    Args:
        chunks: List of retrieved chunk dicts containing context.
        client: Active OllamaClient instance.
        num_questions: Target number of questions to generate.

    Returns:
        dict: {
            "questions": List[Dict[str, Any]],
            "raw_output": str,
            "error": str or None
        }
    """
    result: Dict[str, Any] = {"questions": [], "raw_output": "", "error": None}

    if not chunks:
        result["error"] = "No document content was provided to generate quiz questions."
        return result

    # Assemble reference text with page tags
    context_parts = [f"[{c.get('filename', 'Doc')} - Page {c.get('page_num', '?')}]: {c.get('text', '')}" for c in chunks]
    context_text = "\n\n".join(context_parts)

    prompt = (
        f"Generate {num_questions} multiple-choice study questions from the following lecture notes:\n\n"
        f"<notes>\n{context_text}\n</notes>\n\n"
        "Remember: output strictly valid JSON matching the schema."
    )

    try:
        raw_json_str = client.generate(
            prompt=prompt,
            system=QUIZ_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for strict schema adherence
            format_json=True
        )
        result["raw_output"] = raw_json_str

        parsed = json.loads(raw_json_str)

        # Handle models returning either {"questions": [...]} or directly a list [...]
        questions_list = parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed

        validated_questions = []
        if isinstance(questions_list, list):
            for idx, q in enumerate(questions_list):
                if isinstance(q, dict) and "question" in q and "options" in q:
                    validated_questions.append({
                        "question_id": q.get("question_id", idx + 1),
                        "question": str(q.get("question", "")),
                        "options": [str(opt) for opt in q.get("options", [])],
                        "correct_answer": str(q.get("correct_answer", "")),
                        "explanation": str(q.get("explanation", "Based on provided notes."))
                    })

        if not validated_questions:
            result["error"] = "Model output did not contain valid multiple-choice questions."
        else:
            result["questions"] = validated_questions

    except json.JSONDecodeError:
        result["error"] = "Model failed to generate valid JSON format for the quiz."
    except Exception as e:
        result["error"] = f"Quiz generation failed: {str(e)}"

    return result