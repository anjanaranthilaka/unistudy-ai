"""
tests/test_tools.py
"""

from unittest.mock import MagicMock
from tools.summarizer import summarize_document
from tools.quiz_generator import generate_quiz
from tools.answer_evaluator import evaluate_answer


def test_summarize_document():
    mock_client = MagicMock()
    mock_client.generate.return_value = "This lecture discusses supervised learning and regression."

    chunks = [{"text": "Supervised learning uses labeled training datasets."}]
    res = summarize_document(chunks, mock_client)

    assert res["error"] is None
    assert "supervised learning" in res["summary"].lower()
    assert res["source_chunks_used"] == 1


def test_generate_quiz_valid():
    mock_client = MagicMock()
    mock_client.generate.return_value = """{
        "questions": [
            {
                "question_id": 1,
                "question": "What is overfitting?",
                "options": ["A) High bias", "B) Memorizing noise", "C) Fast training", "D) Linear fit"],
                "correct_answer": "B",
                "explanation": "Overfitting occurs when models learn noise."
            }
        ]
    }"""

    chunks = [{"text": "Overfitting happens when a model learns noise."}]
    res = generate_quiz(chunks, mock_client, num_questions=1)

    assert res["error"] is None
    assert len(res["questions"]) == 1
    assert res["questions"][0]["correct_answer"] == "B"


def test_evaluate_answer_valid():
    mock_client = MagicMock()
    mock_client.generate.return_value = """{
        "score": "8/10",
        "score_note": "Score is approximate.",
        "strengths": "Accurate definition of gradient descent.",
        "gaps": "Did not specify learning rate.",
        "model_answer": "Gradient descent minimizes the loss function iteratively."
    }"""

    chunks = [{"text": "Gradient descent is an optimization algorithm."}]
    res = evaluate_answer(
        question="What is gradient descent?",
        student_answer="It is an algorithm that minimizes loss.",
        reference_chunks=chunks,
        client=mock_client
    )

    assert res["error"] is None
    assert res["score"] == "8/10"
    assert "gradient descent" in res["strengths"].lower()