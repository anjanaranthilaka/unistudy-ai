"""
tests/test_agent.py
"""

from unittest.mock import MagicMock
from agents.study_agent import StudyAgent


def test_empty_query_handling():
    mock_client = MagicMock()
    agent = StudyAgent(client=mock_client)

    result = agent.run("", document_chunks=[])
    assert "Please enter a question" in result["answer"]
    assert result["trace"] == []


def test_missing_documents_refusal():
    mock_client = MagicMock()
    mock_client.generate.return_value = '{"action": "ask_question", "search_query": "machine learning"}'
    agent = StudyAgent(client=mock_client)

    result = agent.run("What is machine learning?", document_chunks=[])
    assert "No documents are currently uploaded" in result["answer"]
    assert len(result["trace"]) > 0
    assert result["trace"][0]["tool"] == "router"


def test_successful_qa_flow_and_trace():
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        '{"action": "ask_question", "search_query": "gradient descent"}',  # Router
        "Gradient descent minimizes loss iteratively."                     # QA Answer
    ]

    agent = StudyAgent(client=mock_client)
    sample_chunks = [
        {
            "chunk_id": 1,
            "filename": "lecture1.pdf",
            "page_num": 3,
            "text": "Gradient descent is an optimization algorithm that minimizes cost."
        }
    ]

    result = agent.run("Explain gradient descent", document_chunks=sample_chunks)

    assert "Gradient descent" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_num"] == 3

    # Check visible trace
    trace = result["trace"]
    assert len(trace) >= 3
    assert trace[0]["tool"] == "router"
    assert trace[1]["tool"] == "search_documents"
    assert trace[1]["status"] == "success"
    assert trace[2]["tool"] == "ollama_generate"


def test_quiz_intent_routing():
    mock_client = MagicMock()
    agent = StudyAgent(client=mock_client)

    route = agent.route_intent("Can you make a quiz from my notes?")
    assert route["action"] == "generate_quiz"