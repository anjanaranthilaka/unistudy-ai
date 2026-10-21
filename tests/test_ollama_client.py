"""
tests/test_ollama_client.py
"""

from unittest.mock import patch, MagicMock
import pytest
from llm.ollama_client import OllamaClient, OllamaConnectionError


@patch("requests.get")
def test_check_health_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:3b"}, {"name": "mistral:latest"}]
    }
    mock_get.return_value = mock_resp

    client = OllamaClient(model="llama3.2:3b")
    status = client.check_health()

    assert status["is_running"] is True
    assert status["model_available"] is True
    assert status["error"] is None


@patch("requests.get")
def test_check_health_missing_model(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "gemma:2b"}]
    }
    mock_get.return_value = mock_resp

    client = OllamaClient(model="llama3.2:3b")
    status = client.check_health()

    assert status["is_running"] is True
    assert status["model_available"] is False
    assert "not found" in status["error"].lower()


@patch("requests.post")
def test_generate_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "Supervised learning uses labeled data."}
    mock_post.return_value = mock_resp

    client = OllamaClient()
    output = client.generate("What is supervised learning?")

    assert "Supervised learning" in output
    assert mock_post.called


@patch("requests.post")
def test_generate_connection_failure(mock_post):
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = OllamaClient()
    with pytest.raises(OllamaConnectionError):
        client.generate("Hello?")