"""
llm/ollama_client.py

Clean REST API client for interacting with a local Ollama daemon.
Uses standard Python 'requests' to avoid heavy dependencies.
"""

import json
from typing import Dict, Any, Optional
import requests


class OllamaConnectionError(Exception):
    """Raised when Ollama server is unreachable or offline."""
    pass


class OllamaModelNotFoundError(Exception):
    """Raised when the specified model is not downloaded in Ollama."""
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Verifies that Ollama is running and checks whether the model is pulled.

        Returns:
            dict: {"is_running": bool, "model_available": bool, "error": str or None}
        """
        status = {"is_running": False, "model_available": False, "error": None}
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                status["is_running"] = True
                models_data = resp.json().get("models", [])
                available_names = [m.get("name", "") for m in models_data]

                # Check direct match or tag prefix (e.g., 'llama3.2:3b' vs 'llama3.2:3b-latest')
                matches = any(self.model in name for name in available_names)
                status["model_available"] = matches
                if not matches:
                    status["error"] = f"Model '{self.model}' not found in Ollama. Run 'ollama pull {self.model}'."
            else:
                status["error"] = f"Ollama returned HTTP status {resp.status_code}."
        except requests.exceptions.ConnectionError:
            status["error"] = "Cannot connect to Ollama. Ensure Ollama is running on your computer."
        except Exception as e:
            status["error"] = f"Health check failed: {str(e)}"

        return status

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.2,
            format_json: bool = False
    ) -> str:
        """
        Sends a generation request to /api/generate.

        Args:
            prompt: User/task prompt text.
            system: Optional system instruction prompt.
            temperature: Sampling temperature (0.0 for deterministic, 0.7 for creative).
            format_json: If True, forces Ollama's constrained JSON mode.

        Returns:
            str: Generated text response from the model.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system:
            payload["system"] = system

        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError("Ollama server is not running at " + self.base_url)
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(f"Ollama request timed out after {self.timeout} seconds.")

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(f"Model '{self.model}' was not found by Ollama.")
        elif resp.status_code != 200:
            raise RuntimeError(f"Ollama error ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data.get("response", "").strip()