import requests
from requests import RequestException

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, REQUEST_TIMEOUT_SECONDS
from src.llm.base_llm import BaseLLM


class OllamaClient(BaseLLM):
    def __init__(self, model: str = OLLAMA_CHAT_MODEL):
        self.model = model
        self.url = f"{OLLAMA_BASE_URL}/api/generate"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start Ollama and ensure "
                f"the '{self.model}' model is installed."
            ) from exc

        answer = response.json().get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer
