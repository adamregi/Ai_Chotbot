import requests
from requests import RequestException

from config.settings import EMBEDDING_MODEL, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS

OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/embed"


def create_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Text to embed cannot be empty.")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama for embeddings. Start Ollama and ensure "
            f"the '{EMBEDDING_MODEL}' model is installed."
        ) from exc

    embeddings = response.json().get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")
    return embeddings[0]
