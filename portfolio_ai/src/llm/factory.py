from config.settings import LLM_PROVIDER
from src.llm.base_llm import BaseLLM


def get_llm() -> BaseLLM:
    """Return the configured LLM client based on LLM_PROVIDER."""
    if LLM_PROVIDER == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient()
    if LLM_PROVIDER == "nvidia":
        from src.llm.nvidia_client import NvidiaClient
        return NvidiaClient()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
        "Supported: 'nvidia', 'ollama'."
    )

