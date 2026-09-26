from config.settings import LLM_PROVIDER
from langchain_core.language_models.chat_models import BaseChatModel
from src.llm.base_llm import BaseLLM


def get_chat_model() -> BaseChatModel:
    """Return the configured LangChain chat model based on LLM_PROVIDER."""
    if LLM_PROVIDER == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient().to_chat_model()
    if LLM_PROVIDER == "nvidia":
        from src.llm.nvidia_client import NvidiaClient
        return NvidiaClient().to_chat_model()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
        "Supported: 'nvidia', 'ollama'."
    )


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
