from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a final answer from a system and user prompt."""

    def to_chat_model(self) -> Any:
        """Return a LangChain BaseChatModel representation of this client."""
        raise NotImplementedError
