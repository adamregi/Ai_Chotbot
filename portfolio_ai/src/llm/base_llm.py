from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a final answer from a system and user prompt."""
