"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models.chat import Message


class LLMProvider(ABC):
    """Plug-and-play LLM provider interface."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

    @abstractmethod
    async def classify(self, text: str, categories: list[str]) -> str:
        """Classify text into one of the given categories."""
        ...
