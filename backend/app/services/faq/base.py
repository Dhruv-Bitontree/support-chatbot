"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod

from app.models.faq import FAQEntry, FAQSearchResult


class VectorStore(ABC):
    """Pluggable vector store interface."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (load index, connect, etc.)."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[FAQSearchResult]:
        """Search for similar FAQ entries."""
        ...

    @abstractmethod
    async def upsert(self, entries: list[FAQEntry]) -> int:
        """Insert or update FAQ entries. Returns count of entries upserted."""
        ...

    @abstractmethod
    async def delete(self, entry_ids: list[str]) -> int:
        """Delete FAQ entries by ID. Returns count deleted."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return total number of entries in the store."""
        ...
