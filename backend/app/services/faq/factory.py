"""Vector store factory."""

import logging

from app.config import VectorStoreProvider, settings
from app.services.faq.base import VectorStore

logger = logging.getLogger(__name__)

_store_instance: VectorStore | None = None


async def get_vector_store() -> VectorStore:
    """Get or create the configured vector store singleton."""
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    provider = settings.vector_store_provider
    logger.info(f"Initializing vector store: {provider.value}")

    if provider == VectorStoreProvider.FAISS:
        from app.services.faq.faiss_store import FAISSStore
        _store_instance = FAISSStore()
    elif provider == VectorStoreProvider.PINECONE:
        from app.services.faq.pinecone_store import PineconeStore
        _store_instance = PineconeStore()
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

    await _store_instance.initialize()
    return _store_instance


def reset_store():
    """Reset singleton (for testing)."""
    global _store_instance
    _store_instance = None
