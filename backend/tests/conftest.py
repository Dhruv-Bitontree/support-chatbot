"""Test fixtures."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.models.chat import Message, MessageRole
from app.services.faq.base import VectorStore
from app.services.llm.base import LLMProvider


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "This is a test response."):
        self.response = response
        self.calls: list[dict] = []

    async def generate(self, messages, system_prompt="", **kwargs) -> str:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        return self.response

    async def stream(self, messages, system_prompt="", **kwargs) -> AsyncIterator[str]:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        for word in self.response.split():
            yield word + " "

    async def classify(self, text: str, categories: list[str]) -> str:
        return categories[0]


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""

    def __init__(self):
        self.entries = []

    async def initialize(self):
        pass

    async def search(self, query, top_k=5):
        from app.models.faq import FAQEntry, FAQSearchResult
        if self.entries:
            return [
                FAQSearchResult(entry=self.entries[0], score=0.9)
            ]
        return []

    async def upsert(self, entries):
        self.entries.extend(entries)
        return len(entries)

    async def delete(self, entry_ids):
        return 0

    async def count(self):
        return len(self.entries)


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def mock_vector_store():
    return MockVectorStore()
