"""Tests for FAQ/vector store (using mock store)."""

import pytest

from app.models.faq import FAQEntry, FAQSearchResult


@pytest.mark.asyncio
class TestFAQStore:
    async def test_upsert_and_count(self, mock_vector_store):
        entries = [
            FAQEntry(question="How to return?", answer="Use our return portal.", category="returns"),
            FAQEntry(question="Shipping time?", answer="5-7 business days.", category="shipping"),
        ]
        count = await mock_vector_store.upsert(entries)
        assert count == 2
        assert await mock_vector_store.count() == 2

    async def test_search_returns_results(self, mock_vector_store):
        entries = [
            FAQEntry(question="Return policy?", answer="30 days.", category="returns"),
        ]
        await mock_vector_store.upsert(entries)
        results = await mock_vector_store.search("return")
        assert len(results) == 1
        assert results[0].score > 0

    async def test_search_empty_store(self, mock_vector_store):
        results = await mock_vector_store.search("anything")
        assert results == []
