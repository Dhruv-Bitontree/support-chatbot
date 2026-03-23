"""Pinecone-based vector store for FAQ search."""

import logging
import uuid

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.exceptions import VectorStoreError
from app.models.faq import FAQEntry, FAQSearchResult
from app.services.faq.base import VectorStore

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index = None

    async def initialize(self) -> None:
        try:
            from pinecone import Pinecone

            if not settings.pinecone_api_key:
                raise VectorStoreError("PINECONE_API_KEY not set")

            self.model = SentenceTransformer(settings.embedding_model)
            pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index = pc.Index(settings.pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {settings.pinecone_index_name}")
        except ImportError:
            raise VectorStoreError("pinecone-client not installed")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {e}")
            raise VectorStoreError(str(e))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def search(self, query: str, top_k: int = 5) -> list[FAQSearchResult]:
        try:
            query_vec = self._embed([query])[0]
            results = self.index.query(
                vector=query_vec,
                top_k=top_k,
                include_metadata=True,
            )
            faq_results = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                entry = FAQEntry(
                    id=match["id"],
                    question=meta.get("question", ""),
                    answer=meta.get("answer", ""),
                    category=meta.get("category", "general"),
                )
                normalized_score = float(max(0, min(1, (match["score"] + 1) / 2)))
                faq_results.append(FAQSearchResult(entry=entry, score=normalized_score))
            return faq_results
        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            raise VectorStoreError(str(e))

    async def upsert(self, entries: list[FAQEntry]) -> int:
        try:
            vectors = []
            for entry in entries:
                if not entry.id:
                    entry.id = str(uuid.uuid4())
                text = f"{entry.question} {entry.answer}"
                embedding = self._embed([text])[0]
                vectors.append({
                    "id": entry.id,
                    "values": embedding,
                    "metadata": {
                        "question": entry.question,
                        "answer": entry.answer,
                        "category": entry.category,
                    },
                })
            # Pinecone supports batch upsert up to 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                self.index.upsert(vectors=vectors[i:i + batch_size])
            return len(entries)
        except Exception as e:
            logger.error(f"Pinecone upsert error: {e}")
            raise VectorStoreError(str(e))

    async def delete(self, entry_ids: list[str]) -> int:
        try:
            self.index.delete(ids=entry_ids)
            return len(entry_ids)
        except Exception as e:
            logger.error(f"Pinecone delete error: {e}")
            raise VectorStoreError(str(e))

    async def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception as e:
            logger.error(f"Pinecone count error: {e}")
            return 0
