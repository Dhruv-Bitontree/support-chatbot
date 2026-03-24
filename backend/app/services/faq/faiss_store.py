"""FAISS-based vector store for FAQ search."""

import json
import logging
import os
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.exceptions import VectorStoreError
from app.models.faq import FAQEntry, FAQSearchResult
from app.services.faq.base import VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.entries: list[FAQEntry] = []
        self.index_path = settings.faiss_index_path
        self.model_name = settings.embedding_model

    async def initialize(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)

            # Load persisted index if available
            meta_path = f"{self.index_path}_meta.json"
            idx_path = f"{self.index_path}.index"
            if os.path.exists(meta_path) and os.path.exists(idx_path):
                logger.info("Loading persisted FAISS index")
                self.index = faiss.read_index(idx_path)
                with open(meta_path) as f:
                    raw = json.load(f)
                self.entries = [FAQEntry(**e) for e in raw]
                logger.info(f"Loaded {len(self.entries)} FAQ entries")
                
                # STEP 7: Cold-start warning if FAQ store is empty after initialization
                if len(self.entries) == 0:
                    logger.warning(
                        "⚠️  FAQ vector store is EMPTY after initialization! "
                        "All FAQ queries will return no results and the LLM will redirect "
                        "to support contacts. Check that data/faqs.json exists and seeding ran."
                    )
        except Exception as e:
            logger.error(f"FAISS initialization error: {e}")
            raise VectorStoreError(str(e))

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}_meta.json", "w") as f:
            json.dump([e.model_dump() for e in self.entries], f, indent=2)

    async def search(self, query: str, top_k: int = 5) -> list[FAQSearchResult]:
        if not self.index or self.index.ntotal == 0:
            return []
        try:
            query_vec = self._embed([query])
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.entries):
                    continue
                # STEP 7 FIX: IndexFlatIP with normalize_embeddings=True already returns
                # cosine similarity in range [-1, 1]. The (score + 1) / 2 formula was
                # for unnormalized vectors. Now we just clamp to [0, 1] directly.
                normalized_score = float(max(0.0, min(1.0, score)))
                results.append(FAQSearchResult(
                    entry=self.entries[idx],
                    score=normalized_score,
                ))
            return results
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            raise VectorStoreError(str(e))

    async def upsert(self, entries: list[FAQEntry]) -> int:
        try:
            # STEP 7: Deduplication - check existing entry IDs before adding
            existing_ids = {e.id for e in self.entries if e.id}
            new_entries = []
            
            for entry in entries:
                if not entry.id:
                    entry.id = str(uuid.uuid4())
                
                # Skip if this ID already exists (prevents duplicates on restart + reseed)
                if entry.id in existing_ids:
                    logger.debug(f"Skipping duplicate FAQ entry: {entry.id}")
                    continue
                
                new_entries.append(entry)

            if not new_entries:
                logger.info("No new FAQ entries to upsert (all were duplicates)")
                return 0

            # STEP 7 FIX: Embed questions only, not question+answer.
            # User queries are question-like, so embedding the answer text
            # dilutes the semantic match signal.
            texts = [e.question for e in new_entries]
            embeddings = self._embed(texts)
            self.index.add(embeddings)
            self.entries.extend(new_entries)
            self._persist()
            logger.info(f"Upserted {len(new_entries)} FAQ entries")
            return len(new_entries)
        except Exception as e:
            logger.error(f"FAISS upsert error: {e}")
            raise VectorStoreError(str(e))

    async def delete(self, entry_ids: list[str]) -> int:
        # FAISS flat index doesn't support deletion natively,
        # so we rebuild the index without the deleted entries.
        try:
            original_count = len(self.entries)
            remaining = [e for e in self.entries if e.id not in entry_ids]
            deleted = original_count - len(remaining)
            if deleted == 0:
                return 0

            self.entries = remaining
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)
            if remaining:
                # STEP 7: Consistent with upsert - embed questions only
                texts = [e.question for e in remaining]
                embeddings = self._embed(texts)
                self.index.add(embeddings)
            self._persist()
            return deleted
        except Exception as e:
            logger.error(f"FAISS delete error: {e}")
            raise VectorStoreError(str(e))

    async def count(self) -> int:
        return self.index.ntotal if self.index else 0
