"""FAQ-related Pydantic models."""

from pydantic import BaseModel, Field


class FAQEntry(BaseModel):
    id: str | None = None
    question: str
    answer: str
    category: str = "general"
    metadata: dict | None = None


class FAQSearchResult(BaseModel):
    entry: FAQEntry
    score: float = Field(..., ge=0.0, le=1.0)


class FAQSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class FAQCreateRequest(BaseModel):
    entries: list[FAQEntry]
