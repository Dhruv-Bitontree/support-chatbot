"""FAQ endpoints."""

from fastapi import APIRouter, Depends

from app.dependencies import get_faq_store
from app.models.faq import FAQCreateRequest, FAQSearchRequest, FAQSearchResult
from app.services.faq.base import VectorStore

router = APIRouter(prefix="/api/faq", tags=["faq"])


@router.post("/search", response_model=list[FAQSearchResult])
async def search_faqs(
    request: FAQSearchRequest,
    store: VectorStore = Depends(get_faq_store),
):
    return await store.search(request.query, top_k=request.top_k)


@router.post("", response_model=dict)
async def create_faqs(
    request: FAQCreateRequest,
    store: VectorStore = Depends(get_faq_store),
):
    count = await store.upsert(request.entries)
    return {"inserted": count}


@router.get("/count")
async def faq_count(store: VectorStore = Depends(get_faq_store)):
    return {"count": await store.count()}
