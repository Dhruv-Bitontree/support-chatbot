"""FastAPI dependency injection."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.chat.orchestrator import ChatOrchestrator
from app.services.faq.base import VectorStore
from app.services.faq.factory import get_vector_store
from app.services.llm.base import LLMProvider
from app.services.llm.factory import get_llm_provider


async def get_llm() -> LLMProvider:
    return get_llm_provider()


async def get_faq_store() -> VectorStore:
    return await get_vector_store()


async def get_orchestrator(
    llm: LLMProvider = None,
    vector_store: VectorStore = None,
    db: AsyncSession = None,
) -> ChatOrchestrator:
    """This is used manually in routers, not as a direct Depends()."""
    return ChatOrchestrator(llm=llm, vector_store=vector_store, db=db)
