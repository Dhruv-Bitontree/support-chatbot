"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import engine
from app.db.models import Base
from app.db.seed import seed_faqs, seed_orders
from app.exceptions import AppError
from app.routers import chat, complaints, faq, health, orders, widget

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Support Chatbot API")
    logger.info(f"LLM Provider: {settings.llm_provider.value}")
    logger.info(f"Vector Store: {settings.vector_store_provider.value}")

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # Seed data
    from app.db.database import async_session
    from app.services.faq.factory import get_vector_store

    try:
        vector_store = await get_vector_store()
        await seed_faqs(vector_store)
    except Exception as e:
        logger.warning(f"FAQ seeding skipped: {e}")

    try:
        async with async_session() as db:
            await seed_orders(db)
            await db.commit()
    except Exception as e:
        logger.warning(f"Order seeding skipped: {e}")

    logger.info("Startup complete")
    yield

    # Shutdown
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Customer Support Chatbot API",
        description="Production-grade customer support chatbot with FAQ, order tracking, and complaint management.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message},
        )

    # Routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(faq.router)
    app.include_router(orders.router)
    app.include_router(complaints.router)
    app.include_router(widget.router)

    return app


app = create_app()
