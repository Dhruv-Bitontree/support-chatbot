"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "support-chatbot-api",
        "version": "1.0.0",
    }
