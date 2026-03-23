"""Widget configuration endpoint."""

from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api/widget", tags=["widget"])


@router.get("/config")
async def get_widget_config():
    return {
        "api_url": f"http://localhost:{settings.backend_port}/api",
        "ws_url": f"ws://localhost:{settings.backend_port}/api/chat/ws",
        "theme": {
            "primary_color": "#2563eb",
            "font_family": "Inter, system-ui, sans-serif",
            "border_radius": "12px",
        },
        "features": {
            "faq_search": True,
            "order_tracking": True,
            "complaints": True,
        },
    }
