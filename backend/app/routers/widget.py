"""Widget configuration endpoint."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/widget", tags=["widget"])


@router.get("/config")
async def get_widget_config(request: Request):
    api_url = f"{request.base_url}api".rstrip("/")
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{request.url.netloc}/api/chat/ws"
    return {
        "api_url": api_url,
        "ws_url": ws_url,
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
