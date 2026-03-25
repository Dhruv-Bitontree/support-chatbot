"""Chat-related Pydantic models."""

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Intent(str, Enum):
    FAQ = "faq"
    ORDER_TRACKING = "order_tracking"
    COMPLAINT = "complaint"
    GENERAL = "general"
    GREETING = "greeting"


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for stable client parsing."""
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    channel: str = "web"


class ChatResponse(BaseModel):
    message: str
    session_id: str
    intent: Intent | None = None
    metadata: dict | None = None
    timestamp: datetime = Field(default_factory=_utc_now)


class StreamChunk(BaseModel):
    content: str
    done: bool = False
    session_id: str | None = None
    intent: Intent | None = None
