"""Complaint and ticket Pydantic models."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SentimentResult(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    label: str  # positive, neutral, negative
    confidence: float = Field(..., ge=0.0, le=1.0)


class ComplaintRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str | None = None
    customer_email: str | None = None
    order_id: str | None = None


class TicketResponse(BaseModel):
    id: str
    session_id: str | None = None
    category: str
    description: str
    sentiment: SentimentResult
    priority: TicketPriority
    status: TicketStatus
    assigned_to: str | None = None
    created_at: datetime
    updated_at: datetime
