"""Order-related Pydantic models."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class OrderItem(BaseModel):
    name: str
    quantity: int
    price: float


class OrderResponse(BaseModel):
    id: str
    customer_email: str
    status: OrderStatus
    items: list[OrderItem]
    total: float
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    created_at: datetime
    updated_at: datetime


class OrderLookupRequest(BaseModel):
    order_id: str | None = None
    email: str | None = None
    tracking_number: str | None = None
