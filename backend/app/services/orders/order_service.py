"""Order tracking service."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Order
from app.exceptions import OrderNotFoundError
from app.models.order import OrderItem, OrderResponse, OrderStatus

logger = logging.getLogger(__name__)


class OrderService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, order_id: str) -> OrderResponse:
        result = await self.db.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        if not order:
            raise OrderNotFoundError(order_id)
        return self._to_response(order)

    async def get_by_email(self, email: str) -> list[OrderResponse]:
        result = await self.db.execute(
            select(Order).where(Order.customer_email == email).order_by(Order.created_at.desc())
        )
        orders = result.scalars().all()
        return [self._to_response(o) for o in orders]

    async def get_by_tracking(self, tracking_number: str) -> OrderResponse:
        result = await self.db.execute(
            select(Order).where(Order.tracking_number == tracking_number)
        )
        order = result.scalar_one_or_none()
        if not order:
            raise OrderNotFoundError(tracking_number)
        return self._to_response(order)

    async def get_status_summary(self, order_id: str) -> str:
        """Get a human-readable status summary for the chat bot."""
        order = await self.get_by_id(order_id)
        status_messages = {
            OrderStatus.PENDING: f"Your order {order.id} is pending confirmation.",
            OrderStatus.CONFIRMED: f"Your order {order.id} has been confirmed and is being prepared.",
            OrderStatus.PROCESSING: f"Your order {order.id} is being processed.",
            OrderStatus.SHIPPED: f"Your order {order.id} has been shipped! Tracking: {order.tracking_number or 'N/A'}.",
            OrderStatus.IN_TRANSIT: f"Your order {order.id} is in transit. Tracking: {order.tracking_number or 'N/A'}. Estimated delivery: {order.estimated_delivery or 'TBD'}.",
            OrderStatus.DELIVERED: f"Your order {order.id} has been delivered!",
            OrderStatus.CANCELLED: f"Your order {order.id} has been cancelled.",
            OrderStatus.RETURNED: f"Your order {order.id} has been returned.",
        }
        items_str = ", ".join(f"{i.name} (x{i.quantity})" for i in order.items)
        summary = status_messages.get(order.status, f"Order {order.id} status: {order.status}")
        return f"{summary}\nItems: {items_str}\nTotal: ${order.total:.2f}"

    def _to_response(self, order: Order) -> OrderResponse:
        items = [OrderItem(**i) if isinstance(i, dict) else i for i in (order.items or [])]
        return OrderResponse(
            id=order.id,
            customer_email=order.customer_email,
            status=OrderStatus(order.status),
            items=items,
            total=order.total,
            tracking_number=order.tracking_number,
            estimated_delivery=order.estimated_delivery,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )
