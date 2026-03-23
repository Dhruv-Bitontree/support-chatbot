"""Tests for order service."""

import pytest
import pytest_asyncio
from datetime import datetime

from app.db.models import Order
from app.exceptions import OrderNotFoundError
from app.services.orders.order_service import OrderService


@pytest.mark.asyncio
class TestOrderService:
    async def test_get_by_id(self, db_session):
        order = Order(
            id="ORD-TEST-001",
            customer_email="test@example.com",
            status="shipped",
            items=[{"name": "Widget", "quantity": 1, "price": 29.99}],
            total=29.99,
            tracking_number="TRK-123",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db_session.add(order)
        await db_session.flush()

        service = OrderService(db_session)
        result = await service.get_by_id("ORD-TEST-001")

        assert result.id == "ORD-TEST-001"
        assert result.status.value == "shipped"
        assert result.total == 29.99

    async def test_get_by_id_not_found(self, db_session):
        service = OrderService(db_session)
        with pytest.raises(OrderNotFoundError):
            await service.get_by_id("ORD-NONEXISTENT")

    async def test_get_by_email(self, db_session):
        for i in range(3):
            order = Order(
                id=f"ORD-EMAIL-{i}",
                customer_email="multi@example.com",
                status="pending",
                items=[],
                total=10.0 * i,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db_session.add(order)
        await db_session.flush()

        service = OrderService(db_session)
        results = await service.get_by_email("multi@example.com")
        assert len(results) == 3

    async def test_get_status_summary(self, db_session):
        order = Order(
            id="ORD-SUM-001",
            customer_email="sum@example.com",
            status="in_transit",
            items=[{"name": "Gadget", "quantity": 2, "price": 15.00}],
            total=30.00,
            tracking_number="TRK-SUM",
            estimated_delivery="2026-03-25",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db_session.add(order)
        await db_session.flush()

        service = OrderService(db_session)
        summary = await service.get_status_summary("ORD-SUM-001")

        assert "in transit" in summary.lower()
        assert "TRK-SUM" in summary
        assert "$30.00" in summary
