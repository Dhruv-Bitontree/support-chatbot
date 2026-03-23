"""Seed database with sample data."""

import json
import logging
import os
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Order
from app.models.faq import FAQEntry
from app.services.faq.base import VectorStore

logger = logging.getLogger(__name__)

SAMPLE_ORDERS = [
    {
        "id": "ORD-1001",
        "customer_email": "alice@example.com",
        "status": "delivered",
        "items": [{"name": "Wireless Headphones", "quantity": 1, "price": 79.99}],
        "total": 79.99,
        "tracking_number": "TRK-998877",
        "estimated_delivery": "2026-03-20",
    },
    {
        "id": "ORD-1002",
        "customer_email": "bob@example.com",
        "status": "in_transit",
        "items": [
            {"name": "Phone Case", "quantity": 2, "price": 14.99},
            {"name": "Screen Protector", "quantity": 1, "price": 9.99},
        ],
        "total": 39.97,
        "tracking_number": "TRK-112233",
        "estimated_delivery": "2026-03-25",
    },
    {
        "id": "ORD-1003",
        "customer_email": "carol@example.com",
        "status": "processing",
        "items": [{"name": "Laptop Stand", "quantity": 1, "price": 49.99}],
        "total": 49.99,
        "tracking_number": None,
        "estimated_delivery": "2026-03-28",
    },
    {
        "id": "ORD-1004",
        "customer_email": "alice@example.com",
        "status": "shipped",
        "items": [
            {"name": "USB-C Hub", "quantity": 1, "price": 34.99},
            {"name": "HDMI Cable", "quantity": 2, "price": 12.99},
        ],
        "total": 60.97,
        "tracking_number": "TRK-445566",
        "estimated_delivery": "2026-03-26",
    },
    {
        "id": "ORD-1005",
        "customer_email": "dave@example.com",
        "status": "cancelled",
        "items": [{"name": "Bluetooth Speaker", "quantity": 1, "price": 59.99}],
        "total": 59.99,
        "tracking_number": None,
        "estimated_delivery": None,
    },
]


async def seed_orders(db: AsyncSession) -> None:
    """Seed sample orders if table is empty."""
    from sqlalchemy import select, func

    result = await db.execute(select(func.count(Order.id)))
    count = result.scalar()
    if count > 0:
        logger.info(f"Orders table has {count} records, skipping seed")
        return

    now = datetime.utcnow()
    for i, order_data in enumerate(SAMPLE_ORDERS):
        order = Order(
            id=order_data["id"],
            customer_email=order_data["customer_email"],
            status=order_data["status"],
            items=order_data["items"],
            total=order_data["total"],
            tracking_number=order_data["tracking_number"],
            estimated_delivery=order_data["estimated_delivery"],
            created_at=now - timedelta(days=10 - i),
            updated_at=now - timedelta(days=5 - i),
        )
        db.add(order)

    await db.flush()
    logger.info(f"Seeded {len(SAMPLE_ORDERS)} sample orders")


async def seed_faqs(vector_store: VectorStore) -> None:
    """Seed FAQ entries from data/faqs.json if store is empty."""
    count = await vector_store.count()
    if count > 0:
        logger.info(f"Vector store has {count} entries, skipping FAQ seed")
        return

    faq_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faqs.json")
    faq_path = os.path.normpath(faq_path)

    if not os.path.exists(faq_path):
        logger.warning(f"FAQ seed file not found: {faq_path}")
        return

    with open(faq_path) as f:
        raw = json.load(f)

    entries = [FAQEntry(**e) for e in raw]
    inserted = await vector_store.upsert(entries)
    logger.info(f"Seeded {inserted} FAQ entries from {faq_path}")
