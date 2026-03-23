"""Order tracking endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.order import OrderLookupRequest, OrderResponse
from app.services.orders.order_service import OrderService

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str, db: AsyncSession = Depends(get_db)):
    service = OrderService(db)
    return await service.get_by_id(order_id)


@router.post("/lookup", response_model=list[OrderResponse])
async def lookup_orders(request: OrderLookupRequest, db: AsyncSession = Depends(get_db)):
    service = OrderService(db)
    if request.order_id:
        order = await service.get_by_id(request.order_id)
        return [order]
    elif request.tracking_number:
        order = await service.get_by_tracking(request.tracking_number)
        return [order]
    elif request.email:
        return await service.get_by_email(request.email)
    return []


@router.get("/{order_id}/status")
async def get_order_status(order_id: str, db: AsyncSession = Depends(get_db)):
    service = OrderService(db)
    summary = await service.get_status_summary(order_id)
    return {"order_id": order_id, "summary": summary}
