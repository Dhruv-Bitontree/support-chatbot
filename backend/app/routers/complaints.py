"""Complaint and ticket endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.complaint import ComplaintRequest, TicketResponse, TicketStatus
from app.services.complaints.ticket_service import TicketService

router = APIRouter(prefix="/api/complaints", tags=["complaints"])


@router.post("", response_model=TicketResponse)
async def create_complaint(
    request: ComplaintRequest,
    db: AsyncSession = Depends(get_db),
):
    service = TicketService(db)
    return await service.create_ticket(request, category="complaint")


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)):
    service = TicketService(db)
    return await service.get_ticket(ticket_id)


@router.get("", response_model=list[TicketResponse])
async def list_tickets(
    status: TicketStatus | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    service = TicketService(db)
    return await service.list_tickets(status=status, limit=limit)
