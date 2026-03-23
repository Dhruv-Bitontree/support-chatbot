"""Ticket creation and escalation service."""

import logging
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Ticket
from app.exceptions import TicketNotFoundError
from app.models.complaint import (
    ComplaintRequest,
    SentimentResult,
    TicketPriority,
    TicketResponse,
    TicketStatus,
)
from app.services.complaints.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)


class TicketService:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _determine_priority(self, sentiment: SentimentResult) -> TicketPriority:
        if sentiment.score <= settings.sentiment_escalation_threshold:
            return TicketPriority.URGENT
        elif sentiment.score <= settings.sentiment_negative_threshold:
            return TicketPriority.HIGH
        elif sentiment.score <= 0.05:
            return TicketPriority.MEDIUM
        return TicketPriority.LOW

    def _determine_status(self, priority: TicketPriority) -> TicketStatus:
        if priority == TicketPriority.URGENT:
            return TicketStatus.ESCALATED
        return TicketStatus.OPEN

    async def create_ticket(
        self,
        request: ComplaintRequest,
        category: str = "general",
    ) -> TicketResponse:
        sentiment = analyze_sentiment(request.message)
        priority = self._determine_priority(sentiment)
        status = self._determine_status(priority)
        ticket_id = str(uuid.uuid4())

        ticket = Ticket(
            id=ticket_id,
            session_id=request.session_id,
            category=category,
            description=request.message,
            sentiment_score=sentiment.score,
            sentiment_label=sentiment.label,
            priority=priority.value,
            status=status.value,
            customer_email=request.customer_email,
            order_id=request.order_id,
        )
        self.db.add(ticket)
        await self.db.flush()

        logger.info(
            f"Created ticket {ticket_id} | priority={priority.value} "
            f"| sentiment={sentiment.score:.2f} | status={status.value}"
        )

        return self._to_response(ticket, sentiment)

    async def get_ticket(self, ticket_id: str) -> TicketResponse:
        result = await self.db.execute(select(Ticket).where(Ticket.id == ticket_id))
        ticket = result.scalar_one_or_none()
        if not ticket:
            raise TicketNotFoundError(ticket_id)
        sentiment = SentimentResult(
            score=ticket.sentiment_score or 0,
            label=ticket.sentiment_label or "neutral",
            confidence=0.8,
        )
        return self._to_response(ticket, sentiment)

    async def list_tickets(
        self,
        status: TicketStatus | None = None,
        limit: int = 50,
    ) -> list[TicketResponse]:
        query = select(Ticket).order_by(Ticket.created_at.desc()).limit(limit)
        if status:
            query = query.where(Ticket.status == status.value)
        result = await self.db.execute(query)
        tickets = result.scalars().all()
        return [
            self._to_response(
                t,
                SentimentResult(
                    score=t.sentiment_score or 0,
                    label=t.sentiment_label or "neutral",
                    confidence=0.8,
                ),
            )
            for t in tickets
        ]

    def _to_response(self, ticket: Ticket, sentiment: SentimentResult) -> TicketResponse:
        return TicketResponse(
            id=ticket.id,
            session_id=ticket.session_id,
            category=ticket.category,
            description=ticket.description,
            sentiment=sentiment,
            priority=TicketPriority(ticket.priority),
            status=TicketStatus(ticket.status),
            assigned_to=ticket.assigned_to,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
        )
