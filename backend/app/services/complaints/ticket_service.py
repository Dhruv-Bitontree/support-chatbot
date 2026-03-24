"""Ticket creation and escalation service."""

import logging
import uuid

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

# FIX 3: sentiment_negative_threshold was referenced from settings but never
# defined in config, causing an AttributeError crash on every complaint.
# Defined here as a module constant so it works without a config change.
# Value of -0.25 means: scores between -0.25 and -0.5 → HIGH priority.
# Scores below -0.5 (settings.sentiment_escalation_threshold) → URGENT.
_NEGATIVE_THRESHOLD = getattr(settings, "sentiment_negative_threshold", -0.25)


class TicketService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_existing_ticket(self, session_id: str) -> Ticket | None:
        """Check if an open or escalated ticket already exists for this session.
        
        STEP 5: Prevents duplicate ticket creation when customer sends multiple
        angry messages in a row.
        """
        result = await self.db.execute(
            select(Ticket)
            .where(Ticket.session_id == session_id)
            .where(Ticket.status.in_(["open", "escalated"]))
            .order_by(Ticket.created_at.desc())
        )
        return result.scalar_one_or_none()

    async def notify_escalation(self, ticket: Ticket) -> None:
        """Notify support team of escalated ticket.
        
        STEP 5: Currently logs at CRITICAL level. In production, this would
        send email/Slack/webhook notification to on-call support team.
        """
        logger.critical(
            f"🚨 ESCALATED TICKET REQUIRES IMMEDIATE ATTENTION 🚨\n"
            f"Ticket ID: {ticket.id}\n"
            f"Session ID: {ticket.session_id}\n"
            f"Customer Email: {ticket.customer_email or 'NOT PROVIDED'}\n"
            f"Order ID: {ticket.order_id or 'N/A'}\n"
            f"Sentiment Score: {ticket.sentiment_score:.2f}\n"
            f"Priority: {ticket.priority}\n"
            f"Description: {ticket.description[:200]}...\n"
            f"Created: {ticket.created_at}\n"
        )

    def _determine_priority(self, sentiment: SentimentResult) -> TicketPriority:
        """Map sentiment score to ticket priority.

        Thresholds (lower score = more negative = higher priority):
          score <= -0.5  → URGENT   (auto-escalated, human-in-the-loop)
          score <= -0.25 → HIGH
          score <=  0.05 → MEDIUM   (neutral or mildly negative)
          score  >  0.05 → LOW      (positive sentiment, likely a polite inquiry)
        """
        if sentiment.score <= settings.sentiment_escalation_threshold:
            return TicketPriority.URGENT
        elif sentiment.score <= _NEGATIVE_THRESHOLD:
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

        # STEP 5: Notify support team if ticket is escalated
        if status == TicketStatus.ESCALATED:
            await self.notify_escalation(ticket)

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
            score=ticket.sentiment_score or 0.0,
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
                    score=t.sentiment_score or 0.0,
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