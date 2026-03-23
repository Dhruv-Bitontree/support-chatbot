"""Tests for complaint/ticket service and sentiment analysis."""

import pytest

from app.models.complaint import ComplaintRequest, TicketPriority, TicketStatus
from app.services.complaints.sentiment import analyze_sentiment
from app.services.complaints.ticket_service import TicketService


class TestSentimentAnalysis:
    def test_positive_sentiment(self):
        result = analyze_sentiment("I love your service! Everything was great!")
        assert result.label == "positive"
        assert result.score > 0

    def test_negative_sentiment(self):
        result = analyze_sentiment("This is terrible. Worst experience ever.")
        assert result.label == "negative"
        assert result.score < 0

    def test_neutral_sentiment(self):
        result = analyze_sentiment("I placed an order yesterday.")
        assert result.label == "neutral"

    def test_very_negative_escalation(self):
        result = analyze_sentiment(
            "This is absolutely unacceptable! I'm furious! You ruined everything!"
        )
        assert result.label == "negative"
        assert result.score < -0.5


@pytest.mark.asyncio
class TestTicketService:
    async def test_create_ticket(self, db_session):
        service = TicketService(db_session)
        request = ComplaintRequest(
            message="The product arrived broken. Very disappointed.",
            customer_email="angry@example.com",
        )
        ticket = await service.create_ticket(request, category="product_damage")

        assert ticket.id
        assert ticket.category == "product_damage"
        assert ticket.sentiment.label == "negative"
        assert ticket.priority in [TicketPriority.HIGH, TicketPriority.URGENT]

    async def test_urgent_escalation(self, db_session):
        service = TicketService(db_session)
        request = ComplaintRequest(
            message="This is the worst service I have ever experienced! Absolutely terrible and unacceptable! I am furious!",
        )
        ticket = await service.create_ticket(request, category="complaint")

        assert ticket.priority == TicketPriority.URGENT
        assert ticket.status == TicketStatus.ESCALATED

    async def test_low_priority_positive(self, db_session):
        service = TicketService(db_session)
        request = ComplaintRequest(
            message="Just a small suggestion - it would be nice to have more color options.",
        )
        ticket = await service.create_ticket(request, category="feedback")

        assert ticket.priority in [TicketPriority.LOW, TicketPriority.MEDIUM]

    async def test_get_ticket(self, db_session):
        service = TicketService(db_session)
        request = ComplaintRequest(message="Item was missing from package.")
        created = await service.create_ticket(request)

        retrieved = await service.get_ticket(created.id)
        assert retrieved.id == created.id
        assert retrieved.description == "Item was missing from package."
