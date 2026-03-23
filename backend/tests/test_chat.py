"""Tests for chat orchestrator and intent classification."""

import pytest
import pytest_asyncio

from app.models.chat import ChatRequest, Intent
from app.services.chat.intent import classify_intent
from app.services.chat.orchestrator import ChatOrchestrator


class TestIntentClassification:
    def test_greeting(self):
        assert classify_intent("Hello!") == Intent.GREETING
        assert classify_intent("Hi there") == Intent.GREETING
        assert classify_intent("Good morning") == Intent.GREETING

    def test_order_tracking(self):
        assert classify_intent("Where is my order?") == Intent.ORDER_TRACKING
        assert classify_intent("Track ORD-1001") == Intent.ORDER_TRACKING
        assert classify_intent("What's my delivery status?") == Intent.ORDER_TRACKING

    def test_complaint(self):
        assert classify_intent("I'm very unhappy with your service") == Intent.COMPLAINT
        assert classify_intent("I want a refund") == Intent.COMPLAINT
        assert classify_intent("This product is broken") == Intent.COMPLAINT

    def test_faq(self):
        assert classify_intent("What is your return policy?") == Intent.FAQ
        assert classify_intent("How do I contact support?") == Intent.FAQ
        assert classify_intent("What are your shipping options?") == Intent.FAQ

    def test_general(self):
        assert classify_intent("Tell me a joke") == Intent.GENERAL


@pytest.mark.asyncio
class TestChatOrchestrator:
    async def test_greeting_response(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request = ChatRequest(message="Hello!", channel="test")
        response = await orchestrator.handle_message(request)

        assert response.intent == Intent.GREETING
        assert response.session_id
        assert "Welcome" in response.message

    async def test_order_tracking_asks_for_id(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request = ChatRequest(message="Where is my order?", channel="test")
        response = await orchestrator.handle_message(request)

        assert response.intent == Intent.ORDER_TRACKING
        assert "order ID" in response.message or "order id" in response.message.lower()

    async def test_complaint_creates_ticket(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request = ChatRequest(
            message="This is terrible! I want a refund immediately!", channel="test"
        )
        response = await orchestrator.handle_message(request)

        assert response.intent == Intent.COMPLAINT
        assert response.metadata is not None
        assert "ticket_id" in response.metadata

    async def test_session_persistence(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request1 = ChatRequest(message="Hello!", channel="test")
        response1 = await orchestrator.handle_message(request1)

        request2 = ChatRequest(
            message="Thanks!", session_id=response1.session_id, channel="test"
        )
        response2 = await orchestrator.handle_message(request2)

        assert response1.session_id == response2.session_id
