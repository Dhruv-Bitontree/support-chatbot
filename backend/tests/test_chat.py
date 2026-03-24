"""Tests for chat orchestrator and intent classification."""

import pytest
import pytest_asyncio

from app.models.chat import ChatRequest, Intent
from app.services.chat.intent import classify_intent
from app.services.chat.orchestrator import ChatOrchestrator


class TestIntentClassification:
    def test_greeting(self):
        intent, _, _ = classify_intent("Hello!")
        assert intent == Intent.GREETING
        intent, _, _ = classify_intent("Hi there")
        assert intent == Intent.GREETING
        intent, _, _ = classify_intent("Good morning")
        assert intent == Intent.GREETING

    def test_order_tracking(self):
        intent, _, _ = classify_intent("Where is my order?")
        assert intent == Intent.ORDER_TRACKING
        intent, _, _ = classify_intent("Track ORD-1001")
        assert intent == Intent.ORDER_TRACKING
        intent, _, _ = classify_intent("What's my delivery status?")
        assert intent == Intent.ORDER_TRACKING

    def test_complaint(self):
        intent, _, _ = classify_intent("I'm very unhappy with your service")
        assert intent == Intent.COMPLAINT
        intent, _, _ = classify_intent("This is terrible and I'm very disappointed")
        assert intent == Intent.COMPLAINT
        intent, _, _ = classify_intent("This product is broken and I'm frustrated")
        assert intent == Intent.COMPLAINT

    def test_faq(self):
        intent, _, _ = classify_intent("What is your return policy?")
        assert intent == Intent.FAQ
        intent, _, _ = classify_intent("How do I contact support?")
        assert intent == Intent.FAQ
        intent, _, _ = classify_intent("What are your shipping options?")
        assert intent == Intent.FAQ

    def test_general(self):
        intent, _, _ = classify_intent("Tell me a joke")
        assert intent == Intent.GENERAL


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
        """Test complaint flow with email-first requirement.
        
        Updated to reflect new behavior: complaints now offer options first,
        then collect email before creating ticket.
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Step 1: Send complaint - should offer options
        request1 = ChatRequest(
            message="This is terrible! I want a refund immediately!", channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        
        assert response1.intent == Intent.COMPLAINT
        assert response1.metadata is not None
        assert "frustration_detected" in response1.metadata
        # Should NOT have ticket_id yet - needs to go through email collection
        assert "ticket_id" not in response1.metadata
        
        # Step 2: Choose to create ticket
        request2 = ChatRequest(
            message="1", session_id=response1.session_id, channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        
        # Should ask for email
        assert "email" in response2.message.lower()
        
        # Step 3: Provide email
        request3 = ChatRequest(
            message="user@example.com", session_id=response1.session_id, channel="test"
        )
        response3 = await orchestrator.handle_message(request3)
        
        # NOW should have ticket_id
        assert response3.metadata is not None
        assert "ticket_id" in response3.metadata

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
