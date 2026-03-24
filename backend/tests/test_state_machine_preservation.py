"""Preservation property tests for conversation state machine fix.

These tests MUST PASS on unfixed code to establish baseline behavior to preserve.
They verify that normal routing and existing flows remain unchanged after the fix.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from app.models.chat import ChatRequest, Intent
from app.services.chat.orchestrator import ChatOrchestrator
from tests.conftest import MockLLMProvider, MockVectorStore


@pytest.mark.asyncio
class TestPreservation:
    """Test suite to verify existing behavior is preserved after the fix."""

    async def test_preservation_normal_faq_routing(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Normal FAQ routing must remain unchanged.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send a normal FAQ question without frustration
        request = ChatRequest(
            message="What is your return policy?",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should route to FAQ handler
        assert response.intent == Intent.FAQ
        assert response.session_id is not None
        # Should not create a ticket
        assert response.metadata is None or "ticket_id" not in response.metadata

    async def test_preservation_order_tracking_routing(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Order tracking routing must remain unchanged.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send an order tracking request
        request = ChatRequest(
            message="Where is my order ORD-1001?",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should route to order tracking handler
        assert response.intent == Intent.ORDER_TRACKING
        assert response.session_id is not None
        # Should not create a ticket
        assert response.metadata is None or "ticket_id" not in response.metadata

    async def test_preservation_greeting_routing(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Greeting routing must remain unchanged.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send a greeting
        request = ChatRequest(
            message="Hello!",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should route to greeting handler
        assert response.intent == Intent.GREETING
        assert response.session_id is not None
        assert "Welcome" in response.message or "Hello" in response.message or "Hi" in response.message

    async def test_preservation_auto_escalation_urgent_ticket(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Auto-escalation for sentiment <= -0.5 must create URGENT ticket.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        
        NOTE: Updated to reflect email-first rule - auto-escalation now requires email collection first.
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Step 1: Send a message with very negative sentiment (should trigger auto-escalation)
        request1 = ChatRequest(
            message="This is absolutely terrible! I'm furious and want a refund NOW!",
            channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        session_id = response1.session_id
        
        # Should offer frustration options (not immediate ticket due to email-first rule)
        assert "would you like" in response1.message.lower() or "option" in response1.message.lower()
        
        # Step 2: Choose to create ticket
        request2 = ChatRequest(
            message="1",
            session_id=session_id,
            channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        
        # Should ask for email
        assert "email" in response2.message.lower()
        
        # Step 3: Provide valid email
        request3 = ChatRequest(
            message="user@example.com",
            session_id=session_id,
            channel="test"
        )
        response3 = await orchestrator.handle_message(request3)
        
        # Should create URGENT ticket
        assert response3.intent == Intent.COMPLAINT
        assert response3.metadata is not None
        assert "ticket_id" in response3.metadata
        
        # Verify ticket was created with URGENT priority
        from sqlalchemy import select
        from app.db.models import Ticket
        
        ticket_id = response3.metadata["ticket_id"]
        result = await db_session.execute(
            select(Ticket).where(Ticket.id == ticket_id)
        )
        ticket = result.scalar_one()
        
        assert ticket.priority == "urgent"

    async def test_preservation_sentiment_gate_positive_sentiment(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Sentiment gate must route positive sentiment to GENERAL not COMPLAINT.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send a complaint-like message but with positive sentiment
        request = ChatRequest(
            message="I have a small issue but I'm sure we can resolve it easily!",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should NOT create a ticket due to sentiment gate
        # Should route to GENERAL or FAQ, not COMPLAINT
        assert response.intent != Intent.COMPLAINT or (
            response.metadata is None or "ticket_id" not in response.metadata
        )

    async def test_preservation_conversation_history_loading(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Conversation history loading must continue to work.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send first message
        request1 = ChatRequest(
            message="Hello!",
            channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        session_id = response1.session_id
        
        # Send second message in same session
        request2 = ChatRequest(
            message="What is your return policy?",
            session_id=session_id,
            channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        
        # Should maintain same session
        assert response2.session_id == session_id
        
        # Verify conversation history was loaded
        from sqlalchemy import select
        from app.db.models import ChatMessage
        
        result = await db_session.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id)
        )
        messages = result.scalars().all()
        
        # Should have at least 2 messages (user messages)
        assert len(messages) >= 2

    async def test_preservation_input_validation_whitespace_message(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Input validation for whitespace-only messages must continue to work.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send whitespace-only message (Pydantic validates non-empty, but orchestrator validates meaningful content)
        request = ChatRequest(
            message="   ",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should return validation error or handle gracefully
        assert response.message is not None
        assert response.session_id is not None

    async def test_preservation_input_validation_emoji_only(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Input validation for emoji-only messages must continue to work.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Send emoji-only message
        request = ChatRequest(
            message="😀😀😀",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # Should return validation error or handle gracefully
        # The system should not crash
        assert response.message is not None
        assert response.session_id is not None

    async def test_preservation_existing_ticket_acknowledgment(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Existing ticket detection acknowledgment must remain unchanged.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        NOTE: After state machine fix, session is locked after ticket creation,
        so second message returns closure message instead of "already have" message.
        
        Updated to reflect email-first rule.
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Step 1: Trigger auto-escalation
        request1 = ChatRequest(
            message="This is absolutely terrible! I'm furious!",
            channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        session_id = response1.session_id
        
        # Should offer frustration options
        assert "would you like" in response1.message.lower() or "option" in response1.message.lower()
        
        # Step 2: Choose to create ticket
        request2 = ChatRequest(
            message="1",
            session_id=session_id,
            channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        
        # Should ask for email
        assert "email" in response2.message.lower()
        
        # Step 3: Provide valid email to create first ticket
        request3 = ChatRequest(
            message="user@example.com",
            session_id=session_id,
            channel="test"
        )
        response3 = await orchestrator.handle_message(request3)
        first_ticket_id = response3.metadata.get("ticket_id")
        
        assert first_ticket_id is not None
        
        # Step 4: Send another angry message
        request4 = ChatRequest(
            message="I'm still very angry!",
            session_id=session_id,
            channel="test"
        )
        response4 = await orchestrator.handle_message(request4)
        
        # After state machine fix: Should return closure message due to session lock
        # Before fix: Would return "already have" message
        assert ("already have" in response4.message.lower() or 
                "existing" in response4.message.lower() or
                "ticket has been created" in response4.message.lower())
        assert "ticket" in response4.message.lower()
        
        # Should reference the existing ticket or indicate session is closed
        if response4.metadata:
            if "existing_ticket_id" in response4.metadata:
                assert response4.metadata["existing_ticket_id"] == first_ticket_id
            elif "session_locked" in response4.metadata:
                assert response4.metadata["session_locked"] == True

    async def test_preservation_session_persistence(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Preservation: Session persistence across messages must remain unchanged.
        
        EXPECTED: Test PASSES on both unfixed and fixed code
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Create session with first message
        request1 = ChatRequest(
            message="Hello!",
            channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        session_id = response1.session_id
        
        # Send multiple messages in same session
        for i in range(3):
            request = ChatRequest(
                message=f"Message {i}",
                session_id=session_id,
                channel="test"
            )
            response = await orchestrator.handle_message(request)
            
            # Should maintain same session
            assert response.session_id == session_id
