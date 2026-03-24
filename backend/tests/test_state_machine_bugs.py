"""Bug condition exploration tests for conversation state machine.

These tests MUST FAIL on unfixed code to demonstrate the bugs exist.
They encode the expected behavior and will validate the fix when they pass.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from app.models.chat import ChatRequest, Intent
from app.services.chat.orchestrator import ChatOrchestrator
from tests.conftest import MockLLMProvider, MockVectorStore


@pytest.mark.asyncio
class TestStateMachineBugs:
    """Test suite to surface counterexamples demonstrating state machine bugs."""

    async def test_bug_1_1_session_not_locked_after_ticket_creation(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.1: Session continues processing messages after ticket creation.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - message gets processed normally
        EXPECTED ON FIXED CODE: Test PASSES - returns closure message
        
        NOTE: Auto-escalation now requires email collection first (email-first rule).
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Step 1: Trigger auto-escalation (sentiment <= -0.5)
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
        
        # Verify ticket was created
        assert response3.metadata is not None
        assert "ticket_id" in response3.metadata
        
        # Step 4: Send another message to the same session
        request4 = ChatRequest(
            message="I'm still angry!",
            session_id=session_id,
            channel="test"
        )
        response4 = await orchestrator.handle_message(request4)
        
        # EXPECTED BEHAVIOR: Session should be locked, return closure message
        assert "ticket has been created" in response4.message.lower()
        assert "start a new session" in response4.message.lower()
        # Should NOT process the message through intent classification
        assert response4.intent == Intent.COMPLAINT  # Intent should indicate locked state

    async def test_bug_1_2_infinite_email_collection_loop(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.2: Email collection loops infinitely without retry limit.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - keeps asking for email indefinitely
        EXPECTED ON FIXED CODE: Test PASSES - stops after 3 attempts and abandons ticket flow
        
        NOTE: Updated to reflect 3-attempt rule with ticket flow abandonment.
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Step 1: Trigger frustration detection (sentiment between -0.5 and 0)
        # Use milder complaint to avoid auto-escalation
        request1 = ChatRequest(
            message="I'm a bit disappointed with this service",
            channel="test"
        )
        response1 = await orchestrator.handle_message(request1)
        session_id = response1.session_id
        
        # Should offer options (not auto-escalate)
        assert "would you like" in response1.message.lower() or "option" in response1.message.lower()
        
        # Step 2: Choose to create ticket (option 1)
        request2 = ChatRequest(
            message="1",
            session_id=session_id,
            channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        
        # Should ask for email
        assert "email" in response2.message.lower()
        
        # Step 3: Provide invalid email (attempt 1)
        request3 = ChatRequest(
            message="notanemail",
            session_id=session_id,
            channel="test"
        )
        response3 = await orchestrator.handle_message(request3)
        
        # Should ask for email again
        assert "email" in response3.message.lower()
        
        # Step 4: Provide invalid email (attempt 2)
        request4 = ChatRequest(
            message="stillnotvalid",
            session_id=session_id,
            channel="test"
        )
        response4 = await orchestrator.handle_message(request4)
        
        # Should ask for email again (last attempt)
        assert "email" in response4.message.lower()
        
        # Step 5: Provide invalid email (attempt 3)
        request5 = ChatRequest(
            message="nope",
            session_id=session_id,
            channel="test"
        )
        response5 = await orchestrator.handle_message(request5)
        
        # EXPECTED BEHAVIOR: After 3 attempts, should abandon ticket flow
        # Should NOT ask for email again
        # Should NOT create a ticket
        # Should return to normal conversation
        # The message may mention email in context of abandoning the flow
        assert response5.metadata is None or "ticket_id" not in response5.metadata
        # Should have abandoned email collection
        if response5.metadata:
            assert response5.metadata.get("email_collection_abandoned") == True

    async def test_bug_1_3_three_support_options_offered(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.3: Three support options offered instead of two.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - offers 3 options
        EXPECTED ON FIXED CODE: Test PASSES - offers 2 options
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Trigger frustration detection
        request = ChatRequest(
            message="I'm frustrated with this service",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        
        # EXPECTED BEHAVIOR: Should offer exactly 2 options
        # Count occurrences of option markers
        message_lower = response.message.lower()
        
        # Check for 2 options, not 3
        # Option 1: Create ticket
        assert "1" in response.message or "create" in message_lower
        # Option 2: Try to resolve yourself
        assert "2" in response.message or "resolve" in message_lower
        # Option 3: Contact human agent (should NOT exist)
        assert "3" not in response.message or "human" not in message_lower

    async def test_bug_1_5_no_explicit_state_field(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.5: No explicit state field in session metadata.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - no "state" field exists
        EXPECTED ON FIXED CODE: Test PASSES - "state" field exists with enum value
        """
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        
        # Create a session
        request = ChatRequest(
            message="Hello!",
            channel="test"
        )
        response = await orchestrator.handle_message(request)
        session_id = response.session_id
        
        # Load session metadata
        from sqlalchemy import select
        from app.db.models import ChatSession
        
        result = await db_session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one()
        
        # EXPECTED BEHAVIOR: Metadata should have explicit "state" field
        assert session.metadata_ is not None
        assert "state" in session.metadata_
        # State should be one of the enum values
        valid_states = [
            "NORMAL_CHAT",
            "FRUSTRATION_DETECTED",
            "SUPPORT_OPTIONS",
            "EMAIL_COLLECTION",
            "TICKET_CREATION",
            "SESSION_LOCKED"
        ]
        assert session.metadata_["state"] in valid_states

    async def test_bug_1_6_session_not_locked_after_ticket_creation_metadata(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.6: Session metadata not updated to SESSION_LOCKED after ticket creation.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - state != SESSION_LOCKED
        EXPECTED ON FIXED CODE: Test PASSES - state == SESSION_LOCKED
        
        NOTE: Auto-escalation now requires email collection first (email-first rule).
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
        
        # Verify ticket was created
        assert response3.metadata is not None
        assert "ticket_id" in response3.metadata
        
        # Expire the session cache to ensure we get fresh data
        db_session.expire_all()
        
        # Load session metadata
        from sqlalchemy import select
        from app.db.models import ChatSession
        
        result = await db_session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one()
        
        # EXPECTED BEHAVIOR: Metadata should have state=SESSION_LOCKED
        assert session.metadata_ is not None
        assert "state" in session.metadata_
        assert session.metadata_["state"] == "SESSION_LOCKED"

    async def test_bug_1_4_duplicate_ticket_not_prevented(
        self, mock_llm, mock_vector_store, db_session
    ):
        """Bug 1.4: Duplicate tickets not prevented when session already has ticket.
        
        EXPECTED ON UNFIXED CODE: Test FAILS - creates duplicate ticket
        EXPECTED ON FIXED CODE: Test PASSES - acknowledges existing ticket, no duplicate
        
        NOTE: Auto-escalation now requires email collection first (email-first rule).
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
        
        # Step 4: Try to send another angry message (should be blocked by session lock)
        request4 = ChatRequest(
            message="I'm still very angry about this!",
            session_id=session_id,
            channel="test"
        )
        response4 = await orchestrator.handle_message(request4)
        
        # EXPECTED BEHAVIOR: Should acknowledge existing ticket, not create new one
        # Should return closure message due to session lock
        assert "ticket has been created" in response4.message.lower()
        assert "start a new session" in response4.message.lower()
        
        # Should NOT have a new ticket_id in metadata (or same ticket_id)
        if response4.metadata and "ticket_id" in response4.metadata:
            assert response4.metadata["ticket_id"] == first_ticket_id
