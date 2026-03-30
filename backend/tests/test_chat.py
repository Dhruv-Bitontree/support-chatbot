"""Tests for chat orchestrator and intent classification."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.db.models import Order, Ticket
from app.models.chat import ChatRequest, Intent
from app.models.faq import FAQEntry
from app.services.chat.intent import FrustrationLevel, classify_intent
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

    def test_acknowledgements_stay_general(self):
        intent, frustration, sarcasm = classify_intent("okay")
        assert intent == Intent.GENERAL
        assert frustration == FrustrationLevel.NONE
        assert sarcasm is False

        intent, _, _ = classify_intent("thanks")
        assert intent == Intent.GENERAL

    def test_neutral_refund_status_query_maps_to_order_tracking(self):
        intent, frustration, sarcasm = classify_intent(
            "my refund has not arrived yet, can you check?"
        )
        assert intent == Intent.ORDER_TRACKING
        assert frustration == FrustrationLevel.NONE
        assert sarcasm is False

    def test_neutral_not_delivered_phrase_maps_to_order_tracking(self):
        intent, frustration, sarcasm = classify_intent("my order is still not delivered")
        assert intent == Intent.ORDER_TRACKING
        assert frustration == FrustrationLevel.NONE
        assert sarcasm is False

    def test_positive_all_caps_without_negative_context_is_not_high_frustration(self):
        intent, frustration, sarcasm = classify_intent("I LOVE THIS")
        assert intent == Intent.GENERAL
        assert frustration == FrustrationLevel.NONE
        assert sarcasm is False

    def test_sarcasm_maps_to_complaint_not_order_tracking(self):
        intent, _, sarcasm = classify_intent(
            "your services is so good i am never getting my order on time hahaha"
        )
        assert intent == Intent.COMPLAINT
        assert sarcasm is True

    def test_legal_threat_with_order_words_maps_to_complaint(self):
        intent, _, _ = classify_intent(
            "order is delivered but i have lost it i will do legal action on you guys"
        )
        assert intent == Intent.COMPLAINT


async def _seed_order(db_session, order_id: str = "ORD-1001") -> None:
    order = Order(
        id=order_id,
        customer_email="buyer@example.com",
        status="processing",
        items=[{"name": "Widget", "quantity": 1, "price": 29.99}],
        total=29.99,
        tracking_number=None,
        estimated_delivery="2026-03-30",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(order)
    await db_session.flush()


async def _complete_other_issue_ticket_flow(
    orchestrator: ChatOrchestrator,
    *,
    session_id: str,
    summary: str = "The support team was dismissive and I still need help.",
    email: str = "user@example.com",
) -> tuple:
    category = await orchestrator.handle_message(
        ChatRequest(message="7", session_id=session_id, channel="test")
    )
    assert category.metadata is not None
    assert category.metadata.get("awaiting_issue_summary") is True

    summary_turn = await orchestrator.handle_message(
        ChatRequest(message=summary, session_id=session_id, channel="test")
    )
    assert summary_turn.metadata is not None
    return category, summary_turn, await orchestrator.handle_message(
        ChatRequest(message=email, session_id=session_id, channel="test")
    )


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

    async def test_chat_response_timestamp_is_timezone_aware_utc(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        response = await orchestrator.handle_message(
            ChatRequest(message="Hello!", channel="test")
        )

        assert response.timestamp.tzinfo is not None
        assert response.timestamp.utcoffset() == timezone.utc.utcoffset(response.timestamp)

    async def test_order_tracking_asks_for_id(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request = ChatRequest(message="Where is my order?", channel="test")
        response = await orchestrator.handle_message(request)

        assert response.intent == Intent.ORDER_TRACKING
        assert "order ID" in response.message or "order id" in response.message.lower()

    async def test_complaint_creates_ticket(self, mock_llm, mock_vector_store, db_session):
        """Ticket creation now collects issue context and validates order linkage first."""
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        await _seed_order(db_session, "ORD-1001")

        request1 = ChatRequest(
            message="create a support ticket for my refund", channel="test"
        )
        response1 = await orchestrator.handle_message(request1)

        assert response1.intent == Intent.COMPLAINT
        assert response1.metadata is not None
        assert response1.metadata.get("awaiting_order_id") is True
        assert "ticket_id" not in response1.metadata

        request2 = ChatRequest(
            message="1001", session_id=response1.session_id, channel="test"
        )
        response2 = await orchestrator.handle_message(request2)
        assert response2.metadata is not None
        assert response2.metadata.get("awaiting_issue_summary") is True

        request3 = ChatRequest(
            message="The refund is still missing after my cancellation.",
            session_id=response1.session_id,
            channel="test",
        )
        response3 = await orchestrator.handle_message(request3)
        assert response3.metadata is not None
        assert response3.metadata.get("awaiting_email") is True

        request4 = ChatRequest(
            message="user@example.com", session_id=response1.session_id, channel="test"
        )
        response4 = await orchestrator.handle_message(request4)

        assert response4.metadata is not None
        assert "ticket_id" in response4.metadata
        assert response4.metadata.get("order_id") == "ORD-1001"
 
        result = await db_session.execute(
            select(Ticket).where(Ticket.session_id == response1.session_id)
        )
        ticket = result.scalar_one()
        assert ticket.order_id == "ORD-1001"

    async def test_awaiting_order_state_does_not_force_order_on_legal_threat(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        first = await orchestrator.handle_message(
            ChatRequest(message="where is my order?", channel="test")
        )
        assert first.intent == Intent.ORDER_TRACKING

        second = await orchestrator.handle_message(
            ChatRequest(
                message="order is delivered but i have lost it i will do legal action on you guys",
                session_id=first.session_id,
                channel="test",
            )
        )
        assert second.intent == Intent.COMPLAINT
        assert "what is this about" in second.message.lower()

    async def test_sarcastic_order_feedback_stays_calm_and_not_order_loop(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        response = await orchestrator.handle_message(
            ChatRequest(
                message="your services is so good i am never getting my order on time hahaha",
                channel="test",
            )
        )

        assert response.intent == Intent.COMPLAINT
        assert "what is this about" in response.message.lower()

    async def test_duplicate_ticket_request_returns_existing_ticket_and_email(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        first = await orchestrator.handle_message(
            ChatRequest(message="I am very disappointed with your service", channel="test")
        )
        assert "what is this about" in first.message.lower()

        category = await orchestrator.handle_message(
            ChatRequest(message="7", session_id=first.session_id, channel="test")
        )
        assert category.metadata is not None
        assert category.metadata.get("awaiting_issue_summary") is True

        summary = await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=first.session_id,
                channel="test",
            )
        )
        assert "1 or 2" in summary.message.lower()

        email_prompt = await orchestrator.handle_message(
            ChatRequest(message="1", session_id=first.session_id, channel="test")
        )
        assert "email" in email_prompt.message.lower()

        third = await orchestrator.handle_message(
            ChatRequest(message="anb@gmail.com", session_id=first.session_id, channel="test")
        )
        assert third.metadata is not None
        first_ticket_id = third.metadata["ticket_id"]

        duplicate = await orchestrator.handle_message(
            ChatRequest(
                message="creat a new ticket for me",
                session_id=first.session_id,
                channel="test",
            )
        )

        assert duplicate.intent == Intent.COMPLAINT
        assert f"#{first_ticket_id[:8]}" in duplicate.message
        assert '"anb@gmail.com"' in duplicate.message
        assert "same ticket" in duplicate.message.lower()
        assert duplicate.metadata is not None
        assert duplicate.metadata.get("existing_ticket_id") == first_ticket_id

        result = await db_session.execute(
            select(Ticket).where(Ticket.session_id == first.session_id)
        )
        tickets = result.scalars().all()
        assert len(tickets) == 1

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

    async def test_ticket_inquiry_not_misrouted_to_order(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )
        request = ChatRequest(message="do i have any ticket", channel="test")
        response = await orchestrator.handle_message(request)

        assert response.intent == Intent.COMPLAINT
        assert "order id" not in response.message.lower()
        assert response.metadata is not None
        assert "ticket_found" in response.metadata

    async def test_awaiting_order_state_allows_topic_switch(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        first = await orchestrator.handle_message(
            ChatRequest(message="where is my order?", channel="test")
        )
        assert first.intent == Intent.ORDER_TRACKING
        assert "order id" in first.message.lower()

        second = await orchestrator.handle_message(
            ChatRequest(
                message="i am disappointed with your services",
                session_id=first.session_id,
                channel="test",
            )
        )
        assert second.intent == Intent.COMPLAINT
        assert "order id" not in second.message.lower()

    async def test_email_collection_allows_topic_switch_to_faq(
        self, mock_llm, mock_vector_store, db_session
    ):
        mock_llm.response = "You can update your shipping address before dispatch."
        await mock_vector_store.upsert(
            [
                FAQEntry(
                    question="Can I change my shipping address after ordering?",
                    answer="Yes, you can request an address update before the order is dispatched.",
                    category="shipping",
                )
            ]
        )
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        start = await orchestrator.handle_message(
            ChatRequest(message="I am disappointed with your service", channel="test")
        )
        assert "what is this about" in start.message.lower()

        choose = await orchestrator.handle_message(
            ChatRequest(message="7", session_id=start.session_id, channel="test")
        )
        assert choose.metadata is not None
        assert choose.metadata.get("awaiting_issue_summary") is True

        details = await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert "1 or 2" in details.message.lower()

        choose_ticket = await orchestrator.handle_message(
            ChatRequest(message="1", session_id=start.session_id, channel="test")
        )
        assert "email" in choose_ticket.message.lower()

        bad_email = await orchestrator.handle_message(
            ChatRequest(message="6366363", session_id=start.session_id, channel="test")
        )
        assert "email format" in bad_email.message.lower()

        faq = await orchestrator.handle_message(
            ChatRequest(
                message="Can I change my shipping address after ordering?",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert faq.intent == Intent.FAQ
        assert "share your email" not in faq.message.lower()
        assert faq.metadata is not None
        assert faq.metadata.get("ticket_flow_cancelled") is True

    async def test_email_collection_exit_phrase_cancels_ticket_flow(
        self, mock_llm, mock_vector_store, db_session
    ):
        mock_llm.response = "Sure, what would you like to know?"
        await mock_vector_store.upsert(
            [
                FAQEntry(
                    question="Can I change my shipping address after ordering?",
                    answer="Yes, before dispatch.",
                    category="shipping",
                )
            ]
        )
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        start = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        assert "what is this about" in start.message.lower()

        choose_category = await orchestrator.handle_message(
            ChatRequest(message="7", session_id=start.session_id, channel="test")
        )
        assert choose_category.metadata is not None
        assert choose_category.metadata.get("awaiting_issue_summary") is True

        summary = await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert "email" in summary.message.lower()

        cancel = await orchestrator.handle_message(
            ChatRequest(
                message="no i do not want to complain just answer my question",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert cancel.intent == Intent.GENERAL
        assert "canceled ticket creation" in cancel.message.lower()
        assert cancel.metadata is not None
        assert cancel.metadata.get("ticket_flow_cancelled") is True

        faq = await orchestrator.handle_message(
            ChatRequest(
                message="Can I change my shipping address after ordering?",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert faq.intent == Intent.FAQ

    async def test_issue_collection_exit_phrase_cancels_ticket_flow(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        start = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        assert start.metadata is not None
        assert start.metadata.get("awaiting_issue_category") is True

        cancel = await orchestrator.handle_message(
            ChatRequest(
                message="I don't want to continue this",
                session_id=start.session_id,
                channel="test",
            )
        )
        assert cancel.intent == Intent.GENERAL
        assert cancel.metadata is not None
        assert cancel.metadata.get("ticket_flow_cancelled") is True
        assert "canceled" in cancel.message.lower()

    async def test_email_extraction_and_confirmation_before_ticket_creation(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        step1 = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        assert "what is this about" in step1.message.lower()

        step2 = await orchestrator.handle_message(
            ChatRequest(message="7", session_id=step1.session_id, channel="test")
        )
        assert step2.metadata is not None
        assert step2.metadata.get("awaiting_issue_summary") is True

        step3 = await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=step1.session_id,
                channel="test",
            )
        )
        assert "email" in step3.message.lower()

        step4 = await orchestrator.handle_message(
            ChatRequest(
                message="my emai is test@gmail.com",
                session_id=step1.session_id,
                channel="test",
            )
        )
        assert step4.intent == Intent.COMPLAINT
        assert '"test@gmail.com"' in step4.message
        assert "confirm" in step4.message.lower()
        assert step4.metadata is not None
        assert step4.metadata.get("awaiting_email_confirmation") is True

        step5 = await orchestrator.handle_message(
            ChatRequest(message="yes", session_id=step1.session_id, channel="test")
        )
        assert step5.intent == Intent.COMPLAINT
        assert step5.metadata is not None
        assert "ticket_id" in step5.metadata
        assert "test@gmail.com" in step5.message

        result = await db_session.execute(
            select(Ticket).where(Ticket.session_id == step1.session_id)
        )
        tickets = result.scalars().all()
        assert len(tickets) == 1

    async def test_email_confirmation_change_path_uses_updated_email(
        self, mock_llm, mock_vector_store, db_session
    ):
        orchestrator = ChatOrchestrator(
            llm=mock_llm, vector_store=mock_vector_store, db=db_session
        )

        step1 = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        await orchestrator.handle_message(
            ChatRequest(message="7", session_id=step1.session_id, channel="test")
        )
        await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=step1.session_id,
                channel="test",
            )
        )

        step3 = await orchestrator.handle_message(
            ChatRequest(
                message="my email is wrong@mail.com",
                session_id=step1.session_id,
                channel="test",
            )
        )
        assert step3.metadata is not None
        assert step3.metadata.get("awaiting_email_confirmation") is True

        step4 = await orchestrator.handle_message(
            ChatRequest(
                message="change",
                session_id=step1.session_id,
                channel="test",
            )
        )
        assert "share the email" in step4.message.lower()

        step5 = await orchestrator.handle_message(
            ChatRequest(
                message="use better@mail.com",
                session_id=step1.session_id,
                channel="test",
            )
        )
        assert '"better@mail.com"' in step5.message
        assert step5.metadata is not None
        assert step5.metadata.get("awaiting_email_confirmation") is True

        step6 = await orchestrator.handle_message(
            ChatRequest(message="yes", session_id=step1.session_id, channel="test")
        )
        assert step6.metadata is not None
        assert "ticket_id" in step6.metadata
        assert "better@mail.com" in step6.message
