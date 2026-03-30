"""Regression tests for stricter order validation and expanded support context."""

from datetime import datetime

import pytest
from sqlalchemy import select

from app.db.models import Order
from app.models.chat import ChatRequest, Intent, MessageRole
from app.services.chat.orchestrator import ChatOrchestrator
from app.services.orders.order_id_utils import looks_like_explicit_order_reference, normalize_order_id


async def _seed_shipped_order(db_session, order_id: str = "ORD-1001") -> None:
    order = Order(
        id=order_id,
        customer_email="buyer@example.com",
        status="shipped",
        items=[{"name": "Widget", "quantity": 1, "price": 29.99}],
        total=29.99,
        tracking_number="TRK-12345",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(order)
    await db_session.flush()


@pytest.mark.asyncio
class TestOrderNormalizationEnhancements:
    async def test_normalize_order_id_variants(self):
        assert normalize_order_id("ORD-1001") == "ORD-1001"
        assert normalize_order_id("ord-1001") == "ORD-1001"
        assert normalize_order_id("ORD1001") == "ORD-1001"
        assert normalize_order_id("ord1001") == "ORD-1001"
        assert normalize_order_id("order 1001") == "ORD-1001"
        assert normalize_order_id("#1001") == "ORD-1001"
        assert normalize_order_id("1001") == "ORD-1001"
        assert normalize_order_id("i paid 1200 rupees") is None
        assert normalize_order_id("2 days ago") is None
        assert normalize_order_id("my address has flat 1221") is None
        assert looks_like_explicit_order_reference("i paid 1200 rupees") is False
        assert looks_like_explicit_order_reference("2 days ago") is False
        assert looks_like_explicit_order_reference("my address has flat 1221") is False

    @pytest.mark.parametrize("order_input", ["ORD-1001", "1001", "ORD1001", "ord-1001", "#1001", "order 1001"])
    async def test_order_flow_accepts_normalized_variants(
        self,
        order_input,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        await _seed_shipped_order(db_session, "ORD-1001")
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(ChatRequest(message="where is my order?", channel="test"))
        assert first.intent == Intent.ORDER_TRACKING
        assert "order id" in first.message.lower()

        second = await orchestrator.handle_message(
            ChatRequest(message=order_input, session_id=first.session_id, channel="test")
        )
        assert second.intent == Intent.ORDER_TRACKING
        assert second.metadata is not None
        assert second.metadata.get("order_id") == "ORD-1001"
        assert "ORD-1001" in second.message

    async def test_nonexistent_order_stays_in_retry_state(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(ChatRequest(message="track my order", channel="test"))
        assert first.intent == Intent.ORDER_TRACKING

        second = await orchestrator.handle_message(
            ChatRequest(message="9999", session_id=first.session_id, channel="test")
        )
        assert second.intent == Intent.ORDER_TRACKING
        assert second.metadata is not None
        assert second.metadata.get("found") is False
        assert second.metadata.get("awaiting_order_id") is True
        assert "ORD-9999" in second.message
        assert "ORD-1001" in second.message

        metadata = await orchestrator._get_session_metadata(first.session_id)
        assert metadata.get("active_order_confirmed") is False
        assert metadata.get("active_order_id") is None
        assert metadata.get("awaiting_order_id") is True

    async def test_repeated_invalid_order_reply_gets_more_guidance(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(ChatRequest(message="track my order", channel="test"))
        second = await orchestrator.handle_message(
            ChatRequest(message="1221", session_id=first.session_id, channel="test")
        )
        third = await orchestrator.handle_message(
            ChatRequest(message="1221", session_id=first.session_id, channel="test")
        )
        fourth = await orchestrator.handle_message(
            ChatRequest(message="1221", session_id=first.session_id, channel="test")
        )

        assert "ORD-1221" in second.message
        assert "confirmation email or sms" in third.message.lower()
        assert "return, refund, or general support question" in fourth.message.lower()

    @pytest.mark.parametrize(
        "message",
        [
            "i paid 1200 rupees",
            "2 days ago",
            "my address has flat 1221",
        ],
    )
    async def test_general_numbers_do_not_activate_order_lookup(
        self,
        message,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        response = await orchestrator.handle_message(ChatRequest(message=message, channel="test"))

        assert response.intent != Intent.ORDER_TRACKING
        assert "ORD-" not in response.message
        assert not (response.metadata or {}).get("order_id")

    async def test_refund_status_query_stays_operational_when_natural_language_is_clear(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        response = await orchestrator.handle_message(
            ChatRequest(message="my refund has not come yet", channel="test")
        )

        assert response.intent == Intent.ORDER_TRACKING
        assert response.message

    async def test_category_collection_accepts_last_order_phrase(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(
            ChatRequest(message="create a ticket", channel="test")
        )
        assert first.metadata is not None
        assert first.metadata.get("awaiting_issue_category") is True

        second = await orchestrator.handle_message(
            ChatRequest(message="this is about my last order", session_id=first.session_id, channel="test")
        )

        assert second.intent == Intent.COMPLAINT
        assert second.metadata is not None
        assert second.metadata.get("awaiting_order_id") is True
        assert "order id" in second.message.lower()

    async def test_confirmed_order_context_survives_one_non_order_turn_then_clears(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        await _seed_shipped_order(db_session, "ORD-1001")
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(ChatRequest(message="where is my order?", channel="test"))
        second = await orchestrator.handle_message(
            ChatRequest(message="1001", session_id=first.session_id, channel="test")
        )
        assert second.metadata is not None
        assert second.metadata.get("order_id") == "ORD-1001"

        ack = await orchestrator.handle_message(
            ChatRequest(message="okay", session_id=first.session_id, channel="test")
        )
        assert ack.intent == Intent.GENERAL

        metadata_after_ack = await orchestrator._get_session_metadata(first.session_id)
        assert metadata_after_ack.get("active_order_confirmed") is True
        assert metadata_after_ack.get("active_order_id") == "ORD-1001"
        assert metadata_after_ack.get("active_order_context_idle_turns") == 1

        await orchestrator.handle_message(
            ChatRequest(message="tell me a joke", session_id=first.session_id, channel="test")
        )
        metadata_after_second_turn = await orchestrator._get_session_metadata(first.session_id)
        assert metadata_after_second_turn.get("active_order_confirmed") is False
        assert metadata_after_second_turn.get("active_order_id") is None

    async def test_ticket_inquiry_resyncs_existing_ticket_metadata(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        start = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        await orchestrator.handle_message(
            ChatRequest(message="7", session_id=start.session_id, channel="test")
        )
        await orchestrator.handle_message(
            ChatRequest(
                message="The support team was dismissive and I still need help.",
                session_id=start.session_id,
                channel="test",
            )
        )
        done = await orchestrator.handle_message(
            ChatRequest(message="agent@example.com", session_id=start.session_id, channel="test")
        )
        assert done.metadata is not None
        ticket_id = done.metadata["ticket_id"]

        await orchestrator._update_session_metadata(
            start.session_id,
            {"has_open_ticket": False, "ticket_id": None, "customer_email": None},
        )

        inquiry = await orchestrator.handle_message(
            ChatRequest(message="do i have any ticket", session_id=start.session_id, channel="test")
        )
        assert inquiry.metadata is not None
        assert inquiry.metadata.get("has_open_ticket") is True
        assert inquiry.metadata.get("ticket_id") == ticket_id
        assert inquiry.metadata.get("customer_email") == "agent@example.com"

        session_metadata = await orchestrator._get_session_metadata(start.session_id)
        assert session_metadata.get("has_open_ticket") is True
        assert session_metadata.get("ticket_id") == ticket_id
        assert session_metadata.get("customer_email") == "agent@example.com"


@pytest.mark.asyncio
class TestSupportContextEnhancements:
    async def test_history_loader_uses_twelve_messages(self, mock_llm, mock_vector_store, db_session):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)
        session_id = "history-limit-session"
        await orchestrator._ensure_session(session_id, "test")

        for idx in range(15):
            await orchestrator._store_message(
                session_id,
                MessageRole.USER if idx % 2 == 0 else MessageRole.ASSISTANT,
                f"message {idx}",
                Intent.GENERAL,
                None,
            )

        history = await orchestrator._load_history(session_id)
        assert len(history) == 12
        assert history[0].content == "message 3"
        assert history[-1].content == "message 14"

    async def test_general_llm_call_receives_support_context_and_twelve_history_messages(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)
        session_id = "general-context-session"
        await orchestrator._ensure_session(session_id, "test")

        for idx in range(14):
            await orchestrator._store_message(
                session_id,
                MessageRole.USER if idx % 2 == 0 else MessageRole.ASSISTANT,
                f"message {idx}",
                Intent.GENERAL,
                None,
            )

        response = await orchestrator.handle_message(
            ChatRequest(message="tell me more", session_id=session_id, channel="test")
        )
        assert response.intent == Intent.GENERAL
        assert mock_llm.calls

        last_call = mock_llm.calls[-1]
        assert "Support context:" in last_call["system_prompt"]
        assert len(last_call["messages"]) == 13

    async def test_support_context_clears_issue_state_on_topic_switch(
        self,
        mock_llm,
        mock_vector_store,
        db_session,
    ):
        orchestrator = ChatOrchestrator(llm=mock_llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(
            ChatRequest(message="create a ticket for my refund", channel="test")
        )
        assert first.metadata is not None
        assert first.metadata.get("awaiting_order_id") is True

        faq = await orchestrator.handle_message(
            ChatRequest(
                message="What is your return policy?",
                session_id=first.session_id,
                channel="test",
            )
        )
        assert faq.intent == Intent.FAQ

        last_call = mock_llm.calls[-1]
        assert '"issue_category": null' in last_call["system_prompt"]
        assert '"active_order_id": null' in last_call["system_prompt"]
