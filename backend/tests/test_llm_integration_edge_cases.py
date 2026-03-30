"""Edge-case tests for hybrid LLM + rule-based chat orchestration.

Focus:
- LLM composer should improve phrasing when safe.
- Guardrails should reject unsafe rewrites and fall back to deterministic drafts.
- Session history should be included in composer calls.
- Real-world intent edge cases should route correctly.
"""

import pytest

from app.models.chat import ChatRequest, Intent
from app.services.chat.intent import classify_intent
from app.services.chat.orchestrator import ChatOrchestrator
from tests.conftest import MockLLMProvider


class DraftAwareLLMProvider(MockLLMProvider):
    """Mock LLM that can return safe/unsafe rewrites for composer calls."""

    def __init__(self, mode: str = "safe"):
        super().__init__(response="Okay, I can help with that.")
        self.mode = mode
        self.composer_calls: list[dict] = []

    @staticmethod
    def _extract_draft(system_prompt: str) -> str:
        marker = "Draft reply:\n"
        if marker not in system_prompt:
            return ""
        return system_prompt.split(marker, 1)[1].strip()

    async def generate(self, messages, system_prompt="", **kwargs) -> str:  # noqa: ANN001
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        if "refining a customer-support reply" in (system_prompt or "").lower():
            draft = self._extract_draft(system_prompt or "")
            self.composer_calls.append(
                {"messages": messages, "system_prompt": system_prompt, "draft": draft}
            )
            if self.mode == "unsafe":
                return "I will take care of everything."
            if self.mode == "empty":
                return ""
            return f"Sure. {draft}"
        return self.response


async def _create_ticket_in_session(orchestrator: ChatOrchestrator) -> tuple[str, str]:
    start = await orchestrator.handle_message(
        ChatRequest(message="create a new ticket for me", channel="test")
    )
    assert "what is this about" in start.message.lower()

    category = await orchestrator.handle_message(
        ChatRequest(message="7", session_id=start.session_id, channel="test")
    )
    assert category.metadata is not None
    assert category.metadata.get("awaiting_issue_summary") is True

    summary = await orchestrator.handle_message(
        ChatRequest(
            message="The support team was dismissive and I still need help.",
            session_id=start.session_id,
            channel="test",
        )
    )
    assert "email" in summary.message.lower()
    done = await orchestrator.handle_message(
        ChatRequest(message="anb@gmail.com", session_id=start.session_id, channel="test")
    )
    assert done.metadata is not None and "ticket_id" in done.metadata
    return start.session_id, done.metadata["ticket_id"]


class TestIntentEdgeCases:
    @pytest.mark.parametrize(
        ("text", "expected_intent"),
        [
            ("your service is amazing lol my package is late again", Intent.COMPLAINT),
            ("can you check my order status ORD-1111", Intent.ORDER_TRACKING),
            ("Can I change my shipping address after ordering?", Intent.FAQ),
            ("hello there", Intent.GREETING),
        ],
    )
    def test_real_world_intent_edge_cases(self, text: str, expected_intent: Intent):
        intent, _, _ = classify_intent(text)
        assert intent == expected_intent


@pytest.mark.asyncio
class TestLLMComposerGuardrails:
    async def test_safe_rewrite_applied_for_ticket_inquiry(self, mock_vector_store, db_session):
        llm = DraftAwareLLMProvider(mode="safe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)
        session_id, ticket_id = await _create_ticket_in_session(orchestrator)

        inquiry = await orchestrator.handle_message(
            ChatRequest(message="do i have any ticket", session_id=session_id, channel="test")
        )
        assert inquiry.intent == Intent.COMPLAINT
        assert inquiry.message.lower().startswith("sure.")
        assert f"#{ticket_id[:8]}" in inquiry.message
        assert "anb@gmail.com" in inquiry.message

    async def test_unsafe_rewrite_falls_back_for_ticket_inquiry(self, mock_vector_store, db_session):
        llm = DraftAwareLLMProvider(mode="unsafe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)
        session_id, ticket_id = await _create_ticket_in_session(orchestrator)

        inquiry = await orchestrator.handle_message(
            ChatRequest(message="do i have any ticket", session_id=session_id, channel="test")
        )
        assert inquiry.intent == Intent.COMPLAINT
        assert inquiry.message.startswith("We already created support ticket")
        assert f"#{ticket_id[:8]}" in inquiry.message
        assert "anb@gmail.com" in inquiry.message
        assert inquiry.message != "I will take care of everything."

    async def test_unsafe_rewrite_cannot_remove_support_options(self, mock_vector_store, db_session):
        llm = DraftAwareLLMProvider(mode="unsafe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)

        first = await orchestrator.handle_message(
            ChatRequest(message="I am disappointed with your service", channel="test")
        )
        assert first.intent == Intent.COMPLAINT
        assert "1." in first.message
        assert "7." in first.message
        assert "reply with the number" in first.message.lower()

    async def test_unsafe_rewrite_cannot_remove_extracted_email_confirmation(
        self, mock_vector_store, db_session
    ):
        llm = DraftAwareLLMProvider(mode="unsafe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)

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
        email_turn = await orchestrator.handle_message(
            ChatRequest(
                message="my emai is test@gmail.com",
                session_id=start.session_id,
                channel="test",
            )
        )

        assert '"test@gmail.com"' in email_turn.message
        assert "yes" in email_turn.message.lower()
        assert "change" in email_turn.message.lower()

    async def test_unsafe_rewrite_cannot_remove_order_not_found_ids(
        self, mock_vector_store, db_session
    ):
        llm = DraftAwareLLMProvider(mode="unsafe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)

        response = await orchestrator.handle_message(
            ChatRequest(message="track ORD-9999", channel="test")
        )
        assert response.intent == Intent.ORDER_TRACKING
        assert "ORD-9999" in response.message
        assert "ORD-1001" in response.message

    async def test_composer_gets_session_history_context(self, mock_vector_store, db_session):
        llm = DraftAwareLLMProvider(mode="safe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)
        session_id, _ = await _create_ticket_in_session(orchestrator)

        await orchestrator.handle_message(
            ChatRequest(message="do i have any ticket", session_id=session_id, channel="test")
        )

        inquiry_calls = [
            c for c in llm.composer_calls if c["draft"].startswith("We already created support ticket")
        ]
        assert inquiry_calls
        messages = inquiry_calls[-1]["messages"]
        contents = [m.content for m in messages]
        assert len(contents) >= 2
        assert any("anb@gmail.com" in c for c in contents)
        assert any("do i have any ticket" in c.lower() for c in contents)

    async def test_composer_does_not_repeat_support_context_block(self, mock_vector_store, db_session):
        llm = DraftAwareLLMProvider(mode="safe")
        orchestrator = ChatOrchestrator(llm=llm, vector_store=mock_vector_store, db=db_session)

        start = await orchestrator.handle_message(
            ChatRequest(message="create a new ticket for me", channel="test")
        )
        assert start.intent == Intent.COMPLAINT

        composer_calls = [
            c for c in llm.composer_calls if "what is this about" in c["draft"].lower()
        ]
        assert composer_calls
        system_prompt = composer_calls[-1]["system_prompt"]
        assert system_prompt.count("Support context:") <= 1
