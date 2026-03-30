"""UI-level integration flow tests for chat API."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.database import get_db
from app.dependencies import get_faq_store, get_llm
from app.models.faq import FAQEntry
from main import create_app
from tests.conftest import MockLLMProvider, MockVectorStore


@pytest.mark.asyncio
async def test_ui_chat_flow_screenshot_sequence(db_session):
    """Mirror screenshot sequence: FAQ -> ticket inquiry -> complaint.

    Verifies we do not regress into incorrect order-ID prompts.
    """
    app = create_app()

    mock_llm = MockLLMProvider(response="We offer a 30-day return policy.")
    mock_store = MockVectorStore()
    await mock_store.upsert(
        [
            FAQEntry(
                question="What is your return policy?",
                answer="We offer a 30-day return policy for unused items in original packaging.",
                category="returns",
            )
        ]
    )

    async def override_get_llm():
        return mock_llm

    async def override_get_faq_store():
        return mock_store

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_llm] = override_get_llm
    app.dependency_overrides[get_faq_store] = override_get_faq_store
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1) FAQ question
            r1 = await client.post(
                "/api/chat",
                json={"message": "what is your return policy", "channel": "web"},
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["intent"] == "faq"
            session_id = b1["session_id"]

            # 2) Ticket inquiry should NOT route to order tracking
            r2 = await client.post(
                "/api/chat",
                json={"message": "do i have any ticket", "session_id": session_id, "channel": "web"},
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["intent"] == "complaint"
            assert "order id" not in b2["message"].lower()

            # 3) Complaint after that should stay complaint flow (not stuck in order-id loop)
            r3 = await client.post(
                "/api/chat",
                json={
                    "message": "i am disappointed with your services",
                    "session_id": session_id,
                    "channel": "web",
                },
            )
            assert r3.status_code == 200
            b3 = r3.json()
            assert b3["intent"] == "complaint"
            assert "order id" not in b3["message"].lower()
            assert "what is this about" in b3["message"].lower()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ui_chat_flow_ticket_creation_sequence(db_session):
    """UI complaint flow: complaint -> option 1 -> email -> ticket created."""
    app = create_app()

    mock_llm = MockLLMProvider(response="I can help with that.")
    mock_store = MockVectorStore()

    async def override_get_llm():
        return mock_llm

    async def override_get_faq_store():
        return mock_store

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_llm] = override_get_llm
    app.dependency_overrides[get_faq_store] = override_get_faq_store
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1) Complaint should start complaint intake
            r1 = await client.post(
                "/api/chat",
                json={"message": "I am very disappointed with your service", "channel": "web"},
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["intent"] == "complaint"
            assert b1["metadata"] is not None
            assert b1["metadata"].get("awaiting_issue_category") is True
            session_id = b1["session_id"]

            # 2) Choose issue category
            r2 = await client.post(
                "/api/chat",
                json={"message": "7", "session_id": session_id, "channel": "web"},
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["intent"] == "complaint"
            assert b2["metadata"] is not None
            assert b2["metadata"].get("awaiting_issue_summary") is True

            # 3) Provide issue summary -> offered support options
            r3 = await client.post(
                "/api/chat",
                json={
                    "message": "The support team was dismissive and I still need help.",
                    "session_id": session_id,
                    "channel": "web",
                },
            )
            assert r3.status_code == 200
            b3 = r3.json()
            assert b3["intent"] == "complaint"
            assert b3["metadata"] is not None
            assert b3["metadata"].get("offered_ticket_options") is True

            # 4) Choose to create ticket
            r4 = await client.post(
                "/api/chat",
                json={"message": "1", "session_id": session_id, "channel": "web"},
            )
            assert r4.status_code == 200
            b4 = r4.json()
            assert b4["intent"] == "complaint"
            assert "email" in b4["message"].lower()
            assert b4["metadata"] is not None
            assert b4["metadata"].get("awaiting_email") is True

            # 5) Provide email -> ticket created
            r5 = await client.post(
                "/api/chat",
                json={"message": "user@example.com", "session_id": session_id, "channel": "web"},
            )
            assert r5.status_code == 200
            b5 = r5.json()
            assert b5["intent"] == "complaint"
            assert b5["metadata"] is not None
            assert "ticket_id" in b5["metadata"]
            assert b5["metadata"].get("has_email") is True
    finally:
        app.dependency_overrides.clear()
