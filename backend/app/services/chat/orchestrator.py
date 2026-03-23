"""Chat orchestrator: routes user messages to the appropriate service."""

import logging
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.models.chat import ChatRequest, ChatResponse, Intent, Message, MessageRole
from app.models.complaint import ComplaintRequest
from app.services.chat.intent import classify_intent, classify_intent_llm
from app.services.complaints.ticket_service import TicketService
from app.services.faq.base import VectorStore
from app.services.llm.base import LLMProvider
from app.services.orders.order_service import OrderService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant. You help customers with:
1. Answering frequently asked questions about products, shipping, returns, etc.
2. Tracking their orders by order ID.
3. Handling complaints and creating support tickets when needed.

Be concise, empathetic, and professional. If you don't know something, say so honestly.
When a customer has a complaint, acknowledge their frustration and let them know a ticket has been created.
For order tracking, always ask for the order ID if not provided.
"""


class ChatOrchestrator:
    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStore,
        db: AsyncSession,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.db = db
        self.order_service = OrderService(db)
        self.ticket_service = TicketService(db)

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid.uuid4())

        # Ensure session exists
        await self._ensure_session(session_id, request.channel)

        # Classify intent
        intent = classify_intent(request.message)

        # If ambiguous (GENERAL), try LLM classification
        if intent == Intent.GENERAL:
            intent = await classify_intent_llm(request.message, self.llm)

        # Store user message
        await self._store_message(session_id, MessageRole.USER, request.message, intent)

        # Route to appropriate handler
        response_text, metadata = await self._route(intent, request.message, session_id)

        # Store assistant message
        await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            intent=intent,
            metadata=metadata,
        )

    async def _route(
        self, intent: Intent, message: str, session_id: str
    ) -> tuple[str, dict | None]:
        try:
            if intent == Intent.FAQ:
                return await self._handle_faq(message)
            elif intent == Intent.ORDER_TRACKING:
                return await self._handle_order(message)
            elif intent == Intent.COMPLAINT:
                return await self._handle_complaint(message, session_id)
            elif intent == Intent.GREETING:
                return await self._handle_greeting(message)
            else:
                return await self._handle_general(message)
        except Exception as e:
            logger.error(f"Error handling {intent}: {e}")
            return await self._handle_general(message)

    async def _handle_faq(self, message: str) -> tuple[str, dict | None]:
        results = await self.vector_store.search(message, top_k=3)

        if not results or results[0].score < 0.55:
            # No good FAQ match, use LLM directly
            response = await self._llm_generate(
                message,
                "The user is asking a question. Answer helpfully based on general knowledge. "
                "If you're unsure, suggest they contact support directly.",
            )
            return response, None

        # Build context from FAQ results
        context = "\n\n".join(
            f"Q: {r.entry.question}\nA: {r.entry.answer}" for r in results
        )
        response = await self._llm_generate(
            message,
            f"Answer the user's question based on these FAQ entries:\n\n{context}\n\n"
            "Synthesize a natural response. Don't say 'according to our FAQ'.",
        )
        return response, {"faq_sources": [r.entry.question for r in results]}

    async def _handle_order(self, message: str) -> tuple[str, dict | None]:
        # Try to extract order ID
        order_id_match = re.search(r"ORD-\w+", message, re.IGNORECASE)
        if not order_id_match:
            # Also try generic patterns like #12345 or order 12345
            order_id_match = re.search(r"#?(\d{4,})", message)

        if order_id_match:
            order_id = order_id_match.group(0)
            if not order_id.startswith("ORD-"):
                order_id = f"ORD-{order_id}"
            try:
                summary = await self.order_service.get_status_summary(order_id)
                return summary, {"order_id": order_id}
            except Exception:
                return (
                    f"I couldn't find order {order_id}. Please double-check the order ID "
                    "and try again, or provide your email address for a lookup.",
                    None,
                )

        return (
            "I'd be happy to help you track your order! Could you please provide your order ID? "
            "It usually looks like ORD-XXXX.",
            None,
        )

    async def _handle_complaint(self, message: str, session_id: str) -> tuple[str, dict | None]:
        complaint_req = ComplaintRequest(message=message, session_id=session_id)
        ticket = await self.ticket_service.create_ticket(complaint_req, category="complaint")

        escalation_note = ""
        if ticket.status.value == "escalated":
            escalation_note = (
                " Due to the urgency of your concern, this has been automatically "
                "escalated to a senior support representative."
            )

        response = await self._llm_generate(
            message,
            f"The customer has filed a complaint. A support ticket (#{ticket.id[:8]}) has been created "
            f"with {ticket.priority.value} priority.{escalation_note} "
            "Acknowledge their frustration, confirm the ticket was created, and assure them "
            "someone will follow up. Be empathetic.",
        )
        return response, {
            "ticket_id": ticket.id,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
        }

    async def _handle_greeting(self, message: str) -> tuple[str, dict | None]:
        return (
            "Hello! Welcome to our customer support. I'm here to help you with:\n\n"
            "- **FAQs** - Questions about our products, shipping, returns, etc.\n"
            "- **Order Tracking** - Check the status of your order\n"
            "- **Support** - File a complaint or get help with an issue\n\n"
            "How can I assist you today?",
            None,
        )

    async def _handle_general(self, message: str) -> tuple[str, dict | None]:
        response = await self._llm_generate(message)
        return response, None

    async def _llm_generate(self, user_message: str, extra_context: str = "") -> str:
        system = SYSTEM_PROMPT
        if extra_context:
            system += f"\n\nAdditional context: {extra_context}"

        messages = [Message(role=MessageRole.USER, content=user_message)]
        return await self.llm.generate(messages, system_prompt=system)

    async def _ensure_session(self, session_id: str, channel: str) -> None:
        from sqlalchemy import select

        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        if not result.scalar_one_or_none():
            session = ChatSession(id=session_id, channel=channel)
            self.db.add(session)
            await self.db.flush()

    async def _store_message(
        self, session_id: str, role: MessageRole, content: str, intent: Intent
    ) -> None:
        msg = ChatMessage(
            session_id=session_id,
            role=role.value,
            content=content,
            intent=intent.value,
        )
        self.db.add(msg)
        await self.db.flush()
