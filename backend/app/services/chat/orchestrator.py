"""Chat orchestrator: routes user messages to support handlers.

Quality-focused updates:
- No hard session lock after ticket creation.
- Email-first ticket flow (valid email or explicit 'skip').
- Topic switching out of complaint states.
- Consistent message persistence in state-machine branches.
- Single input validator.
- Legacy metadata-state normalization.
"""

import logging
import json
import re
import uuid
from enum import Enum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.exceptions import OrderNotFoundError
from app.models.chat import ChatRequest, ChatResponse, Intent, Message, MessageRole
from app.models.complaint import ComplaintRequest
from app.services.chat.intent import FrustrationLevel, classify_intent, classify_intent_llm, is_order_lookup_request
from app.services.complaints.sentiment import analyze_sentiment
from app.services.complaints.ticket_service import TicketService
from app.services.faq.base import VectorStore
from app.services.llm.base import LLMProvider
from app.services.orders.order_id_utils import looks_like_explicit_order_reference, normalize_order_id
from app.services.orders.order_service import OrderService

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    NORMAL = "NORMAL_CHAT"
    AWAITING_ORDER = "AWAITING_ORDER_ID"
    AWAITING_ISSUE_CATEGORY = "AWAITING_ISSUE_CATEGORY"
    AWAITING_ISSUE_SUMMARY = "AWAITING_ISSUE_SUMMARY"
    SUPPORT_OPTIONS = "SUPPORT_OPTIONS"
    EMAIL_COLLECTION = "EMAIL_COLLECTION"


CONVERSATION_HISTORY_LIMIT = 12
AUTO_ESCALATE_THRESHOLD = -0.9
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
EMAIL_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

ISSUE_CATEGORY_OPTIONS = {
    "1": "order_tracking",
    "2": "refund",
    "3": "damaged_or_wrong_item",
    "4": "payment_or_charge",
    "5": "faq_or_policy",
    "6": "account",
    "7": "other",
}

ISSUE_CATEGORY_LABELS = {
    "order_tracking": "order, delivery, or tracking",
    "refund": "refund or return",
    "damaged_or_wrong_item": "damaged or wrong item",
    "payment_or_charge": "payment or charge",
    "faq_or_policy": "FAQ or policy",
    "account": "account",
    "other": "something else",
}

ORDER_RELATED_ISSUE_CATEGORIES = {
    "order_tracking",
    "refund",
    "damaged_or_wrong_item",
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(you\s+are|your\s+role|everything)",
    r"you\s+are\s+now\s+(DAN|a\s+different)",
    r"system\s*:\s*",
    r"<\s*script\s*>",
    r"reveal\s+(your\s+)?(prompt|instructions|system)",
]

SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant.

You help customers with:
1. Answering FAQs.
2. Tracking orders by ID.
3. Handling complaints and creating support tickets.

STRICT PROHIBITIONS:
- Never invent business facts (prices, timelines, contacts, order status, tracking).
- If missing verified data, direct user to contact the support team directly.
- Never promise outcomes.
- For order status, only use returned database data.

RESPONSE STYLE:
- Sound natural, warm, and specific instead of scripted.
- Start with the direct answer or action taken, then add next steps.
- Keep simple replies short (1-3 sentences); use more detail only when needed.
- Avoid repeating the same opening phrase every turn.
- If the user is upset, acknowledge it briefly and move to concrete help.
"""

FAQ_GROUNDED_PROMPT = """Answer using ONLY the FAQ entries below.
Do not add information not present in these entries.
If incomplete, say so and direct the customer to contact the support team directly.
Use natural conversational wording (not policy-jargon).

FAQ entries:
{context}"""

FAQ_NO_MATCH_PROMPT = """No FAQ entry matches this question.
Do not guess.
Tell the customer you don't have that specific info and ask them to contact the support team directly.
Keep it brief and empathetic."""

OPERATIONS_COMPOSER_PROMPT = """You are refining a customer-support reply that is already factually correct.

Your job:
1. Keep every factual detail exactly grounded in the provided facts/draft.
2. Improve clarity, calm tone, and natural wording.
3. Keep it concise and actionable.

Hard rules:
- Do not add new facts, promises, contacts, dates, IDs, or statuses.
- Preserve ticket IDs, order IDs, emails, options, and instructions exactly.
- If the draft asks for specific user action (for example: reply with 1 or 2, share email, type skip), keep that action.
- Return plain text only.

Facts:
{facts_json}

Draft reply:
{draft}
"""


class ChatOrchestrator:
    def __init__(self, llm: LLMProvider, vector_store: VectorStore, db: AsyncSession):
        self.llm = llm
        self.vector_store = vector_store
        self.db = db
        self.order_service = OrderService(db)
        self.ticket_service = TicketService(db)

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid.uuid4())

        validation_error = self._validate_input(request.message)
        if validation_error:
            return ChatResponse(
                message=validation_error,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"validation_error": True},
            )

        await self._ensure_session(session_id, request.channel)
        metadata = await self._get_session_metadata(session_id)
        current_state = self._resolve_state(metadata)

        if metadata.get("state") != current_state:
            await self._update_session_metadata(session_id, {"state": current_state})
            metadata["state"] = current_state

        detected_email = self._extract_email_candidate(request.message)
        if detected_email:
            metadata["last_provided_email"] = detected_email
            await self._update_session_metadata(
                session_id,
                {"last_provided_email": detected_email},
            )

        if current_state == ConversationState.EMAIL_COLLECTION.value or metadata.get("awaiting_email"):
            return await self._handle_email_collection(request.message, session_id, metadata)

        if current_state == ConversationState.SUPPORT_OPTIONS.value or metadata.get("offered_ticket_options"):
            return await self._handle_ticket_confirmation(request.message, session_id, metadata)

        if (
            current_state == ConversationState.AWAITING_ISSUE_CATEGORY.value
            or metadata.get("awaiting_issue_category")
        ):
            return await self._handle_issue_category_collection(request.message, session_id, metadata)

        if (
            current_state == ConversationState.AWAITING_ISSUE_SUMMARY.value
            or metadata.get("awaiting_issue_summary")
        ):
            return await self._handle_issue_summary_collection(request.message, session_id, metadata)

        if current_state == ConversationState.AWAITING_ORDER.value or metadata.get("awaiting_order_id"):
            if self._is_flow_cancel_request(request.message):
                await self._store_message(session_id, MessageRole.USER, request.message, Intent.GENERAL, None)
                cleared_updates = {
                    **self._clear_complaint_pending_updates(),
                    "active_order_id": None if not metadata.get("active_order_confirmed") else metadata.get("active_order_id"),
                    "active_order_confirmed": bool(metadata.get("active_order_confirmed")),
                }
                await self._update_session_metadata(
                    session_id,
                    cleared_updates,
                )
                metadata.update(cleared_updates)
                reply = "Understood. I canceled that lookup for now. If you want to resume, just share the order ID later."
                await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.GENERAL, None)
                return ChatResponse(
                    message=reply,
                    session_id=session_id,
                    intent=Intent.GENERAL,
                    metadata={"ticket_flow_cancelled": True},
                )

            if self._looks_like_order_followup(request.message):
                response_intent = (
                    Intent.COMPLAINT if metadata.get("complaint_ticket_requested") or metadata.get("issue_category") else Intent.ORDER_TRACKING
                )
                await self._store_message(session_id, MessageRole.USER, request.message, response_intent, None)
                if metadata.get("complaint_ticket_requested") or metadata.get("issue_category"):
                    response_text, meta_out = await self._handle_complaint_order_id_collection(
                        request.message,
                        session_id,
                        metadata,
                    )
                else:
                    response_text, meta_out = await self._handle_order(request.message, session_id, metadata)
                await self._store_message(session_id, MessageRole.ASSISTANT, response_text, response_intent, None)
                return ChatResponse(
                    message=response_text,
                    session_id=session_id,
                    intent=response_intent,
                    metadata=meta_out,
                )

            # User switched topic while we were waiting for an order ID.
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(
                session_id,
                cleared_updates,
            )
            metadata.update(cleared_updates)

        if self._is_ticket_creation_request(request.message):
            existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
            if existing_ticket:
                await self._store_message(session_id, MessageRole.USER, request.message, Intent.COMPLAINT, None)
                await self._sync_existing_ticket_metadata(session_id, existing_ticket, metadata)
                draft_reply, meta_out = self._build_existing_ticket_reply(existing_ticket, metadata)
                response_text = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=request.message,
                    draft=draft_reply,
                    facts=meta_out,
                    required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
                )
                await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
                return ChatResponse(
                    message=response_text,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata=meta_out,
                )
            await self._store_message(session_id, MessageRole.USER, request.message, Intent.COMPLAINT, None)
            response_text, meta_out = await self._start_complaint_intake(
                request.message,
                session_id,
                metadata,
                ticket_requested=True,
                source_intent=Intent.COMPLAINT,
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        if self._is_ticket_inquiry(request.message):
            intent = Intent.COMPLAINT
            sentiment_score = analyze_sentiment(request.message).score
            await self._store_message(session_id, MessageRole.USER, request.message, intent, sentiment_score)
            response_text, meta_out = await self._handle_ticket_inquiry(session_id, request.message)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=intent,
                metadata=meta_out,
            )

        intent, frustration, sarcasm = classify_intent(request.message)
        if intent == Intent.GENERAL and self._should_use_llm_intent_disambiguation(request.message):
            intent, frustration, sarcasm = await classify_intent_llm(request.message, self.llm)

        sentiment_score = analyze_sentiment(request.message).score

        await self._apply_active_order_context_policy(
            session_id,
            metadata,
            message=request.message,
            intent=intent,
        )

        if intent == Intent.COMPLAINT:
            genuinely_negative = (
                sentiment_score <= 0.0
                or frustration in (FrustrationLevel.MODERATE, FrustrationLevel.HIGH)
                or sarcasm
            )
            if not genuinely_negative:
                intent = Intent.GENERAL

        await self._store_message(session_id, MessageRole.USER, request.message, intent, sentiment_score)

        if intent == Intent.COMPLAINT:
            response_text, meta_out = await self._handle_frustration(
                request.message,
                session_id,
                frustration,
                sentiment_score,
            )
        else:
            response_text, meta_out = await self._route(intent, request.message, session_id, metadata)

        await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent, None)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            intent=intent,
            metadata=meta_out,
        )

    def _resolve_state(self, metadata: dict) -> str:
        state = metadata.get("state", ConversationState.NORMAL.value)
        legacy_map = {
            "NORMAL_CHAT": ConversationState.NORMAL.value,
            "AWAITING_ORDER_ID": ConversationState.AWAITING_ORDER.value,
            "AWAITING_ISSUE_CATEGORY": ConversationState.AWAITING_ISSUE_CATEGORY.value,
            "AWAITING_ISSUE_SUMMARY": ConversationState.AWAITING_ISSUE_SUMMARY.value,
            "SUPPORT_OPTIONS": ConversationState.SUPPORT_OPTIONS.value,
            "EMAIL_COLLECTION": ConversationState.EMAIL_COLLECTION.value,
            "TICKET_CREATION": ConversationState.EMAIL_COLLECTION.value,
            "SESSION_LOCKED": ConversationState.NORMAL.value,
            "normal": ConversationState.NORMAL.value,
            "awaiting_order": ConversationState.AWAITING_ORDER.value,
            "awaiting_issue_category": ConversationState.AWAITING_ISSUE_CATEGORY.value,
            "awaiting_issue_summary": ConversationState.AWAITING_ISSUE_SUMMARY.value,
            "support_options": ConversationState.SUPPORT_OPTIONS.value,
            "email_collection": ConversationState.EMAIL_COLLECTION.value,
        }
        return legacy_map.get(state, state)

    def _build_support_context(self, metadata: dict | None) -> dict:
        metadata = metadata or {}
        return {
            "current_state": self._resolve_state(metadata),
            "active_order_id": metadata.get("active_order_id"),
            "active_order_confirmed": bool(metadata.get("active_order_confirmed")),
            "active_order_context_idle_turns": int(metadata.get("active_order_context_idle_turns", 0)),
            "has_open_ticket": bool(metadata.get("has_open_ticket")),
            "ticket_id": metadata.get("ticket_id"),
            "customer_email": metadata.get("customer_email"),
            "awaiting_email": bool(metadata.get("awaiting_email")),
            "awaiting_order_id": bool(metadata.get("awaiting_order_id")),
            "issue_category": metadata.get("issue_category"),
            "issue_summary": metadata.get("issue_summary"),
            "issue_order_related": metadata.get("issue_order_related"),
            "ticket_context_collected": metadata.get("ticket_context_collected"),
        }

    def _complaint_state_updates(
        self,
        *,
        state: ConversationState = ConversationState.NORMAL,
        awaiting_issue_category: bool = False,
        awaiting_issue_summary: bool = False,
        awaiting_order_id: bool = False,
        complaint_ticket_requested: bool | None = None,
    ) -> dict:
        updates = {
            "state": state.value,
            "awaiting_issue_category": awaiting_issue_category,
            "awaiting_issue_summary": awaiting_issue_summary,
            "awaiting_order_id": awaiting_order_id,
        }
        if complaint_ticket_requested is not None:
            updates["complaint_ticket_requested"] = complaint_ticket_requested
        return updates

    def _clear_complaint_pending_updates(self) -> dict:
        return {
            "state": ConversationState.NORMAL.value,
            "awaiting_issue_category": False,
            "awaiting_issue_summary": False,
            "awaiting_order_id": False,
            "offered_ticket_options": False,
            "awaiting_email": False,
            "awaiting_email_confirmation": False,
            "pending_email": None,
            "email_attempts": 0,
            "complaint_ticket_requested": False,
            "issue_category": None,
            "issue_summary": None,
            "issue_order_related": False,
            "ticket_context_collected": False,
            "original_complaint_message": None,
            "original_sentiment_score": None,
            "order_lookup_failures": 0,
            "active_order_context_idle_turns": 0,
        }

    def _is_flow_cancel_request(self, message: str) -> bool:
        text = message.lower().strip()
        cancel_patterns = (
            r"\bnever\s*mind\b",
            r"\bforget\s+it\b",
            r"\bforget\s+support\b",
            r"\bleave\s+it\b",
            r"\bleave\s+the\s+complaint\b",
            r"\bleave\s+support\b",
            r"\bdrop\s+it\b",
            r"\bdrop\s+the\s+complaint\b",
            r"\bcancel\s+(this|it|that|ticket|request|flow)\b",
            r"\bi\s+do\s+not\s+want\s+to\s+continue\s+(this|it)\b",
            r"\bi\s+don't\s+want\s+to\s+continue\s+(this|it)\b",
            r"\bstop\s+this\b",
            r"\bstop\b",
            r"\bnot\s+now\b",
            r"\bno\s+thanks\b",
        )
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in cancel_patterns)

    def _normalize_issue_category(self, message: str) -> str | None:
        text = message.lower().strip()
        if not text:
            return None

        for option, category in ISSUE_CATEGORY_OPTIONS.items():
            if text == option or text == f"option {option}":
                return category

        keyword_map = {
            "order_tracking": (
                "delivery",
                "tracking",
                "shipment",
                "package",
                "shipping status",
                "order status",
                "where is my order",
                "last order",
                "recent order",
            ),
            "refund": (
                "refund",
                "return",
                "money back",
                "cancel order",
            ),
            "damaged_or_wrong_item": (
                "damaged",
                "broken",
                "wrong item",
                "incorrect item",
                "defective",
                "missing item",
            ),
            "payment_or_charge": (
                "payment",
                "charged",
                "charge",
                "billing",
                "card",
                "transaction",
            ),
            "faq_or_policy": (
                "policy",
                "faq",
                "warranty",
                "terms",
                "return policy",
            ),
            "account": (
                "account",
                "login",
                "password",
                "sign in",
                "signin",
                "profile",
            ),
            "other": ("other", "something else"),
        }
        for category, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                return category
        return None

    def _is_order_related_issue(self, category: str | None, message_hint: str = "") -> bool:
        if not category:
            return False
        if category in ORDER_RELATED_ISSUE_CATEGORIES:
            return True
        if category != "payment_or_charge":
            return False

        lowered = message_hint.lower()
        return any(term in lowered for term in ("order", "delivery", "shipment", "item", "refund"))

    def _build_issue_category_prompt(self) -> str:
        return (
            "I can help with that. Before I create or escalate anything, what is this about?\n\n"
            "1. Order, delivery, or tracking\n"
            "2. Refund or return\n"
            "3. Damaged or wrong item\n"
            "4. Payment or charge\n"
            "5. FAQ or policy\n"
            "6. Account\n"
            "7. Something else\n\n"
            "Reply with the number or short category name."
        )

    def _build_issue_summary_prompt(self, category: str | None) -> str:
        label = ISSUE_CATEGORY_LABELS.get(category or "", "issue")
        return (
            f"Thanks. Please describe the {label} issue in 1-2 short sentences "
            "so I can capture the right support details."
        )

    def _build_order_id_prompt(self, category: str | None) -> str:
        label = ISSUE_CATEGORY_LABELS.get(category or "", "issue")
        return (
            f"Please share the order ID for this {label} issue so I can link the complaint correctly. "
            "Example: ORD-1001."
        )

    def _build_order_not_found_reply(self, normalized_order_id: str, attempt_count: int = 1) -> str:
        if attempt_count <= 1:
            return (
                f"I could not find order {normalized_order_id}. "
                "Please check the order ID and try again. Example format: ORD-1001."
            )
        if attempt_count == 2:
            return (
                f"I still could not find order {normalized_order_id}. "
                "Please recheck the order ID from your confirmation email or SMS and try again. "
                "Example format: ORD-1001."
            )
        return (
            f"I still could not find order {normalized_order_id}. "
            "Please recheck the order ID from your confirmation email or SMS. "
            "If you want, I can also help with a return, refund, or general support question without the order lookup."
        )

    def _build_ticket_description(self, metadata: dict, fallback_message: str) -> str:
        issue_category = metadata.get("issue_category") or "complaint"
        issue_summary = metadata.get("issue_summary") or fallback_message
        order_id = metadata.get("active_order_id") if metadata.get("active_order_confirmed") else None
        original_message = metadata.get("original_complaint_message") or fallback_message

        lines = [
            f"Category: {issue_category}",
            f"Summary: {issue_summary}",
        ]
        if order_id:
            lines.append(f"Order ID: {order_id}")
        if original_message and original_message != issue_summary:
            lines.append(f"Original message: {original_message}")
        return "\n".join(lines)

    def _extract_issue_summary_candidate(self, message: str, category: str | None = None) -> str | None:
        text = message.strip()
        if not text:
            return None

        lowered = text.lower()
        if self._is_ticket_creation_request(message):
            return None
        if looks_like_explicit_order_reference(message):
            return None
        if category and self._normalize_issue_category(message) == category and len(lowered.split()) <= 4:
            return None
        if len(re.findall(r"[A-Za-z0-9]+", text)) < 4:
            return None
        return text

    def _validate_input(self, message: str) -> str | None:
        stripped = message.strip()
        if not stripped:
            return "I didn't receive a message. Could you please type your question?"

        if sum(char.isalnum() for char in stripped) == 0:
            return "I'm having trouble understanding that. Could you rephrase using words?"

        msg_lower = message.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                logger.warning("Prompt injection attempt: %s", message[:100])
                return "I'm here to help with order tracking, FAQs, and support. How can I assist you today?"

        return None

    def _looks_like_order_followup(self, message: str) -> bool:
        """Return True when a message likely continues order-tracking flow."""
        message_lower = message.lower().strip()
        if looks_like_explicit_order_reference(message):
            return True

        # If the user sounds angry/sarcastic while we're awaiting an ID,
        # let normal intent routing handle it instead of forcing order flow.
        intent_hint, frustration_hint, sarcasm_hint = classify_intent(message)
        if intent_hint == Intent.COMPLAINT and (
            frustration_hint in (FrustrationLevel.MODERATE, FrustrationLevel.HIGH)
            or sarcasm_hint
        ):
            return False

        threat_or_complaint_markers = (
            "legal action",
            "lawsuit",
            "lawyer",
            "sue",
            "court",
            "never again",
            "not placing",
            "not on time",
            "late again",
            "lost my order",
            "lost package",
        )
        if any(marker in message_lower for marker in threat_or_complaint_markers):
            return False

        return is_order_lookup_request(message)

    def _is_ticket_inquiry(self, message: str) -> bool:
        """Detect whether user asks to check an existing support ticket."""
        message_lower = message.lower().strip()
        if not any(word in message_lower for word in ("ticket", "case", "complaint")):
            return False
        creation_signals = (
            "create ticket",
            "open ticket",
            "raise ticket",
            "make ticket",
            "new ticket",
            "file complaint",
        )
        if any(signal in message_lower for signal in creation_signals):
            return False
        inquiry_signals = (
            "do i have",
            "have any",
            "check",
            "status",
            "update",
            "progress",
            "existing",
            "open",
            "my ticket",
            "ticket id",
        )
        return any(signal in message_lower for signal in inquiry_signals)

    def _is_ticket_creation_request(self, message: str) -> bool:
        """Detect explicit requests to create another/new ticket."""
        message_lower = message.lower().strip()
        if "ticket" not in message_lower and "complaint" not in message_lower and "case" not in message_lower:
            return False
        return bool(
            re.search(
                r"\b(create|creat|open|raise|make|new|another)\b.*\b(ticket|complaint|case)\b",
                message_lower,
            )
        )

    def _is_ticket_flow_exit_request(self, message: str) -> bool:
        """Detect explicit requests to stop/cancel ticket flow."""
        exit_phrases = (
            "dont want to complain",
            "don't want to complain",
            "do not want to complain",
            "don't want ticket",
            "dont want ticket",
            "do not want ticket",
            "cancel ticket",
            "stop ticket",
            "no complaint",
            "without ticket",
            "just answer my question",
            "just answer the question",
            "forget support",
            "leave the complaint",
            "leave support",
            "i don't want to continue this",
            "i dont want to continue this",
            "i do not want to continue this",
        )
        return any(phrase in message for phrase in exit_phrases)

    def _is_email_refusal(self, message: str) -> bool:
        """Detect natural-language refusal to share an email."""
        refusal_phrases = (
            "do not want to share my email",
            "don't want to share my email",
            "dont want to share my email",
            "do not want to give my email",
            "don't want to give my email",
            "dont want to give my email",
            "prefer not to share my email",
            "i do not want to share email",
            "i don't want to share email",
            "without sharing email",
            "without email",
            "no email",
        )
        return any(phrase in message for phrase in refusal_phrases)

    def _should_use_llm_intent_disambiguation(self, message: str) -> bool:
        """Use LLM intent classification only when helpful, not for vague follow-ups."""
        text = message.lower().strip()
        if not text:
            return False

        conversational_general_phrases = (
            "answer my question",
            "just answer my question",
            "forget about it",
            "leave it",
            "never mind",
            "no thanks",
            "thanks",
        )
        if any(phrase in text for phrase in conversational_general_phrases):
            return False

        domain_terms = (
            "order",
            "track",
            "tracking",
            "delivery",
            "package",
            "ticket",
            "complaint",
            "refund",
            "return",
            "shipping",
            "policy",
            "faq",
            "support",
        )
        short_vague = len(text.split()) <= 4 and not any(term in text for term in domain_terms)
        if short_vague:
            return False

        return True

    def _extract_email_candidate(self, text: str) -> str | None:
        """Extract the first valid-looking email from free-form text."""
        match = EMAIL_CANDIDATE_RE.search(text)
        if not match:
            return None
        candidate = match.group(0).strip()
        if EMAIL_RE.match(candidate):
            return candidate
        return None

    def _build_existing_ticket_reply(self, existing_ticket, metadata: dict) -> tuple[str, dict]:
        ticket_id_short = existing_ticket.id[:8]
        ticket_email = existing_ticket.customer_email or metadata.get("customer_email")
        if ticket_email:
            reply = (
                f'We already created support ticket #{ticket_id_short} for this conversation. '
                f'You can keep sharing any additional issue details here, and our team will review them under the same ticket. '
                f'We will follow up using "{ticket_email}".'
            )
        else:
            reply = (
                f"We already created support ticket #{ticket_id_short} for this conversation. "
                "You can keep sharing any additional issue details here, and our team will review them under the same ticket."
            )
        meta = {
            "existing_ticket_id": existing_ticket.id,
            "ticket_id": existing_ticket.id,
            "ticket_status": existing_ticket.status,
            "ticket_found": True,
            "has_email": bool(ticket_email),
            "has_open_ticket": True,
            "additional_issue_supported": True,
        }
        if ticket_email:
            meta["customer_email"] = ticket_email
        return reply, meta

    def _existing_ticket_session_updates(self, existing_ticket, metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        ticket_email = existing_ticket.customer_email or metadata.get("customer_email")
        updates = {
            "has_open_ticket": True,
            "ticket_id": existing_ticket.id,
            "state": ConversationState.NORMAL.value,
            "offered_ticket_options": False,
            "awaiting_email": False,
            "awaiting_email_confirmation": False,
            "pending_email": None,
            "email_attempts": 0,
        }
        if ticket_email:
            updates["customer_email"] = ticket_email
        return updates

    async def _sync_existing_ticket_metadata(self, session_id: str, existing_ticket, metadata: dict | None = None) -> dict:
        updates = self._existing_ticket_session_updates(existing_ticket, metadata)
        await self._update_session_metadata(session_id, updates)
        if metadata is not None:
            metadata.update(updates)
        return updates

    async def _apply_active_order_context_policy(
        self,
        session_id: str,
        metadata: dict,
        *,
        message: str,
        intent: Intent,
    ) -> None:
        if not metadata.get("active_order_confirmed"):
            return

        message_lower = message.lower()
        order_words = {"order", "delivery", "tracking", "package", "shipment", "refund"}
        if intent in (Intent.ORDER_TRACKING, Intent.COMPLAINT) or any(word in message_lower for word in order_words):
            if metadata.get("active_order_context_idle_turns"):
                updates = {"active_order_context_idle_turns": 0}
                await self._update_session_metadata(session_id, updates)
                metadata.update(updates)
            return

        explicit_topic_switch = intent == Intent.FAQ and message.strip().endswith("?")
        idle_turns = int(metadata.get("active_order_context_idle_turns", 0)) + 1
        if explicit_topic_switch or idle_turns >= 2:
            updates = {
                "active_order_id": None,
                "active_order_confirmed": False,
                "active_order_context_idle_turns": 0,
                "awaiting_order_id": False,
                "order_lookup_failures": 0,
                "state": ConversationState.NORMAL.value,
            }
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            return

        updates = {"active_order_context_idle_turns": idle_turns}
        await self._update_session_metadata(session_id, updates)
        metadata.update(updates)

    async def _start_complaint_intake(
        self,
        message: str,
        session_id: str,
        metadata: dict,
        *,
        ticket_requested: bool,
        source_intent: Intent,
    ) -> tuple[str, dict]:
        issue_category = metadata.get("issue_category") or self._normalize_issue_category(message)
        issue_order_related = self._is_order_related_issue(issue_category, message)
        issue_summary = metadata.get("issue_summary")
        if not issue_summary and issue_category:
            issue_summary = self._extract_issue_summary_candidate(message, issue_category)

        updates = {
            "original_complaint_message": metadata.get("original_complaint_message") or message,
            "complaint_ticket_requested": bool(ticket_requested or metadata.get("complaint_ticket_requested")),
            "issue_category": issue_category,
            "issue_order_related": issue_order_related,
            "issue_summary": issue_summary,
            "ticket_context_collected": bool(
                issue_category
                and issue_summary
                and (not issue_order_related or metadata.get("active_order_confirmed"))
            ),
        }
        await self._update_session_metadata(session_id, updates)
        metadata.update(updates)
        return await self._advance_complaint_intake(session_id, metadata, message, source_intent=source_intent)

    async def _advance_complaint_intake(
        self,
        session_id: str,
        metadata: dict,
        user_message: str,
        *,
        source_intent: Intent,
    ) -> tuple[str, dict]:
        issue_category = metadata.get("issue_category")
        issue_order_related = self._is_order_related_issue(issue_category, metadata.get("original_complaint_message", ""))
        metadata["issue_order_related"] = issue_order_related

        if not issue_category:
            updates = {
                **self._complaint_state_updates(
                    state=ConversationState.AWAITING_ISSUE_CATEGORY,
                    awaiting_issue_category=True,
                    complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
                ),
                "issue_order_related": False,
                "order_lookup_failures": 0,
            }
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = self._build_issue_category_prompt()
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=user_message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=["1", "2", "3", "4", "5", "6", "7"],
            )
            return reply, {"awaiting_issue_category": True, "state": ConversationState.AWAITING_ISSUE_CATEGORY.value}

        if issue_order_related and not metadata.get("active_order_confirmed"):
            updates = self._complaint_state_updates(
                state=ConversationState.AWAITING_ORDER,
                awaiting_order_id=True,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            )
            updates["order_lookup_failures"] = 0
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = self._build_order_id_prompt(issue_category)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=user_message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=["order id", "ord-1001"],
            )
            return reply, {
                "awaiting_order_id": True,
                "issue_category": issue_category,
                "issue_order_related": True,
            }

        if not metadata.get("issue_summary"):
            updates = self._complaint_state_updates(
                state=ConversationState.AWAITING_ISSUE_SUMMARY,
                awaiting_issue_summary=True,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            )
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = self._build_issue_summary_prompt(issue_category)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=user_message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=["1-2", "issue"],
            )
            return reply, {
                "awaiting_issue_summary": True,
                "issue_category": issue_category,
                "issue_order_related": issue_order_related,
            }

        updates = {
            "ticket_context_collected": True,
            **self._complaint_state_updates(
                state=ConversationState.NORMAL,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            ),
        }
        await self._update_session_metadata(session_id, updates)
        metadata.update(updates)

        if metadata.get("complaint_ticket_requested"):
            return await self._begin_email_collection_for_ticket(
                user_message=user_message,
                session_id=session_id,
                metadata=metadata,
            )

        await self._update_session_metadata(
            session_id,
            {
                "offered_ticket_options": True,
                "state": ConversationState.SUPPORT_OPTIONS.value,
            },
        )
        metadata["offered_ticket_options"] = True
        metadata["state"] = ConversationState.SUPPORT_OPTIONS.value
        draft = (
            "Thanks, I have the basic issue details. Would you like me to:\n\n"
            "1. Create a support ticket\n"
            "2. Try to resolve this here\n\n"
            "Please reply with 1 or 2."
        )
        reply = await self._compose_operational_reply(
            session_id=session_id,
            user_message=user_message,
            draft=draft,
            facts=self._build_support_context(metadata),
            required_terms=["1", "2", "ticket"],
        )
        return reply, {
            "offered_ticket_options": True,
            "issue_category": issue_category,
            "issue_summary": metadata.get("issue_summary"),
            "issue_order_related": issue_order_related,
        }

    async def _handle_issue_category_collection(self, message: str, session_id: str, metadata: dict) -> ChatResponse:
        if self._is_flow_cancel_request(message):
            await self._store_message(session_id, MessageRole.USER, message, Intent.GENERAL, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(session_id, cleared_updates)
            metadata.update(cleared_updates)
            reply = "Understood. I canceled the complaint flow for now. If you want to resume later, just say create ticket."
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.GENERAL, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"ticket_flow_cancelled": True},
            )

        category = self._normalize_issue_category(message)
        if category:
            summary_candidate = self._extract_issue_summary_candidate(message, category)
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            updates = {
                "issue_category": category,
                "issue_order_related": self._is_order_related_issue(
                    category,
                    metadata.get("original_complaint_message", message),
                ),
            }
            if summary_candidate:
                updates["issue_summary"] = summary_candidate
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            response_text, meta_out = await self._advance_complaint_intake(
                session_id,
                metadata,
                message,
                source_intent=Intent.COMPLAINT,
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(message=response_text, session_id=session_id, intent=Intent.COMPLAINT, metadata=meta_out)

        intent_check, frustration_hint, sarcasm_hint = classify_intent(message)
        clear_switch = (
            intent_check == Intent.GREETING
            or (
                intent_check in (Intent.FAQ, Intent.ORDER_TRACKING)
                and message.strip().endswith("?")
                and frustration_hint == FrustrationLevel.NONE
                and not sarcasm_hint
            )
        )
        if clear_switch:
            await self._store_message(session_id, MessageRole.USER, message, intent_check, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(session_id, cleared_updates)
            metadata.update(cleared_updates)
            response_text, meta_out = await self._route(intent_check, message, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent_check, None)
            route_meta = dict(meta_out or {})
            route_meta["ticket_flow_cancelled"] = True
            return ChatResponse(message=response_text, session_id=session_id, intent=intent_check, metadata=route_meta)

        await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
        draft = self._build_issue_category_prompt()
        reply = await self._compose_operational_reply(
            session_id=session_id,
            user_message=message,
            draft=draft,
            facts=self._build_support_context(metadata),
            required_terms=["1", "2", "3", "4", "5", "6", "7"],
        )
        await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
        return ChatResponse(
            message=reply,
            session_id=session_id,
            intent=Intent.COMPLAINT,
            metadata={"awaiting_issue_category": True},
        )

    async def _handle_issue_summary_collection(self, message: str, session_id: str, metadata: dict) -> ChatResponse:
        if self._is_flow_cancel_request(message):
            await self._store_message(session_id, MessageRole.USER, message, Intent.GENERAL, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(session_id, cleared_updates)
            metadata.update(cleared_updates)
            reply = "Understood. I canceled the complaint flow for now. If you want to resume later, just say create ticket."
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.GENERAL, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"ticket_flow_cancelled": True},
            )

        summary_candidate = self._extract_issue_summary_candidate(message, metadata.get("issue_category"))
        intent_check, frustration_hint, sarcasm_hint = classify_intent(message)
        clear_switch = (
            intent_check == Intent.GREETING
            or (
                intent_check in (Intent.FAQ, Intent.ORDER_TRACKING)
                and message.strip().endswith("?")
                and frustration_hint == FrustrationLevel.NONE
                and not sarcasm_hint
            )
        )
        if clear_switch:
            await self._store_message(session_id, MessageRole.USER, message, intent_check, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(session_id, cleared_updates)
            metadata.update(cleared_updates)
            response_text, meta_out = await self._route(intent_check, message, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent_check, None)
            route_meta = dict(meta_out or {})
            route_meta["ticket_flow_cancelled"] = True
            return ChatResponse(message=response_text, session_id=session_id, intent=intent_check, metadata=route_meta)

        if not summary_candidate:
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            draft = self._build_issue_summary_prompt(metadata.get("issue_category"))
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=["issue", "1-2"],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata={"awaiting_issue_summary": True},
            )

        summary = summary_candidate
        await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
        updates = {
            "issue_summary": summary,
            "ticket_context_collected": bool(
                metadata.get("issue_category")
                and (not metadata.get("issue_order_related") or metadata.get("active_order_confirmed"))
            ),
        }
        await self._update_session_metadata(session_id, updates)
        metadata.update(updates)
        response_text, meta_out = await self._advance_complaint_intake(
            session_id,
            metadata,
            message,
            source_intent=Intent.COMPLAINT,
        )
        await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
        return ChatResponse(message=response_text, session_id=session_id, intent=Intent.COMPLAINT, metadata=meta_out)

    async def _handle_complaint_order_id_collection(
        self,
        message: str,
        session_id: str,
        metadata: dict,
    ) -> tuple[str, dict]:
        normalized_order_id = normalize_order_id(message)
        if not normalized_order_id:
            updates = self._complaint_state_updates(
                state=ConversationState.AWAITING_ORDER,
                awaiting_order_id=True,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            )
            updates["order_lookup_failures"] = 0
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = self._build_order_id_prompt(metadata.get("issue_category"))
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=["order id", "ord-1001"],
            )
            return reply, {"awaiting_order_id": True, "issue_category": metadata.get("issue_category")}

        try:
            await self.order_service.get_by_id(normalized_order_id)
        except OrderNotFoundError:
            next_attempt_count = int(metadata.get("order_lookup_failures", 0)) + 1
            updates = {
                "active_order_id": None,
                "active_order_confirmed": False,
                "active_order_context_idle_turns": 0,
                "order_lookup_failures": next_attempt_count,
                **self._complaint_state_updates(
                    state=ConversationState.AWAITING_ORDER,
                    awaiting_order_id=True,
                    complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
                ),
            }
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = self._build_order_not_found_reply(normalized_order_id, attempt_count=next_attempt_count)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=[normalized_order_id, "not find", "ord-1001"],
            )
            return reply, {
                "order_id": normalized_order_id,
                "found": False,
                "awaiting_order_id": True,
                "active_order_confirmed": False,
            }
        except Exception as exc:
            logger.error("Complaint order lookup failed (%s): %s", normalized_order_id, exc, exc_info=True)
            return (
                "I'm sorry, I couldn't verify that order right now. Please try again in a moment.",
                {"order_id": normalized_order_id, "lookup_error": True},
            )

        updates = {
            "active_order_id": normalized_order_id,
            "active_order_confirmed": True,
            "active_order_context_idle_turns": 0,
            "order_lookup_failures": 0,
            **self._complaint_state_updates(
                state=ConversationState.NORMAL,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            ),
        }
        await self._update_session_metadata(session_id, updates)
        metadata.update(updates)

        if not metadata.get("issue_summary"):
            updates = self._complaint_state_updates(
                state=ConversationState.AWAITING_ISSUE_SUMMARY,
                awaiting_issue_summary=True,
                complaint_ticket_requested=bool(metadata.get("complaint_ticket_requested")),
            )
            await self._update_session_metadata(session_id, updates)
            metadata.update(updates)
            draft = (
                f"I found order {normalized_order_id}. "
                f"{self._build_issue_summary_prompt(metadata.get('issue_category'))}"
            )
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=self._build_support_context(metadata),
                required_terms=[normalized_order_id, "1-2", "issue"],
            )
            return reply, {
                "order_id": normalized_order_id,
                "found": True,
                "active_order_confirmed": True,
                "awaiting_issue_summary": True,
            }

        return await self._advance_complaint_intake(
            session_id,
            metadata,
            message,
            source_intent=Intent.COMPLAINT,
        )

    async def _begin_email_collection_for_ticket(
        self,
        *,
        user_message: str,
        session_id: str,
        metadata: dict,
    ) -> tuple[str, dict]:
        if metadata.get("customer_email"):
            return await self._handle_complaint(
                metadata.get("original_complaint_message", user_message),
                session_id,
                metadata,
            )

        last_email = metadata.get("last_provided_email")
        if isinstance(last_email, str) and EMAIL_RE.match(last_email):
            await self._update_session_metadata(
                session_id,
                {
                    "offered_ticket_options": False,
                    "awaiting_email": True,
                    "awaiting_email_confirmation": True,
                    "pending_email": last_email,
                    "email_attempts": 0,
                    "state": ConversationState.EMAIL_COLLECTION.value,
                },
            )
            metadata.update(
                {
                    "offered_ticket_options": False,
                    "awaiting_email": True,
                    "awaiting_email_confirmation": True,
                    "pending_email": last_email,
                    "email_attempts": 0,
                    "state": ConversationState.EMAIL_COLLECTION.value,
                }
            )
            reply = (
                f'I found this email from earlier in this chat: "{last_email}". '
                'Is this correct? Reply "yes" to confirm or "change" to provide another email.'
            )
            return reply, {
                "awaiting_email": True,
                "awaiting_email_confirmation": True,
                "extracted_email": last_email,
            }

        await self._update_session_metadata(
            session_id,
            {
                "offered_ticket_options": False,
                "awaiting_email": True,
                "awaiting_email_confirmation": False,
                "pending_email": None,
                "email_attempts": 0,
                "state": ConversationState.EMAIL_COLLECTION.value,
            },
        )
        metadata.update(
            {
                "offered_ticket_options": False,
                "awaiting_email": True,
                "awaiting_email_confirmation": False,
                "pending_email": None,
                "email_attempts": 0,
                "state": ConversationState.EMAIL_COLLECTION.value,
            }
        )
        reply = (
            "I have the main issue details now. Could you share your email so our team can follow up with you? "
            "If you'd rather not, just type 'skip'."
        )
        return reply, {"awaiting_email": True, "awaiting_email_confirmation": False}

    async def _handle_ticket_inquiry(self, session_id: str, user_message: str) -> tuple[str, dict | None]:
        """Handle explicit ticket status/inquiry requests."""
        existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
        if existing_ticket:
            session_metadata = await self._get_session_metadata(session_id)
            await self._sync_existing_ticket_metadata(session_id, existing_ticket, session_metadata)
            draft, meta = self._build_existing_ticket_reply(existing_ticket, session_metadata)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=user_message,
                draft=draft,
                facts=meta,
                required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
            )
            return reply, meta

        await self._update_session_metadata(
            session_id,
            {
                "offered_ticket_options": True,
                "original_complaint_message": user_message,
                "state": ConversationState.SUPPORT_OPTIONS.value,
            },
        )
        draft = (
            "I couldn't find an open ticket for this chat yet. "
            "Would you like me to create one now?\n\n"
            "1. Yes, create a support ticket\n"
            "2. No, continue without a ticket\n\n"
            "Please reply with 1 or 2."
        )
        meta = {"ticket_found": False, "offered_ticket_options": True}
        return draft, meta
    async def _handle_ticket_confirmation(self, message: str, session_id: str, metadata: dict) -> ChatResponse:
        msg_lower = message.lower().strip()

        intent_check, _, _ = classify_intent(message)
        if intent_check in (Intent.FAQ, Intent.ORDER_TRACKING, Intent.GREETING):
            await self._update_session_metadata(
                session_id,
                {
                    "offered_ticket_options": False,
                    "complaint_ticket_requested": False,
                    "state": ConversationState.NORMAL.value,
                },
            )
            await self._store_message(session_id, MessageRole.USER, message, intent_check, None)
            response_text, meta_out = await self._route(intent_check, message, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent_check, None)
            return ChatResponse(message=response_text, session_id=session_id, intent=intent_check, metadata=meta_out)

        option_2_phrases = {
            "2",
            "option 2",
            "second",
            "resolve here",
            "help me here",
            "self",
            "no ticket",
            "no thanks",
            "together",
            "dont create ticket",
            "don't create ticket",
            "without ticket",
            "no",
        }
        if any(phrase in msg_lower for phrase in option_2_phrases):
            await self._update_session_metadata(
                session_id,
                {
                    "offered_ticket_options": False,
                    "complaint_ticket_requested": False,
                    "state": ConversationState.NORMAL.value,
                },
            )
            await self._store_message(session_id, MessageRole.USER, message, Intent.GENERAL, None)
            history = await self._load_history(session_id)
            original_message = metadata.get("original_complaint_message", message)
            response = await self._llm_generate(
                user_message=original_message,
                extra_context=(
                    "The customer chose to resolve this without creating a ticket. Help them in chat with concrete next steps."
                ),
                history=history,
                support_context=self._build_support_context(metadata),
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, response, Intent.GENERAL, None)
            return ChatResponse(
                message=response,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"self_serve_chosen": True},
            )

        option_1_phrases = {
            "1",
            "option 1",
            "first",
            "create ticket",
            "make ticket",
            "raise ticket",
            "open ticket",
            "go ahead",
            "yes create",
            "yes ticket",
            "please create",
            "yes",
            "yeah",
            "y",
        }
        create_ticket_pattern = r"\b(create|open|make|raise)\s+(a\s+)?ticket\b"
        if any(phrase in msg_lower for phrase in option_1_phrases) or re.search(create_ticket_pattern, msg_lower):
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
            if existing_ticket:
                await self._sync_existing_ticket_metadata(session_id, existing_ticket, metadata)
                draft_reply, meta_out = self._build_existing_ticket_reply(existing_ticket, metadata)
                response_text = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=message,
                    draft=draft_reply,
                    facts=meta_out,
                    required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
                )
                await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
                return ChatResponse(
                    message=response_text,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata=meta_out,
                )

            metadata["complaint_ticket_requested"] = True
            await self._update_session_metadata(
                session_id,
                {
                    "complaint_ticket_requested": True,
                    "offered_ticket_options": False,
                },
            )
            if not metadata.get("ticket_context_collected"):
                response_text, meta_out = await self._advance_complaint_intake(
                    session_id,
                    metadata,
                    metadata.get("original_complaint_message", message),
                    source_intent=Intent.COMPLAINT,
                )
            else:
                response_text, meta_out = await self._begin_email_collection_for_ticket(
                    user_message=message,
                    session_id=session_id,
                    metadata=metadata,
                )
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
        clarification = (
            "Thanks for clarifying. Would you like me to:\n"
            "1. Create a support ticket\n"
            "2. Try to resolve this here\n\n"
            "Please reply with 1 or 2."
        )
        await self._store_message(session_id, MessageRole.ASSISTANT, clarification, Intent.COMPLAINT, None)
        return ChatResponse(
            message=clarification,
            session_id=session_id,
            intent=Intent.COMPLAINT,
            metadata={"offered_ticket_options": True},
        )

    async def _handle_email_collection(self, message: str, session_id: str, metadata: dict) -> ChatResponse:
        stripped = message.strip()
        msg_lower = stripped.lower()
        email_attempts = int(metadata.get("email_attempts", 0))
        awaiting_confirmation = bool(metadata.get("awaiting_email_confirmation"))
        pending_email = metadata.get("pending_email")
        existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
        if existing_ticket:
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            await self._sync_existing_ticket_metadata(session_id, existing_ticket, metadata)
            draft_reply, meta_out = self._build_existing_ticket_reply(existing_ticket, metadata)
            response_text = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft_reply,
                facts=meta_out,
                required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        intent_check, _, _ = classify_intent(message)
        if "@" not in stripped and self._is_ticket_flow_exit_request(msg_lower):
            await self._store_message(session_id, MessageRole.USER, message, Intent.GENERAL, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(
                session_id,
                cleared_updates,
            )
            metadata.update(cleared_updates)
            reply = "Understood. I canceled ticket creation for now. Please share your question, and I'll help right away."
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=reply,
                facts={"ticket_flow_cancelled": True},
                required_terms=["question"],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.GENERAL, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"ticket_flow_cancelled": True},
            )

        if intent_check in (Intent.FAQ, Intent.ORDER_TRACKING, Intent.GREETING) and "@" not in stripped:
            await self._store_message(session_id, MessageRole.USER, message, intent_check, None)
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(
                session_id,
                cleared_updates,
            )
            metadata.update(cleared_updates)
            response_text, meta_out = await self._route(intent_check, message, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent_check, None)
            route_meta = dict(meta_out or {})
            route_meta["ticket_flow_cancelled"] = True
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=intent_check,
                metadata=route_meta,
            )

        skip_words = {"skip", "no thanks", "skip it", "pass"}
        # During email-confirmation step, bare "no" means "change email", not "skip ticket email".
        if (
            msg_lower in skip_words
            or (msg_lower == "no" and not awaiting_confirmation)
            or self._is_email_refusal(msg_lower)
        ):
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            await self._update_session_metadata(
                session_id,
                {
                    "state": ConversationState.NORMAL.value,
                    "awaiting_email": False,
                    "awaiting_email_confirmation": False,
                    "pending_email": None,
                    "email_attempts": 0,
                    "customer_email": None,
                    "offered_ticket_options": False,
                },
            )
            metadata["customer_email"] = None
            complaint_text = metadata.get("original_complaint_message", message)
            response_text, meta_out = await self._handle_complaint(complaint_text, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        yes_words = {"yes", "y", "yeah", "yep", "correct", "right", "confirm", "ok", "okay"}
        no_change_words = {"no", "n", "change", "wrong", "different", "edit", "update"}
        if awaiting_confirmation:
            if msg_lower in yes_words:
                if not pending_email:
                    await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
                    await self._update_session_metadata(
                        session_id,
                        {
                            "state": ConversationState.EMAIL_COLLECTION.value,
                            "awaiting_email": True,
                            "awaiting_email_confirmation": False,
                            "pending_email": None,
                        },
                    )
                    reply = "I don't have an email to confirm yet. Please share it like name@example.com."
                    reply = await self._compose_operational_reply(
                        session_id=session_id,
                        user_message=message,
                        draft=reply,
                        facts={"awaiting_email": True, "awaiting_email_confirmation": False},
                        required_terms=["email"],
                    )
                    await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
                    return ChatResponse(
                        message=reply,
                        session_id=session_id,
                        intent=Intent.COMPLAINT,
                        metadata={"awaiting_email": True, "awaiting_email_confirmation": False},
                    )

                await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
                await self._update_session_metadata(
                    session_id,
                    {
                        "state": ConversationState.NORMAL.value,
                        "awaiting_email": False,
                        "awaiting_email_confirmation": False,
                        "pending_email": None,
                        "email_attempts": 0,
                        "customer_email": pending_email,
                        "offered_ticket_options": False,
                    },
                )
                metadata["customer_email"] = pending_email
                complaint_text = metadata.get("original_complaint_message", message)
                response_text, meta_out = await self._handle_complaint(complaint_text, session_id, metadata)
                await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
                return ChatResponse(
                    message=response_text,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata=meta_out,
                )

            extracted_updated_email = self._extract_email_candidate(stripped)
            if (
                msg_lower in no_change_words
                or "change" in msg_lower
                or "different" in msg_lower
                or "wrong" in msg_lower
            ):
                await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
                if extracted_updated_email:
                    await self._update_session_metadata(
                        session_id,
                        {
                            "state": ConversationState.EMAIL_COLLECTION.value,
                            "awaiting_email": True,
                            "awaiting_email_confirmation": True,
                            "pending_email": extracted_updated_email,
                        },
                    )
                    reply = (
                        f'I found this email: "{extracted_updated_email}". '
                        'Is this correct? Reply "yes" to confirm or "change" to provide another email.'
                    )
                    reply = await self._compose_operational_reply(
                        session_id=session_id,
                        user_message=message,
                        draft=reply,
                        facts={"pending_email": extracted_updated_email, "awaiting_email_confirmation": True},
                        required_terms=[extracted_updated_email, "yes", "change"],
                    )
                    await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
                    return ChatResponse(
                        message=reply,
                        session_id=session_id,
                        intent=Intent.COMPLAINT,
                        metadata={
                            "awaiting_email": True,
                            "awaiting_email_confirmation": True,
                            "extracted_email": extracted_updated_email,
                        },
                    )

                await self._update_session_metadata(
                    session_id,
                    {
                        "state": ConversationState.EMAIL_COLLECTION.value,
                        "awaiting_email": True,
                        "awaiting_email_confirmation": False,
                        "pending_email": None,
                    },
                )
                reply = "No problem. Please share the email you'd like us to use for ticket follow-up."
                reply = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=message,
                    draft=reply,
                    facts={"awaiting_email": True, "awaiting_email_confirmation": False},
                    required_terms=["email"],
                )
                await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
                return ChatResponse(
                    message=reply,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata={"awaiting_email": True, "awaiting_email_confirmation": False},
                )

            if extracted_updated_email:
                await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
                await self._update_session_metadata(
                    session_id,
                    {
                        "state": ConversationState.EMAIL_COLLECTION.value,
                        "awaiting_email": True,
                        "awaiting_email_confirmation": True,
                        "pending_email": extracted_updated_email,
                    },
                )
                reply = (
                    f'I found this email: "{extracted_updated_email}". '
                    'Is this correct? Reply "yes" to confirm or "change" to provide another email.'
                )
                reply = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=message,
                    draft=reply,
                    facts={"pending_email": extracted_updated_email, "awaiting_email_confirmation": True},
                    required_terms=[extracted_updated_email, "yes", "change"],
                )
                await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
                return ChatResponse(
                    message=reply,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata={
                        "awaiting_email": True,
                        "awaiting_email_confirmation": True,
                        "extracted_email": extracted_updated_email,
                    },
                )

            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            reply = (
                f'I currently have "{pending_email}" as your email. '
                'Reply "yes" to confirm or "change" to provide a different one.'
            )
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=reply,
                facts={"pending_email": pending_email, "awaiting_email_confirmation": True},
                required_terms=[pending_email or "", "yes", "change"],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata={
                    "awaiting_email": True,
                    "awaiting_email_confirmation": True,
                    "extracted_email": pending_email,
                },
            )

        if EMAIL_RE.match(stripped):
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            await self._update_session_metadata(
                session_id,
                {
                    "state": ConversationState.NORMAL.value,
                    "awaiting_email": False,
                    "awaiting_email_confirmation": False,
                    "pending_email": None,
                    "email_attempts": 0,
                    "customer_email": stripped,
                    "offered_ticket_options": False,
                },
            )
            metadata["customer_email"] = stripped
            complaint_text = metadata.get("original_complaint_message", message)
            response_text, meta_out = await self._handle_complaint(complaint_text, session_id, metadata)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        extracted_email = self._extract_email_candidate(stripped)
        if extracted_email:
            await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
            await self._update_session_metadata(
                session_id,
                {
                    "state": ConversationState.EMAIL_COLLECTION.value,
                    "awaiting_email": True,
                    "awaiting_email_confirmation": True,
                    "pending_email": extracted_email,
                    "email_attempts": email_attempts,
                },
            )
            reply = (
                f'I found this email: "{extracted_email}". '
                'Is this correct? Reply "yes" to confirm or "change" to provide another email.'
            )
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=reply,
                facts={"pending_email": extracted_email, "awaiting_email_confirmation": True},
                required_terms=[extracted_email, "yes", "change"],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata={
                    "awaiting_email": True,
                    "awaiting_email_confirmation": True,
                    "extracted_email": extracted_email,
                },
            )

        await self._store_message(session_id, MessageRole.USER, message, Intent.COMPLAINT, None)
        email_attempts += 1

        if email_attempts >= 3:
            cleared_updates = self._clear_complaint_pending_updates()
            await self._update_session_metadata(
                session_id,
                cleared_updates,
            )
            metadata.update(cleared_updates)
            reply = (
                "I still couldn't validate an email, so I paused ticket creation for now. "
                "If you want to try again, say 'create ticket' anytime."
            )
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=reply,
                facts={"email_collection_abandoned": True},
                required_terms=["create ticket"],
            )
            await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.GENERAL, None)
            return ChatResponse(
                message=reply,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"email_collection_abandoned": True},
            )
        else:
            await self._update_session_metadata(
                session_id,
                {
                    "state": ConversationState.EMAIL_COLLECTION.value,
                    "awaiting_email": True,
                    "awaiting_email_confirmation": False,
                    "pending_email": None,
                    "email_attempts": email_attempts,
                },
            )
            reply = (
                f"That email format doesn't look right yet. "
                "Please use name@example.com, or type 'skip' to continue without email."
            )
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=reply,
                facts={
                    "awaiting_email": True,
                    "awaiting_email_confirmation": False,
                    "email_attempts": min(email_attempts, 3),
                },
                required_terms=["email", "skip"],
            )

        await self._store_message(session_id, MessageRole.ASSISTANT, reply, Intent.COMPLAINT, None)
        return ChatResponse(
            message=reply,
            session_id=session_id,
            intent=Intent.COMPLAINT,
            metadata={
                "awaiting_email": True,
                "awaiting_email_confirmation": False,
                "email_attempts": min(email_attempts, 3),
            },
        )

    async def _route(self, intent: Intent, message: str, session_id: str, metadata: dict) -> tuple[str, dict | None]:
        try:
            if intent == Intent.FAQ:
                return await self._handle_faq(message, session_id)
            if intent == Intent.ORDER_TRACKING:
                return await self._handle_order(message, session_id, metadata)
            if intent == Intent.GREETING:
                return await self._handle_greeting(session_id)
            return await self._handle_general(message, session_id)
        except Exception as exc:
            logger.error("Route error (%s): %s", intent, exc, exc_info=True)
            return "I'm sorry, I'm having trouble with that right now. Please try again.", None

    async def _handle_frustration(
        self,
        message: str,
        session_id: str,
        frustration: FrustrationLevel,
        sentiment_score: float,
    ) -> tuple[str, dict | None]:
        existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
        if existing_ticket:
            session_metadata = await self._get_session_metadata(session_id)
            await self._sync_existing_ticket_metadata(session_id, existing_ticket, session_metadata)
            draft, meta = self._build_existing_ticket_reply(existing_ticket, session_metadata)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=meta,
                required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
            )
            return reply, meta

        ticket_requested = sentiment_score <= AUTO_ESCALATE_THRESHOLD
        intake_updates = {
            "original_complaint_message": message,
            "original_sentiment_score": sentiment_score,
            "complaint_ticket_requested": ticket_requested,
        }
        await self._update_session_metadata(
            session_id,
            intake_updates,
        )
        metadata = await self._get_session_metadata(session_id)
        meta = {
            "frustration_detected": True,
            "frustration_level": frustration.value,
            "sentiment_score": sentiment_score,
            "auto_escalate": ticket_requested,
        }
        reply, intake_meta = await self._start_complaint_intake(
            message,
            session_id,
            metadata,
            ticket_requested=ticket_requested,
            source_intent=Intent.COMPLAINT,
        )
        combined_meta = dict(intake_meta or {})
        combined_meta.update(meta)
        return reply, combined_meta

    async def _handle_complaint(self, message: str, session_id: str, metadata: dict) -> tuple[str, dict | None]:
        existing_ticket = await self.ticket_service.get_latest_ticket_for_session(session_id)
        if existing_ticket:
            await self._sync_existing_ticket_metadata(session_id, existing_ticket, metadata)
            draft, meta = self._build_existing_ticket_reply(existing_ticket, metadata)
            reply = await self._compose_operational_reply(
                session_id=session_id,
                user_message=message,
                draft=draft,
                facts=meta,
                required_terms=[existing_ticket.id[:8], existing_ticket.customer_email or ""],
            )
            return reply, meta

        customer_email = metadata.get("customer_email")
        order_id = metadata.get("active_order_id") if metadata.get("active_order_confirmed") else None
        ticket_message = self._build_ticket_description(metadata, fallback_message=message)
        ticket_category = metadata.get("issue_category") or "complaint"

        complaint_req = ComplaintRequest(
            message=ticket_message,
            session_id=session_id,
            customer_email=customer_email,
            order_id=order_id,
        )
        ticket = await self.ticket_service.create_ticket(complaint_req, category=ticket_category)

        await self._update_session_metadata(
            session_id,
            {
                "has_open_ticket": True,
                "ticket_id": ticket.id,
                "offered_ticket_options": False,
                "awaiting_email": False,
                "awaiting_email_confirmation": False,
                "pending_email": None,
                "email_attempts": 0,
                "ticket_context_collected": True,
                "complaint_ticket_requested": False,
                "state": ConversationState.NORMAL.value,
            },
        )

        ticket_id_short = ticket.id[:8]
        if not customer_email:
            if ticket.status.value == "escalated":
                response = (
                    f"I created urgent ticket #{ticket_id_short} and escalated it to our senior team. "
                    "Since we don't have your email, please reference this ticket when contacting "
                    "our support team directly."
                )
            else:
                response = (
                    f"I created ticket #{ticket_id_short} with {ticket.priority.value} priority. "
                    "Since we don't have your email, please reference this ticket when contacting "
                    "our support team directly."
                )
        else:
            if ticket.status.value == "escalated":
                response = (
                    f"I created urgent ticket #{ticket_id_short} and escalated it. "
                    f"Our senior team will follow up at {customer_email}."
                )
            else:
                response = (
                    f"I created ticket #{ticket_id_short} with {ticket.priority.value} priority. "
                    f"Our team will follow up at {customer_email}."
                )

        meta = {
            "ticket_id": ticket.id,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "has_email": bool(customer_email),
            "issue_category": ticket_category,
        }
        if order_id:
            meta["order_id"] = order_id
        reply = await self._compose_operational_reply(
            session_id=session_id,
            user_message=message,
            draft=response,
            facts=meta,
            required_terms=[ticket.id[:8], customer_email or "", ticket.status.value],
        )
        return (reply, meta)
    async def _handle_faq(self, message: str, session_id: str) -> tuple[str, dict | None]:
        results = await self.vector_store.search(message, top_k=3)
        history = await self._load_history(session_id)
        support_context = self._build_support_context(await self._get_session_metadata(session_id))

        if not results or results[0].score < 0.55:
            response = await self._llm_generate(
                user_message=message,
                extra_context=FAQ_NO_MATCH_PROMPT,
                history=history,
                support_context=support_context,
            )
            return response, {"faq_match": False}

        context = "\n\n".join(f"Q: {r.entry.question}\nA: {r.entry.answer}" for r in results)
        response = await self._llm_generate(
            user_message=message,
            extra_context=FAQ_GROUNDED_PROMPT.format(context=context),
            history=history,
            support_context=support_context,
        )
        return response, {"faq_sources": [r.entry.question for r in results], "faq_match": True}

    async def _handle_order(self, message: str, session_id: str, metadata: dict) -> tuple[str, dict | None]:
        order_id = None
        message_lower = message.lower()
        state = self._resolve_state(metadata)

        if looks_like_explicit_order_reference(message) or "order" in message_lower:
            order_id = normalize_order_id(message)

        if not order_id and metadata.get("active_order_confirmed"):
            follow_up_words = {
                "order", "it", "status", "where", "when", "tracking", "delivery", "shipped", "package",
            }
            if any(word in message_lower for word in follow_up_words):
                order_id = metadata.get("active_order_id")

        if order_id:
            try:
                order = await self.order_service.get_by_id(order_id)

                await self._update_session_metadata(
                    session_id,
                    {
                        "awaiting_order_id": False,
                        "active_order_id": order_id,
                        "active_order_confirmed": True,
                        "active_order_context_idle_turns": 0,
                        "order_lookup_failures": 0,
                        "state": ConversationState.NORMAL.value,
                    },
                )

                history = await self._load_history(session_id)

                if order.status.value == "cancelled":
                    response = await self._llm_generate(
                        user_message=message,
                        extra_context=(
                            f"Order {order.id} is cancelled. Items: "
                            f"{', '.join(f'{i.name} (x{i.quantity})' for i in order.items)}. "
                            f"Total: ${order.total:.2f}. Explain clearly and offer help."
                        ),
                        history=history,
                        support_context=self._build_support_context({**metadata, "active_order_id": order_id, "active_order_confirmed": True}),
                    )
                    return response, {"order_id": order_id, "status": "cancelled"}

                if order.tracking_number is None and order.status.value in {"processing", "pending", "confirmed"}:
                    response = await self._llm_generate(
                        user_message=message,
                        extra_context=(
                            f"Order {order.id} status is {order.status.value}. It has not shipped yet, "
                            "so no tracking number is available. "
                            f"Estimated delivery: {order.estimated_delivery or 'not yet confirmed'}."
                        ),
                        history=history,
                        support_context=self._build_support_context({**metadata, "active_order_id": order_id, "active_order_confirmed": True}),
                    )
                    return (
                        response,
                        {"order_id": order_id, "status": order.status.value, "tracking_available": False},
                    )

                summary = await self.order_service.get_status_summary(order_id)
                meta = {"order_id": order_id, "status": order.status.value}
                summary = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=message,
                    draft=summary,
                    facts=meta,
                    required_terms=[order_id, order.status.value],
                )
                return summary, meta

            except OrderNotFoundError:
                next_attempt_count = int(metadata.get("order_lookup_failures", 0)) + 1
                await self._update_session_metadata(
                    session_id,
                    {
                        "awaiting_order_id": True,
                        "active_order_id": None,
                        "active_order_confirmed": False,
                        "active_order_context_idle_turns": 0,
                        "order_lookup_failures": next_attempt_count,
                        "state": ConversationState.AWAITING_ORDER.value,
                    },
                )
                metadata.update(
                    {
                        "awaiting_order_id": True,
                        "active_order_id": None,
                        "active_order_confirmed": False,
                        "active_order_context_idle_turns": 0,
                        "order_lookup_failures": next_attempt_count,
                        "state": ConversationState.AWAITING_ORDER.value,
                    }
                )
                draft = self._build_order_not_found_reply(order_id, attempt_count=next_attempt_count)
                meta = {"order_id": order_id, "found": False, "awaiting_order_id": True}
                reply = await self._compose_operational_reply(
                    session_id=session_id,
                    user_message=message,
                    draft=draft,
                    facts=self._build_support_context(metadata) | meta,
                    required_terms=[order_id, "ord-1001"],
                )
                return (reply, meta)
            except Exception as exc:
                logger.error("Order lookup failed (%s): %s", order_id, exc, exc_info=True)
                return (
                    "I'm sorry, I couldn't look up that order right now. Please try again in a moment.",
                    {"order_id": order_id, "lookup_error": True},
                )

        await self._update_session_metadata(
            session_id,
            {
                "awaiting_order_id": True,
                "active_order_context_idle_turns": 0,
                "order_lookup_failures": 0,
                "state": ConversationState.AWAITING_ORDER.value,
            },
        )
        draft = "I'd be happy to track your order. Please share your order ID (for example, ORD-1001)."
        reply = await self._compose_operational_reply(
            session_id=session_id,
            user_message=message,
            draft=draft,
            facts={"awaiting_order_id": True},
            required_terms=["order id", "ord-1001"],
        )
        return reply, None

    async def _handle_greeting(self, session_id: str) -> tuple[str, dict | None]:
        draft = (
            "Hi there, Welcome to customer support. I can help with FAQs, order tracking, and support tickets. "
            "What would you like to do first?"
        )
        reply = await self._compose_operational_reply(
            session_id=session_id,
            user_message="greeting",
            draft=draft,
            facts={"capabilities": ["faq", "order_tracking", "support_tickets"]},
            required_terms=["faq", "order", "support"],
        )
        return (reply, None)

    async def _handle_general(self, message: str, session_id: str) -> tuple[str, dict | None]:
        history = await self._load_history(session_id)
        support_context = self._build_support_context(await self._get_session_metadata(session_id))
        response = await self._llm_generate(user_message=message, history=history, support_context=support_context)
        return response, None

    def _infer_required_terms_from_draft(self, draft: str) -> list[str]:
        """Infer non-negotiable terms that must survive LLM rewriting."""
        lower = draft.lower()
        terms: list[str] = []

        for email in EMAIL_CANDIDATE_RE.findall(draft):
            terms.append(email.lower())
        for order_id in re.findall(r"\bORD-[A-Z0-9]+\b", draft, re.IGNORECASE):
            terms.append(order_id.lower())
        for ticket_short in re.findall(r"#([a-zA-Z0-9]{4,12})\b", draft):
            terms.append(ticket_short.lower())

        if "reply with 1 or 2" in lower or ("1." in draft and "2." in draft):
            terms.extend(["1", "2"])
        if "email" in lower:
            terms.append("email")
        if "skip" in lower:
            terms.append("skip")
        if "order id" in lower:
            terms.append("order id")
        if "ticket" in lower:
            terms.append("ticket")

        # Preserve order while removing duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for t in terms:
            norm = t.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        return deduped

    def _is_composed_reply_safe(self, draft: str, candidate: str, required_terms: list[str]) -> bool:
        """Validate LLM rewrite keeps critical information."""
        cand = candidate.strip()
        if not cand:
            return False
        if len(cand) < max(20, min(len(draft) // 3, 60)):
            return False

        cand_lower = cand.lower()
        for term in required_terms:
            if term and term.lower() not in cand_lower:
                return False

        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "your", "you", "our",
            "will", "have", "has", "are", "was", "were", "not", "can", "please", "reply",
            "share", "team", "help", "would", "like", "just", "here", "there", "them",
        }
        draft_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9@#.\-]+", draft.lower())
            if len(tok) >= 4 and tok not in stopwords
        }
        if draft_tokens:
            cand_tokens = set(re.findall(r"[a-z0-9@#.\-]+", cand_lower))
            overlap = len(draft_tokens & cand_tokens) / len(draft_tokens)
            if overlap < 0.18:
                return False

        return True

    async def _compose_operational_reply(
        self,
        *,
        session_id: str,
        user_message: str,
        draft: str,
        facts: dict | None = None,
        required_terms: list[str] | None = None,
    ) -> str:
        """Use LLM to polish deterministic replies while preserving exact facts."""
        merged_required = self._infer_required_terms_from_draft(draft)
        if required_terms:
            for term in required_terms:
                if term:
                    merged_required.append(term.lower())
        merged_required = list(dict.fromkeys(merged_required))

        session_metadata = await self._get_session_metadata(session_id)
        support_context = self._build_support_context(session_metadata)
        context = OPERATIONS_COMPOSER_PROMPT.format(
            facts_json=json.dumps(facts or {}, ensure_ascii=True, sort_keys=True),
            draft=draft,
        )
        llm_support_context = None if facts == support_context else support_context
        history = await self._load_history(session_id)
        try:
            candidate = await self._llm_generate(
                user_message=user_message,
                extra_context=context,
                history=history,
                support_context=llm_support_context,
            )
        except Exception as exc:
            logger.warning("Operational reply composition failed, using draft. error=%s", exc)
            return draft

        if self._is_composed_reply_safe(draft=draft, candidate=candidate, required_terms=merged_required):
            return candidate.strip()
        return draft

    async def _llm_generate(
        self,
        user_message: str,
        extra_context: str = "",
        history: list[Message] | None = None,
        support_context: dict | None = None,
    ) -> str:
        system = SYSTEM_PROMPT
        if extra_context:
            system = f"{system}\n\n{extra_context}"
        if support_context:
            system = f"{system}\n\nSupport context:\n{json.dumps(support_context, ensure_ascii=True, sort_keys=True)}"

        messages: list[Message] = []
        if history:
            messages.extend(history)
        messages.append(Message(role=MessageRole.USER, content=user_message))

        return await self.llm.generate(messages, system_prompt=system)

    async def _load_history(self, session_id: str) -> list[Message]:
        try:
            result = await self.db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(CONVERSATION_HISTORY_LIMIT)
            )
            rows = list(reversed(result.scalars().all()))
            return [Message(role=MessageRole(row.role), content=row.content) for row in rows]
        except Exception as exc:
            logger.warning("History load failed (%s): %s", session_id, exc)
            return []
    async def _ensure_session(self, session_id: str, channel: str) -> None:
        result = await self.db.execute(select(ChatSession).where(ChatSession.id == session_id))
        if not result.scalar_one_or_none():
            self.db.add(
                ChatSession(
                    id=session_id,
                    channel=channel,
                    metadata_={"state": ConversationState.NORMAL.value},
                )
            )
            await self.db.flush()

    async def _get_session_metadata(self, session_id: str) -> dict:
        try:
            result = await self.db.execute(select(ChatSession).where(ChatSession.id == session_id))
            session = result.scalar_one_or_none()
            if session and session.metadata_:
                return dict(session.metadata_)
            return {"state": ConversationState.NORMAL.value}
        except Exception as exc:
            logger.warning("Metadata load failed (%s): %s", session_id, exc)
            return {"state": ConversationState.NORMAL.value}

    async def _update_session_metadata(self, session_id: str, updates: dict) -> None:
        try:
            from sqlalchemy.orm import attributes

            result = await self.db.execute(select(ChatSession).where(ChatSession.id == session_id))
            session = result.scalar_one_or_none()
            if session:
                current = dict(session.metadata_ or {})
                current.update(updates)
                session.metadata_ = current
                attributes.flag_modified(session, "metadata_")
                await self.db.flush()
        except Exception as exc:
            logger.warning("Metadata update failed (%s): %s", session_id, exc)

    async def _store_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        intent: Intent,
        sentiment_score: float | None = None,
    ) -> None:
        self.db.add(
            ChatMessage(
                session_id=session_id,
                role=role.value,
                content=content,
                intent=intent.value,
                sentiment_score=sentiment_score,
            )
        )
        await self.db.flush()
