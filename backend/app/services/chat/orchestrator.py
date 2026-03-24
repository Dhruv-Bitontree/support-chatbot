"""Chat orchestrator: routes user messages to the appropriate service.

CHANGES (TARGETED FIX):
- BUG 1: Added sentiment gate - positive sentiment (>0.0) with low frustration → no ticket
- BUG 2: Fixed ask-before-ticket flow - now actually triggers before ticket creation
- BUG 3: Added state machine - checks metadata flags BEFORE intent classification
- BUG 4: Added email collection state with validation
- BUG 5: Fixed hallucination - explicit contact info handling, never invents email/phone
- BUG 6: Removed "added to existing ticket" lie - now just acknowledges existing ticket

STATE MACHINE FIX:
- Added ConversationState enum for explicit state tracking
- Session locking after ticket creation
- Bounded email collection (max 2 attempts)
- Simplified support options (2 instead of 3)
"""

import json
import logging
import re
import uuid
from enum import Enum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.models.chat import ChatRequest, ChatResponse, Intent, Message, MessageRole
from app.models.complaint import ComplaintRequest
from app.services.chat.intent import (
    FrustrationLevel,
    classify_intent,
    classify_intent_llm,
)
from app.services.complaints.sentiment import analyze_sentiment
from app.services.complaints.ticket_service import TicketService
from app.services.faq.base import VectorStore
from app.services.llm.base import LLMProvider
from app.services.orders.order_service import OrderService

logger = logging.getLogger(__name__)


# ── Conversation State Machine ────────────────────────────────────────────────

class ConversationState(str, Enum):
    """Explicit conversation states for state machine tracking.
    
    String enum for JSON serialization compatibility with session metadata.
    """
    NORMAL_CHAT = "NORMAL_CHAT"  # Default state - normal conversation flow
    AWAITING_ORDER_ID = "AWAITING_ORDER_ID"  # Waiting for user to provide order ID
    FRUSTRATION_DETECTED = "FRUSTRATION_DETECTED"  # Frustration detected, before offering options
    SUPPORT_OPTIONS = "SUPPORT_OPTIONS"  # Options offered, waiting for user choice
    EMAIL_COLLECTION = "EMAIL_COLLECTION"  # Collecting email for ticket creation
    TICKET_CREATION = "TICKET_CREATION"  # Creating ticket (transient state)
    SESSION_LOCKED = "SESSION_LOCKED"  # Ticket created, session closed


# ── Constants ──────────────────────────────────────────────────────────────────

# How many previous messages to load into LLM context.
# Keeps token usage bounded while giving the bot short-term memory.
CONVERSATION_HISTORY_LIMIT = 6  # 3 user + 3 assistant turns

# ── Input validation patterns ─────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(you\s+are|your\s+role|everything)",
    r"you\s+are\s+now\s+(DAN|a\s+different)",
    r"system\s*:\s*",
    r"<\s*script\s*>",
    r"reveal\s+(your\s+)?(prompt|instructions|system)",
]

# ── Email validation ──────────────────────────────────────────────────────────
# BUG 4: Basic email validation for collection flow

EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# ── System prompt ────────────────────────────────────────────────────────────
# STEP 8: Enhanced with tone rules and stricter anti-hallucination guidelines

SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant.

You help customers with:
1. Answering frequently asked questions about products, shipping, returns, etc.
2. Tracking their orders by order ID.
3. Handling complaints and creating support tickets when needed.

STRICT PROHIBITIONS — follow these without exception:
- NEVER invent, guess, or assume any business-specific information such as:
  prices, fees, percentages, dollar amounts, return windows, shipping times,
  phone numbers, email addresses, order statuses, tracking numbers, delivery dates.
- If you do not have verified data from the FAQ entries or the order database,
  say clearly: "I don't have that information — please contact our support team
  directly at support@example.com or 1-800-555-0123."
- NEVER make up an order status, tracking number, or delivery estimate.
  Only report what the order database has returned to you.
- NEVER say "our policy is..." unless that exact policy is in the provided FAQ context.
- NEVER promise outcomes like "we'll resolve this in 24 hours" or "you'll get a refund".
- BUG 5 FIX: NEVER say "via the email address associated with your account" or
  "by phone if provided" unless you have verified that information from the session
  metadata. If you do not have the customer's contact details, say: "Since we don't
  have your contact details on file, please reference your ticket ID when following
  up with us at support@example.com or 1-800-555-0123."
- For order tracking: always ask for the order ID if not provided.
  Do NOT speculate about order status without a confirmed database lookup.
- For complaints: be empathetic and confirm the ticket number. Do NOT promise
  specific resolution timelines or outcomes.

TONE RULES:
- Never be dismissive of customer frustration.
- Never say "I understand your frustration" as a hollow opener without substance.
- Be specific about what action was taken (ticket # created, order status found).
- Short responses for simple queries, fuller responses for complaints.
- When a customer is frustrated, acknowledge it sincerely and offer concrete next steps.

If asked something not in your knowledge: direct to support contacts explicitly.
If order data is null/missing: say clearly "that information isn't available yet".
If asked about timelines: say "I can't give a specific timeframe".
If asked to do something outside your scope: redirect to human support.
"""

# ── FAQ-specific prompt ───────────────────────────────────────────────────────
# FIX 1 (continued): Two separate FAQ prompts — one grounded, one for no-match.
# The original single prompt used "general knowledge" as fallback which caused
# hallucinated policy answers.

FAQ_GROUNDED_PROMPT = """Answer the user's question using ONLY the FAQ entries provided below.
Do not add any information that is not present in these entries.
Synthesize a natural, conversational response — do not say "according to our FAQ".
If the FAQ entries do not fully answer the question, say so and suggest
the customer contact support directly.

FAQ entries:
{context}"""

FAQ_NO_MATCH_PROMPT = """The user is asking a question but we don't have a specific FAQ entry for it.

DO NOT guess or invent an answer.
Politely tell the customer you don't have that specific information,
and offer to create a support ticket so the team can help them directly.

Keep it brief and empathetic."""


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

        # Input validation before any processing
        validation_error = self._validate_input(request.message)
        if validation_error:
            return ChatResponse(
                message=validation_error,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"validation_error": True},
            )

        # Ensure session exists
        await self._ensure_session(session_id, request.channel)

        # STATE MACHINE FIX: Load metadata FIRST and check for SESSION_LOCKED state
        metadata_dict = await self._get_session_metadata(session_id)
        
        # CRITICAL: Check if session is locked (ticket already created)
        current_state = metadata_dict.get("state")
        if current_state == ConversationState.SESSION_LOCKED.value:
            # Session is locked - return closure message immediately
            return ChatResponse(
                message="Your ticket has been created with the email you provided. Our team will contact you soon. To start a new conversation, start a new session.",
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata={"session_locked": True},
            )
        
        # BUG 3 FIX: Check state flags BEFORE intent classification
        
        # STATE MACHINE: Check if we're in a specific state
        if current_state == ConversationState.EMAIL_COLLECTION.value or metadata_dict.get("awaiting_email"):
            # BUG 4: Email collection state
            return await self._handle_email_collection(request.message, session_id, metadata_dict)
        
        if current_state == ConversationState.SUPPORT_OPTIONS.value or metadata_dict.get("offered_ticket_options"):
            # BUG 3: Confirmation state
            return await self._handle_ticket_confirmation(request.message, session_id, metadata_dict)
        
        # AWAITING_ORDER_ID state: User should provide order ID
        if current_state == ConversationState.AWAITING_ORDER_ID.value or metadata_dict.get("awaiting_order_id"):
            # Route to order handler which will extract the ID
            response_text, metadata_out = await self._handle_order(request.message, session_id, metadata_dict)
            await self._store_message(session_id, MessageRole.ASSISTANT, response_text, Intent.ORDER_TRACKING, None)
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=Intent.ORDER_TRACKING,
                metadata=metadata_out,
            )

        # Normal flow: classify intent
        intent, frustration, sarcasm = classify_intent(request.message)

        # If ambiguous (GENERAL), try LLM classification
        if intent == Intent.GENERAL:
            intent, frustration, sarcasm = await classify_intent_llm(request.message, self.llm)

        # Calculate sentiment for this message
        sentiment = analyze_sentiment(request.message)
        sentiment_score = sentiment.score
        
        # Clear stale order context if user switches topics
        if metadata_dict.get("active_order_confirmed") and intent not in [Intent.ORDER_TRACKING]:
            # Check if message is unrelated to orders
            message_lower = request.message.lower()
            order_keywords = ["order", "delivery", "tracking", "package", "shipment", "my order", "the order"]
            if not any(keyword in message_lower for keyword in order_keywords):
                # User switched topics - clear order context
                await self._update_session_metadata(session_id, {
                    "active_order_id": None,
                    "active_order_confirmed": False,
                    "awaiting_order_id": False
                })
                logger.info(f"Cleared stale order context - user switched to {intent.value}")

        # Store user message with sentiment
        await self._store_message(
            session_id, MessageRole.USER, request.message, intent, sentiment_score
        )

        # Load history for frustration tracking
        history = await self._load_history(session_id)

        # BUG 1 FIX: Sentiment gate - don't create ticket for positive sentiment
        if intent == Intent.COMPLAINT:
            # Check if this is genuinely negative or just weak complaint wording
            if sentiment_score > 0.0 and frustration in (FrustrationLevel.NONE, FrustrationLevel.MILD):
                # Positive sentiment with weak frustration → treat as feedback, not complaint
                logger.info(f"Sentiment gate: score={sentiment_score:.2f}, frustration={frustration.value} → routing to GENERAL")
                intent = Intent.GENERAL

        # BUG 2 FIX: Route to frustration handler BEFORE creating ticket
        # CRITICAL: Even auto-escalation requires email collection first
        if intent == Intent.COMPLAINT:
            # Always go through frustration handler to collect email
            # The handler will mark it as urgent based on sentiment score
            response_text, metadata_out = await self._handle_frustration(
                request.message, session_id, intent, frustration, sentiment_score, metadata_dict
            )
        else:
            # Normal routing for non-complaint intents
            response_text, metadata_out = await self._route(
                intent, request.message, session_id, metadata_dict
            )

        # Store assistant message
        await self._store_message(session_id, MessageRole.ASSISTANT, response_text, intent, None)

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            intent=intent,
            metadata=metadata_out,
        )

    def _validate_input(self, message: str) -> str | None:
        """Validate user input before processing.
        
        STEP 2: Detect empty input, prompt injection, emoji-only messages.
        Returns error message if invalid, None if valid.
        """
        # Strip whitespace
        stripped = message.strip()
        
        # Check empty
        if not stripped:
            return "I didn't receive a message. Could you please type your question or concern?"
        
        # Check emoji-only or non-alphanumeric garbage
        alphanumeric_count = sum(c.isalnum() for c in stripped)
        if alphanumeric_count == 0:
            return "I'm having trouble understanding that. Could you please rephrase your question using words?"
        
        # Check prompt injection patterns
        message_lower = message.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(f"Prompt injection attempt detected: {message[:100]}")
                return (
                    "I'm here to help with customer support questions about orders, "
                    "products, and policies. How can I assist you today?"
                )
        
        return None  # Valid input

    async def _handle_ticket_confirmation(
        self, message: str, session_id: str, metadata: dict
    ) -> ChatResponse:
        """BUG 3 FIX: Handle customer's response to ticket options.
        
        State: SUPPORT_OPTIONS
        Parse response and route accordingly.
        STATE MACHINE FIX: Now handles only 2 options and updates states properly.
        """
        message_lower = message.lower().strip()
        
        # Check for ticket creation confirmation (Option 1)
        ticket_keywords = ["1", "ticket", "create", "yes", "go ahead", "option 1", "first"]
        if any(kw in message_lower for kw in ticket_keywords):
            # Customer wants a ticket - clear flag and go to email collection
            
            # Check if we already have email
            if metadata.get("customer_email"):
                # Have email - create ticket directly
                response_text, meta_out = await self._handle_complaint(
                    message, session_id, metadata, auto_escalate=False
                )
                return ChatResponse(
                    message=response_text,
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata=meta_out,
                )
            else:
                # Need email - enter email collection state
                await self._update_session_metadata(session_id, {
                    "awaiting_email": True,
                    "offered_ticket_options": False,
                    "state": ConversationState.EMAIL_COLLECTION.value
                })
                return ChatResponse(
                    message="Could you share your email address so our team can follow up with you? (You can type 'skip' if you prefer not to)",
                    session_id=session_id,
                    intent=Intent.COMPLAINT,
                    metadata={"awaiting_email": True},
                )
        
        # Check for self-serve request (Option 2 - was Option 3)
        self_serve_keywords = ["2", "here", "resolve", "no ticket", "self", "option 2", "second", "together"]
        if any(kw in message_lower for kw in self_serve_keywords):
            # Customer wants to resolve here - clear flag and route to general
            await self._update_session_metadata(session_id, {
                "offered_ticket_options": False,
                "state": ConversationState.NORMAL_CHAT.value
            })
            history = await self._load_history(session_id)
            response = await self._llm_generate(message, history=history)
            await self._store_message(session_id, MessageRole.ASSISTANT, response, Intent.GENERAL, None)
            return ChatResponse(
                message=response,
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"self_serve_chosen": True},
            )
        
        # Unclear response - ask again (only 2 options now)
        return ChatResponse(
            message="Just to confirm — would you like me to:\n1️⃣ Create a support ticket\n2️⃣ Try to resolve this here together\n\nPlease let me know which option you prefer.",
            session_id=session_id,
            intent=Intent.COMPLAINT,
            metadata={"offered_ticket_options": True},
        )

    async def _handle_email_collection(
        self, message: str, session_id: str, metadata: dict
    ) -> ChatResponse:
        """Handle email collection state with 3-attempt limit.

        State: EMAIL_COLLECTION
        CRITICAL: NO ticket is created without a valid email address.
        After 3 failed attempts, abandon ticket flow and return to normal chat.
        """
        message_stripped = message.strip()
        message_lower = message_stripped.lower()

        # Get current email attempt count (0-indexed, so 0, 1, 2 = 3 attempts)
        email_attempts = metadata.get("email_attempts", 0)

        # Check if customer refuses to provide email
        if message_lower in ["skip", "no", "no thanks", "skip it", "no email", "don't want to"]:
            email_attempts += 1

            # Check if max attempts exceeded
            if email_attempts >= 3:
                # Abandon ticket flow - clear all ticket-related state
                await self._update_session_metadata(session_id, {
                    "state": ConversationState.NORMAL_CHAT.value,
                    "awaiting_email_for_ticket": False,
                    "awaiting_email": False,
                    "email_attempts": 0,
                    "ticket_flow_active": False,
                    "offered_support_options": False
                })

                return ChatResponse(
                    message="I understand you prefer not to share your email. I'm here to help with other questions you may have.",
                    session_id=session_id,
                    intent=Intent.GENERAL,
                    metadata={"email_collection_abandoned": True}
                )

            # Still have attempts left - ask again
            await self._update_session_metadata(session_id, {"email_attempts": email_attempts})

            if email_attempts == 1:
                message_text = "I need your email address to create a support ticket so our team can reach you. Could you please provide your email?"
            else:  # email_attempts == 2 (last chance)
                message_text = "This is my last request - I need a valid email to create your ticket. Please provide your email, or let me know if you'd prefer to continue without creating a ticket."

            return ChatResponse(
                message=message_text,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata={"awaiting_email": True, "email_attempts": email_attempts}
            )

        # Check if it looks like a valid email
        if "@" in message_stripped and EMAIL_RE.match(message_stripped):
            # Valid email - store it and create ticket
            await self._update_session_metadata(session_id, {
                "awaiting_email": False,
                "awaiting_email_for_ticket": False,
                "customer_email": message_stripped,
                "state": ConversationState.TICKET_CREATION.value,
                "email_attempts": 0  # Reset counter
            })

            # Update metadata with email for ticket creation
            metadata["customer_email"] = message_stripped

            # Get the original complaint message from metadata (stored when frustration was detected)
            complaint_message = metadata.get("original_complaint_message", message)

            response_text, meta_out = await self._handle_complaint(
                complaint_message, session_id, metadata, auto_escalate=False
            )

            # Confirm ticket creation with email
            ticket_id = meta_out.get("ticket_id", "UNKNOWN")[:8]
            response_with_email = (
                f"I've created ticket #{ticket_id}. Our team will reach out to you at {message_stripped} shortly."
            )

            await self._store_message(session_id, MessageRole.ASSISTANT, response_with_email, Intent.COMPLAINT, None)
            return ChatResponse(
                message=response_with_email,
                session_id=session_id,
                intent=Intent.COMPLAINT,
                metadata=meta_out,
            )

        # Invalid email format - increment attempt counter
        email_attempts += 1

        # Check if max attempts exceeded
        if email_attempts >= 3:
            # Abandon ticket flow - clear all ticket-related state
            await self._update_session_metadata(session_id, {
                "state": ConversationState.NORMAL_CHAT.value,
                "awaiting_email_for_ticket": False,
                "awaiting_email": False,
                "email_attempts": 0,
                "ticket_flow_active": False,
                "offered_support_options": False
            })

            return ChatResponse(
                message="I understand you prefer not to share your email. I'm here to help with other questions you may have.",
                session_id=session_id,
                intent=Intent.GENERAL,
                metadata={"email_collection_abandoned": True}
            )

        # Still have attempts left - ask again and update counter
        await self._update_session_metadata(session_id, {"email_attempts": email_attempts})

        if email_attempts == 1:
            message_text = "That doesn't look like a valid email address. Please provide your email in the format: name@example.com"
        else:  # email_attempts == 2 (last chance)
            message_text = "I still need a valid email to create your ticket. This is my last request - please provide your email, or let me know if you'd prefer to continue without creating a ticket."

        return ChatResponse(
            message=message_text,
            session_id=session_id,
            intent=Intent.COMPLAINT,
            metadata={"awaiting_email": True, "email_attempts": email_attempts}
        )


    def _validate_input(self, message: str) -> str | None:
        """Validate user input before processing.
        
        Detect empty input, prompt injection, emoji-only messages.
        Returns error message if invalid, None if valid.
        """
        # Strip whitespace
        stripped = message.strip()
        
        # Check empty
        if not stripped:
            return "I didn't receive a message. Could you please type your question or concern?"
        
        # Check emoji-only or non-alphanumeric garbage
        alphanumeric_count = sum(c.isalnum() for c in stripped)
        if alphanumeric_count == 0:
            return "I'm having trouble understanding that. Could you please rephrase your question using words?"
        
        # Check prompt injection patterns
        message_lower = message.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(f"Prompt injection attempt detected: {message[:100]}")
                return (
                    "I'm here to help with customer support questions about orders, "
                    "products, and policies. How can I assist you today?"
                )
        
        return None  # Valid input

    async def _get_session_metadata(self, session_id: str) -> dict:
        """Load session metadata from DB.
        
        Session metadata stores state, customer_email, frustration_level, has_open_ticket.
        STATE MACHINE FIX: Now includes explicit "state" field.
        """
        try:
            result = await self.db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session and session.metadata_:
                return session.metadata_
            # Return default state for new sessions
            return {"state": ConversationState.NORMAL_CHAT.value}
        except Exception as e:
            logger.warning(f"Failed to load session metadata: {e}")
            return {"state": ConversationState.NORMAL_CHAT.value}

    async def _update_session_metadata(self, session_id: str, updates: dict) -> None:
        """Update session metadata in DB.
        
        STEP 6: Merge updates into existing metadata.
        STATE MACHINE FIX: Flush changes to ensure they're persisted.
        """
        try:
            from sqlalchemy.orm import attributes
            
            result = await self.db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                current = session.metadata_ or {}
                current.update(updates)
                session.metadata_ = current
                # Mark the attribute as modified so SQLAlchemy tracks the change
                attributes.flag_modified(session, "metadata_")
                await self.db.flush()
        except Exception as e:
            logger.warning(f"Failed to update session metadata: {e}")

    def _check_frustration_trajectory(self, history: list[Message], current_score: float) -> bool:
        """Check if frustration is escalating across conversation turns.
        
        STEP 6: Returns True if sentiment is getting more negative over time.
        A customer who was neutral for 3 messages then becomes angry is escalating.
        """
        if len(history) < 2:
            return False
        
        # Get sentiment scores from recent history (stored in DB now)
        # For now, we'll use a simple heuristic: if current message is very negative
        # and previous messages existed, consider it escalating
        # TODO: Once sentiment_score is populated in history, calculate actual trajectory
        
        if current_score <= -0.4:
            # Very negative current message suggests escalation
            return True
        
        return False

    async def _route(
        self, intent: Intent, message: str, session_id: str, metadata: dict
    ) -> tuple[str, dict | None]:
        try:
            if intent == Intent.FAQ:
                return await self._handle_faq(message, session_id)
            elif intent == Intent.ORDER_TRACKING:
                return await self._handle_order(message, session_id, metadata)
            elif intent == Intent.COMPLAINT:
                return await self._handle_complaint(message, session_id, metadata)
            elif intent == Intent.GREETING:
                return await self._handle_greeting()
            else:
                return await self._handle_general(message, session_id)
        except Exception as e:
            logger.error(f"Error handling {intent}: {e}", exc_info=True)
            return (
                "I'm sorry, I'm having trouble processing your request right now. "
                "Please try again in a moment.",
                None,
            )

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def _handle_frustration(
        self,
        message: str,
        session_id: str,
        intent: Intent,
        frustration: FrustrationLevel,
        sentiment_score: float,
        metadata: dict,
    ) -> tuple[str, dict | None]:
        """BUG 2 FIX: Ask customer what they want - don't create ticket yet.
        
        This is the ask-before-ticket flow. Give customer options and wait for confirmation.
        STATE MACHINE FIX: Now offers only 2 options and sets SUPPORT_OPTIONS state.
        """
        # Check if customer already has an open ticket
        existing_ticket = await self.ticket_service.check_existing_ticket(session_id)
        if existing_ticket:
            # BUG 6 FIX: Don't claim to add message - just acknowledge existing ticket
            # Lock the session since ticket already exists
            await self._update_session_metadata(session_id, {
                "state": ConversationState.SESSION_LOCKED.value,
                "has_open_ticket": True
            })
            return (
                f"I can see you already have an open ticket #{existing_ticket.id[:8]} for this session. "
                f"Your concern has been noted. Our team will be in touch soon.",
                {"existing_ticket_id": existing_ticket.id},
            )
        
        # STATE MACHINE FIX: Offer only 2 options (removed "human agent" option)
        response = (
            "I'm sorry you're experiencing this. I want to help you in the best way. "
            "Would you like me to:\n\n"
            "1️⃣ Create a support ticket so our team follows up with you\n"
            "2️⃣ Try to resolve this here together\n\n"
            "Please let me know which option you prefer."
        )
        
        # Mark that we've offered options and set state
        # CRITICAL: Store sentiment score and original complaint message for later ticket creation
        await self._update_session_metadata(session_id, {
            "offered_ticket_options": True,
            "state": ConversationState.SUPPORT_OPTIONS.value,
            "original_complaint_message": message,
            "original_sentiment_score": sentiment_score
        })
        
        return response, {
            "frustration_detected": True,
            "frustration_level": frustration.value,
            "sentiment_score": sentiment_score,
            "offered_options": True,
        }

    async def _handle_complaint(
        self,
        message: str,
        session_id: str,
        metadata: dict,
        auto_escalate: bool = False,
    ) -> tuple[str, dict | None]:
        """Handle complaint by creating a support ticket.
        
        BUG 2/4/5/6 FIXES: Only creates ticket after confirmation or auto-escalate.
        Handles email properly. Never claims to add messages to existing tickets.
        """
        # Check for existing open ticket
        existing_ticket = await self.ticket_service.check_existing_ticket(session_id)
        if existing_ticket:
            # BUG 6 FIX: Don't lie about adding message - just acknowledge
            return (
                f"I can see you already have an open ticket #{existing_ticket.id[:8]} for this session. "
                f"Your concern has been noted. Our team will be in touch soon.",
                {"existing_ticket_id": existing_ticket.id},
            )
        
        # Get customer email from metadata
        customer_email = metadata.get("customer_email")
        
        # Extract order ID if mentioned in message
        order_id = None
        ord_match = re.search(r"\bORD-[A-Z0-9]+\b", message, re.IGNORECASE)
        if ord_match:
            order_id = ord_match.group(0).upper()
        
        # Create the ticket
        complaint_req = ComplaintRequest(
            message=message,
            session_id=session_id,
            customer_email=customer_email,
            order_id=order_id,
        )
        ticket = await self.ticket_service.create_ticket(complaint_req, category="complaint")

        # STATE MACHINE FIX: Lock session after ticket creation
        await self._update_session_metadata(session_id, {
            "has_open_ticket": True,
            "state": ConversationState.SESSION_LOCKED.value
        })

        # BUG 5 FIX: Build response with explicit contact info handling
        ticket_id_short = ticket.id[:8]
        has_email = bool(customer_email)
        
        # CRITICAL: Tickets should only be created with valid email
        # This is a safety check - email should already be validated
        if not has_email:
            logger.error(f"Ticket {ticket.id} created without email - this should not happen!")
        
        if ticket.status.value == "escalated":
            response = (
                f"I've created ticket #{ticket_id_short} with URGENT priority. "
                f"Due to the urgency of your concern, this has been automatically escalated "
                f"to a senior support representative. They will reach out to you at {customer_email} shortly."
            )
        else:
            response = (
                f"I've created ticket #{ticket_id_short} with {ticket.priority.value} priority. "
                f"Our team will reach out to you at {customer_email} to help resolve this."
            )
        
        return response, {
            "ticket_id": ticket.id,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "has_email": has_email,
        }
    async def _handle_faq(self, message: str, session_id: str) -> tuple[str, dict | None]:
        results = await self.vector_store.search(message, top_k=3)

        if not results or results[0].score < 0.55:
            # No good FAQ match — use a strict no-hallucination prompt
            history = await self._load_history(session_id)
            response = await self._llm_generate(
                user_message=message,
                extra_context=FAQ_NO_MATCH_PROMPT,
                history=history,
            )
            return response, {"faq_match": False}

        # Build context from matched FAQ entries
        context = "\n\n".join(
            f"Q: {r.entry.question}\nA: {r.entry.answer}" for r in results
        )
        history = await self._load_history(session_id)
        response = await self._llm_generate(
            user_message=message,
            extra_context=FAQ_GROUNDED_PROMPT.format(context=context),
            history=history,
        )
        return response, {"faq_sources": [r.entry.question for r in results], "faq_match": True}

    async def _handle_order(
        self, message: str, session_id: str, metadata: dict
    ) -> tuple[str, dict | None]:
        """Handle order tracking with STRICT context rules.

        CRITICAL SAFETY RULES:
        1. NEVER pull order IDs from distant conversation history
        2. NEVER guess or invent order IDs
        3. Only use order ID if explicitly provided or confirmed as active context
        4. Clear stale order context when user switches topics
        """
        order_id: str | None = None
        message_lower = message.lower()

        # Rule 1: Extract from CURRENT message only (Priority 1)
        ord_match = re.search(r"\bORD-[A-Z0-9]+\b", message, re.IGNORECASE)
        if ord_match:
            order_id = ord_match.group(0).upper()

        # Rule 2: Check if bot is awaiting order ID from previous turn (Priority 2)
        if not order_id and metadata.get("awaiting_order_id"):
            # User's message might be just the numeric ID without ORD- prefix
            context_match = re.search(r"\b(\d{4,6})\b", message)
            if context_match:
                order_id = f"ORD-{context_match.group(1)}"

        # Rule 3: Check active confirmed order context (Priority 3)
        if not order_id and metadata.get("active_order_confirmed"):
            # User is referring to previously confirmed order
            # Only if message contains order-related keywords
            if any(keyword in message_lower for keyword in ["my order", "the order", "it", "status", "where", "when", "tracking"]):
                order_id = metadata.get("active_order_id")

        # If we have an order ID, look it up
        if order_id:
            try:
                order = await self.order_service.get_by_id(order_id)

                # Mark this order as active context
                await self._update_session_metadata(session_id, {
                    "awaiting_order_id": False,
                    "active_order_id": order_id,
                    "active_order_confirmed": True,
                    "state": ConversationState.NORMAL_CHAT.value
                })

                # Handle cancelled orders
                if order.status.value == "cancelled":
                    history = await self._load_history(session_id)
                    response = await self._llm_generate(
                        user_message=message,
                        extra_context=(
                            f"Order {order.id} was cancelled. "
                            f"Items: {', '.join(f'{i.name} (x{i.quantity})' for i in order.items)}. "
                            f"Total: ${order.total:.2f}. "
                            "Explain that the order was cancelled. If they ask why, offer to create a support ticket."
                        ),
                        history=history,
                    )
                    return response, {"order_id": order_id, "status": "cancelled"}

                # Handle orders without tracking numbers
                if order.tracking_number is None and order.status.value in ["processing", "pending", "confirmed"]:
                    history = await self._load_history(session_id)
                    response = await self._llm_generate(
                        user_message=message,
                        extra_context=(
                            f"Order {order.id} is currently {order.status.value} and hasn't shipped yet, "
                            f"so no tracking number is available. "
                            f"Estimated delivery: {order.estimated_delivery or 'TBD'}. "
                            f"Items: {', '.join(f'{i.name} (x{i.quantity})' for i in order.items)}. "
                            "Explain that the order is being prepared and tracking will be available once it ships."
                        ),
                        history=history,
                    )
                    return response, {"order_id": order_id, "status": order.status.value, "tracking_available": False}

                # Normal order status summary
                summary = await self.order_service.get_status_summary(order_id)
                return summary, {"order_id": order_id, "status": order.status.value}

            except Exception as e:
                logger.warning(f"Order lookup failed for {order_id}: {e}")
                # Clear the failed order from context
                await self._update_session_metadata(session_id, {
                    "awaiting_order_id": False,
                    "active_order_id": None,
                    "active_order_confirmed": False
                })
                return (
                    f"I couldn't find order {order_id} in our system. "
                    "Please double-check the order ID and try again.",
                    {"order_id": order_id, "found": False},
                )

        # Rule 4: No order ID found - ask for it
        # Set awaiting state
        await self._update_session_metadata(session_id, {
            "awaiting_order_id": True,
            "state": ConversationState.AWAITING_ORDER_ID.value
        })

        return (
            "I'd be happy to help you track your order! Could you please provide your order ID? It usually looks like ORD-1001.",
            None,
        )


    async def _handle_greeting(self) -> tuple[str, dict | None]:
        return (
            "Hello! Welcome to our customer support. I'm here to help you with:\n\n"
            "- **FAQs** — Questions about products, shipping, returns, and more\n"
            "- **Order Tracking** — Check the status of your order\n"
            "- **Support** — File a complaint or get help with an issue\n\n"
            "How can I assist you today?",
            None,
        )

    async def _handle_general(self, message: str, session_id: str) -> tuple[str, dict | None]:
        history = await self._load_history(session_id)
        response = await self._llm_generate(message, history=history)
        return response, None

    # ── LLM generation ───────────────────────────────────────────────────────

    async def _llm_generate(
        self,
        user_message: str,
        extra_context: str = "",
        history: list[Message] | None = None,
    ) -> str:
        """Generate an LLM response with optional conversation history.

        FIX 2: Conversation history is now loaded from DB and passed here.
        Previously every call was zero-shot — the bot had no memory of what
        was said earlier in the same session.
        """
        system = SYSTEM_PROMPT
        if extra_context:
            system += f"\n\n{extra_context}"

        # Build message list: history + current user message
        messages: list[Message] = []
        if history:
            messages.extend(history)
        messages.append(Message(role=MessageRole.USER, content=user_message))

        return await self.llm.generate(messages, system_prompt=system)

    # ── History loading ───────────────────────────────────────────────────────

    async def _load_history(self, session_id: str) -> list[Message]:
        """Load recent conversation history for this session from the DB.

        FIX 2: This method was completely missing in the original.
        ChatMessage rows were written to DB but never read back, making
        every LLM call zero-shot despite the session infrastructure existing.

        Returns the last CONVERSATION_HISTORY_LIMIT messages in
        chronological order, ready to pass directly to the LLM.
        """
        try:
            result = await self.db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(CONVERSATION_HISTORY_LIMIT)
            )
            rows = result.scalars().all()
            # Rows are newest-first from the query; reverse for chronological order
            rows = list(reversed(rows))
            return [
                Message(
                    role=MessageRole(row.role),
                    content=row.content,
                )
                for row in rows
            ]
        except Exception as e:
            # History is best-effort — never break the main flow if it fails
            logger.warning(f"Failed to load conversation history for {session_id}: {e}")
            return []

    # ── Session / message persistence ────────────────────────────────────────

    async def _ensure_session(self, session_id: str, channel: str) -> None:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        if not result.scalar_one_or_none():
            # STATE MACHINE FIX: Initialize session with default state
            session = ChatSession(
                id=session_id,
                channel=channel,
                metadata_={"state": ConversationState.NORMAL_CHAT.value}
            )
            self.db.add(session)
            await self.db.flush()

    async def _store_message(
        self, session_id: str, role: MessageRole, content: str, intent: Intent, sentiment_score: float | None = None
    ) -> None:
        """Store message in DB with optional sentiment score.
        
        STEP 6: Now stores sentiment_score for frustration tracking across turns.
        """
        msg = ChatMessage(
            session_id=session_id,
            role=role.value,
            content=content,
            intent=intent.value,
            sentiment_score=sentiment_score,
        )
        self.db.add(msg)
        await self.db.flush()