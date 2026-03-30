# Codex Implementation Prompt v3 - Production LangGraph Customer Support Chatbot (Repo-Aligned, Hybrid, State-Aware)

---

## Context

You are refactoring an existing production customer support chatbot from a monolithic
`ChatOrchestrator` pattern into a LangGraph-based multi-node graph architecture.

Current stack and constraints:

- Backend: FastAPI + SQLAlchemy async + SQLite
- Frontend: Next.js 15 + React 19
- LLM access: provider abstraction already exists in `backend/app/services/llm/`
- FAQ retrieval: existing FAISS / vector store stack
- Sentiment: existing VADER-based sentiment service
- Existing behavior is protected by tests and must be preserved unless explicitly improved here

This is a backend orchestration refactor, not a product redesign.

Non-negotiable migration principles:

- Do not break API compatibility with the current frontend
- Do not delete `backend/app/services/chat/orchestrator.py` until graph behavior reaches parity
- Do not remove DB metadata compatibility during the first migration phase
- Prefer deterministic routing for operational flows and use LLM only where ambiguity or wording benefit exists

---

## Objective

Replace `backend/app/services/chat/orchestrator.py` and surrounding orchestration wiring with a LangGraph graph that is:

1. Hybrid routed: deterministic first, LLM only for ambiguity
2. State-aware: active flows continue before generic routing
3. Production-safe: prompt injection blocking, explicit parsing, deterministic guardrails
4. Repo-aligned: use the real service interfaces, models, and response contracts in this codebase
5. Backward-compatible during migration: read and write legacy session metadata until cutover is complete

The new graph must handle:

- input validation and injection detection
- context loading from DB
- active-flow continuation routing
- supervisor routing
- FAQ, Order, Complaint, Greeting, and General behavior
- complaint subflow: acknowledge -> support options -> email collection / confirmation -> ticket creation or safe cancellation
- deterministic guardrail checking
- persistence back to DB-compatible metadata and messages

---

## Architecture Decision

This project should not use "LLM-first everywhere".

Use deterministic logic for:

- input rejection
- prompt injection detection
- order ID extraction and normalization
- email extraction and validation
- explicit ticket creation and inquiry triggers
- active-flow continuation
- duplicate ticket prevention
- flow transitions
- final response safety validation

Use the LLM for:

- ambiguous intent classification
- FAQ grounded answer phrasing
- general/greeting responses
- optional response polish when a deterministic draft already exists

If a deterministic production behavior conflicts with a generic LLM-first idea, preserve the deterministic production behavior.

---

## Real Repo Contracts To Honor

These are mandatory:

1. The current LLM factory is `app.services.llm.factory.get_llm_provider()`
2. `get_llm_provider()` is not async
3. The provider interface uses:
   - `generate(messages, system_prompt=...)`
   - `stream(...)`
   - `classify(text, categories)`
4. LLM calls should use real `Message` / `MessageRole` objects from `app.models.chat`
5. The FAQ vector store comes from `app.services.faq.factory.get_vector_store()`
6. FAQ search results use:
   - `result.entry.question`
   - `result.entry.answer`
   - `result.score`
7. `OrderService` requires `db` in the constructor
8. `TicketService` requires `db` in the constructor
9. `analyze_sentiment()` returns a `SentimentResult` model with:
   - `.score`
   - `.label`
   - `.confidence`
10. API responses must remain compatible with the current frontend:
   - keep `message`
   - keep `session_id`
   - keep `intent`
   - keep `metadata`
   - keep `timestamp`
11. WebSocket shape should remain compatible with the current backend endpoint

---

## Tech Stack

Keep:

- Python 3.11+
- FastAPI
- SQLAlchemy async + aiosqlite
- FAISS + sentence-transformers
- Pydantic v2
- VADER Sentiment
- existing provider abstraction for Gemini / OpenAI / Anthropic

Add:

```txt
langgraph>=0.2.0
langchain-core>=0.3.0
langsmith>=0.1.0
```

Optional:

- LangSmith tracing

Do not require a database schema change in phase 1 unless a truly new field is intentionally introduced.

---

## Phase-1 Behavioral Preservation Rules

These current behaviors must remain true after the graph refactor:

### Order flow

- `track my order` asks for an order ID when none is known
- a plain numeric follow-up like `1001` is accepted while awaiting an ID
- `ORD-[A-Z0-9]+` extraction remains supported
- order flow can be interrupted by a complaint or FAQ topic switch
- active order context can be reused for relevant follow-up questions

### Complaint and ticket flow

- complaint handling remains stronger than simple intent classification
- explicit ticket creation requests go straight into complaint/ticket flow
- explicit ticket inquiry requests go to complaint flow, not order flow
- duplicate tickets in the same session are prevented
- email collection supports:
  - direct email
  - email inside a sentence
  - confirmation
  - change
  - skip
  - natural-language refusal
- topic switching during email collection is allowed
- a ticket may still be created without email when the user skips or refuses
- do not silently create multiple tickets

### FAQ flow

- answers must remain grounded in retrieved FAQ content
- if no good FAQ match exists, do not invent a policy answer

### General flow

- greetings and acknowledgements should not be over-forced into complaint or order
- out-of-scope questions should be politely redirected back to support scope

---

## Graph Structure

Create:

```txt
backend/app/services/chat/
  graph_state.py
  graph.py
  routing.py
  nodes/
    __init__.py
    llm_utils.py
    input_validator.py
    context_loader.py
    active_flow_router.py
    supervisor_router.py
    faq_node.py
    order_node.py
    complaint_node.py
    general_node.py
    response_composer.py
    guardrail_checker.py
    persist_node.py
```

Critical routing order:

```txt
input_validator
  -> context_loader
  -> active_flow_router
     -> order_node
     -> complaint_node
     -> supervisor_router
     -> END
  -> FAQ / Order / Complaint / General nodes
  -> response_composer
  -> guardrail_checker
  -> persist_node
  -> END
```

Do not route directly from `context_loader` to `supervisor_router`.

---

## Graph State

Create `backend/app/services/chat/graph_state.py`

```python
from typing import TypedDict, Optional, Literal, List


class ChatHistoryItem(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str
    intent: Optional[str]
    sentiment_score: Optional[float]


class TicketData(TypedDict, total=False):
    ticket_id: str
    short_id: str
    priority: Literal["urgent", "high", "medium", "low"]
    status: Literal["open", "escalated", "resolved", "closed"]
    category: str
    customer_email: Optional[str]


class GraphState(TypedDict, total=False):
    session_id: str
    channel: str
    user_message: str

    current_datetime: str
    current_date: str

    history: List[ChatHistoryItem]
    session_metadata: dict

    intent: Optional[Literal["faq", "order_tracking", "complaint", "general", "greeting"]]
    sub_intent: Optional[
        Literal[
            "none",
            "ticket_create",
            "ticket_inquiry",
            "complaint_message",
            "flow_continue",
            "acknowledgement",
            "capability_query",
        ]
    ]
    routing_confidence: Optional[float]
    router_reason: Optional[str]
    is_topic_switch: bool
    active_flow_route: Optional[Literal["supervisor", "order", "complaint", "end"]]

    flow_state: Literal[
        "NORMAL_CHAT",
        "AWAITING_ORDER_ID",
        "SUPPORT_OPTIONS",
        "EMAIL_COLLECTION",
        "EMAIL_CONFIRM",
        "COMPLAINT_GATHER",
    ]

    awaiting_order_id: bool
    active_order_id: Optional[str]
    active_order_confirmed: bool
    active_order_data: Optional[dict]

    complaint_description: Optional[str]
    complaint_category: Optional[str]
    complaint_sentiment_score: Optional[float]
    complaint_sentiment_label: Optional[str]
    complaint_sentiment_confidence: Optional[float]
    complaint_frustration_level: Optional[int]
    offered_ticket: bool

    has_open_ticket: bool
    ticket_id: Optional[str]
    ticket_data: Optional[TicketData]

    customer_email: Optional[str]
    pending_email: Optional[str]
    email_confirmed: bool
    email_attempts: int
    last_provided_email: Optional[str]

    draft_response: Optional[str]
    final_response: Optional[str]
    quick_replies: List[str]
    response_metadata: dict

    error: Optional[str]
    should_end: bool
    guardrail_passed: bool
```

Rules:

- nodes return partial dicts only
- runtime graph state is the control source for the current request
- `session_metadata` is still required for compatibility during migration
- preserve support for legacy metadata keys already used by the current orchestrator

---

## Shared LLM Utilities

Create `backend/app/services/chat/nodes/llm_utils.py`

```python
import json
import logging
import re

from app.models.chat import Message, MessageRole
from app.services.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)


async def llm_json_decision(
    system_prompt: str,
    user_content: str,
    fallback: dict,
    retries: int = 2,
) -> dict:
    llm = get_llm_provider()

    for attempt in range(retries):
        try:
            raw = await llm.generate(
                messages=[Message(role=MessageRole.USER, content=user_content)],
                system_prompt=system_prompt,
                temperature=0.0,
            )
            cleaned = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
            return json.loads(cleaned)
        except Exception as exc:
            logger.warning("LLM JSON parse attempt %s failed: %s", attempt + 1, exc)
            if attempt == retries - 1:
                logger.error("Falling back to deterministic JSON fallback: %s", fallback)
                return fallback

    return fallback


def format_history_for_prompt(history: list, max_turns: int = 8) -> str:
    if not history:
        return "[No prior conversation]"

    lines = []
    for i, msg in enumerate(history[-max_turns:], 1):
        role = "Customer" if msg.get("role") == "user" else "Support Bot"
        lines.append(f"[{i}] {role}: {msg.get('content', '')}")
    return "\n".join(lines)


def format_state_context(state: dict) -> str:
    parts = []

    if state.get("flow_state") and state["flow_state"] != "NORMAL_CHAT":
        parts.append(f"Flow state: {state['flow_state']}")
    if state.get("active_order_id"):
        parts.append(f"Active order: {state['active_order_id']}")
    if state.get("complaint_description"):
        parts.append(f"Complaint on file: {state['complaint_description'][:160]}")
    if state.get("complaint_category"):
        parts.append(f"Complaint category: {state['complaint_category']}")
    if state.get("customer_email"):
        parts.append(f"Customer email known: {state['customer_email']}")
    if state.get("has_open_ticket"):
        parts.append("Open ticket already exists")

    return "\n".join(parts) if parts else "No special session state."
```

Do not use raw dicts as messages when calling the provider.

---

## Graph Definition

Create `backend/app/services/chat/graph.py`

```python
from langgraph.graph import END, StateGraph

from .graph_state import GraphState
from .nodes.active_flow_router import active_flow_router_node
from .nodes.complaint_node import complaint_node
from .nodes.context_loader import context_loader_node
from .nodes.faq_node import faq_node
from .nodes.general_node import general_node
from .nodes.guardrail_checker import guardrail_checker_node
from .nodes.input_validator import input_validator_node
from .nodes.order_node import order_node
from .nodes.persist_node import persist_node
from .nodes.response_composer import response_composer_node
from .nodes.supervisor_router import supervisor_router_node
from .routing import (
    route_after_active_flow_router,
    route_after_guardrail,
    route_after_supervisor,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_validator", input_validator_node)
    graph.add_node("context_loader", context_loader_node)
    graph.add_node("active_flow_router", active_flow_router_node)
    graph.add_node("supervisor_router", supervisor_router_node)
    graph.add_node("faq_node", faq_node)
    graph.add_node("order_node", order_node)
    graph.add_node("complaint_node", complaint_node)
    graph.add_node("general_node", general_node)
    graph.add_node("response_composer", response_composer_node)
    graph.add_node("guardrail_checker", guardrail_checker_node)
    graph.add_node("persist_node", persist_node)

    graph.set_entry_point("input_validator")
    graph.add_edge("input_validator", "context_loader")
    graph.add_edge("context_loader", "active_flow_router")

    graph.add_conditional_edges(
        "active_flow_router",
        route_after_active_flow_router,
        {
            "order": "order_node",
            "complaint": "complaint_node",
            "supervisor": "supervisor_router",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "supervisor_router",
        route_after_supervisor,
        {
            "faq": "faq_node",
            "order": "order_node",
            "complaint": "complaint_node",
            "general": "general_node",
            "end": END,
        },
    )

    for node_name in ["faq_node", "order_node", "complaint_node", "general_node"]:
        graph.add_edge(node_name, "response_composer")

    graph.add_edge("response_composer", "guardrail_checker")
    graph.add_conditional_edges(
        "guardrail_checker",
        route_after_guardrail,
        {
            "persist": "persist_node",
            "recompose": "response_composer",
        },
    )
    graph.add_edge("persist_node", END)
    return graph.compile()
```

Do not hard-code a global graph in a way that prevents tests from overriding dependencies.
If needed, expose a builder and inject dependencies through runtime state or service wrappers.

---

## Routing Functions

Create `backend/app/services/chat/routing.py`

```python
from .graph_state import GraphState


def route_after_active_flow_router(state: GraphState) -> str:
    if state.get("should_end"):
        return "end"
    return state.get("active_flow_route", "supervisor")


def route_after_supervisor(state: GraphState) -> str:
    if state.get("should_end") or state.get("error"):
        return "end"

    intent = state.get("intent") or "general"
    mapping = {
        "faq": "faq",
        "order_tracking": "order",
        "complaint": "complaint",
        "greeting": "general",
        "general": "general",
    }
    return mapping.get(intent, "general")


def route_after_guardrail(state: GraphState) -> str:
    if state.get("guardrail_passed", True):
        return "persist"
    return "persist"
```

---

## Node Responsibilities

### input_validator_node

Keep this mostly deterministic.

Responsibilities:

- reject empty input
- reject clearly invalid payloads
- block obvious prompt injection
- return a safe support-scoped response when blocked

Do not make this fully LLM-first.

### context_loader_node

Responsibilities:

- load session metadata from DB
- load recent chat history
- inject `current_date` and `current_datetime`
- normalize legacy metadata into runtime graph state
- reconstruct fields like:
  - `flow_state`
  - `awaiting_order_id`
  - `pending_email`
  - `email_attempts`
  - `customer_email`
  - `ticket_id`
  - `has_open_ticket`
  - `active_order_id`

History should remain short and useful, similar to the current orchestrator behavior.

### active_flow_router_node

This node is critical.

It must handle active flows before generic routing.

If `flow_state` is one of:

- `EMAIL_COLLECTION`
- `EMAIL_CONFIRM`
- `SUPPORT_OPTIONS`
- `COMPLAINT_GATHER`

then prefer routing to `complaint_node` for obvious continuation messages:

- yes
- no
- skip
- change
- email address
- short responses continuing the ticket/email flow

If `flow_state == "AWAITING_ORDER_ID"` then prefer routing to `order_node` when:

- message contains `ORD-[A-Z0-9]+`
- message is a plain 4-6 digit number
- message is an order lookup continuation

Allow a clear topic switch back to `supervisor_router`.

Do not waste LLM calls on obvious continuations.

### supervisor_router_node

Use hybrid routing:

1. deterministic pre-routing
2. LLM disambiguation only when necessary

Deterministic pre-routing should catch:

- explicit order IDs
- explicit ticket inquiry phrases
- explicit ticket creation phrases
- obvious greetings
- obvious acknowledgements
- obvious complaint phrases
- clear FAQ/policy questions

Conflict rules:

- order words plus strong frustration or legal threat -> complaint
- polite informational refund/process questions may still be FAQ
- short acknowledgements should not be forced into complaint
- explicit ticket inquiry/create must go to complaint with an appropriate sub-intent

Use the current project's routing philosophy as the source of truth, not a generic prompt taxonomy.

### faq_node

Use the real FAQ vector store factory.

Behavior:

1. search the vector store deterministically
2. if there is no credible match, return a safe "I don't have that confirmed" response
3. optionally use the LLM only to phrase a grounded answer from retrieved FAQ entries
4. never invent policy details outside retrieved FAQ content

Do not rely on LLM-only relevance without respecting retrieval scores.

### order_node

This node must remain operationally deterministic.

Rules:

- extract and normalize order IDs via regex first
- continue to support:
  - `ORD-1001`
  - `ord-1001`
  - numeric follow-up like `1001` while awaiting an ID
- reuse `active_order_id` when safe
- instantiate `OrderService(db)`
- write back:
  - `active_order_id`
  - `active_order_confirmed`
  - `customer_email` when learned from order data

Use LLM only for optional wording polish, not for operational truth.

### complaint_node

This node owns:

- complaint acknowledgement
- support option offering
- ticket creation triggers
- ticket inquiry handling
- email collection
- email confirmation
- skip/refusal behavior
- duplicate ticket prevention

Important repo-aligned rules:

- explicit ticket inquiry stays deterministic where possible
- `TicketService(db)` must be used
- `skip` and natural-language refusal may still create a ticket without email, matching current behavior
- do not silently create repeated tickets
- topic switch during email collection must still work
- do not add fake support contact details unless they are confirmed product data

Use LLM only where it helps:

- classifying ambiguous complaint text
- phrasing empathetic but concise replies
- interpreting rich free-form complaint descriptions

### general_node

Handles:

- greetings
- capability intros
- acknowledgements
- out-of-scope messages
- vague messages

Keep this scoped and concise.
Do not answer random general-knowledge questions as though this is a general assistant.

### response_composer_node

Responsibilities:

- choose `final_response` from:
  - existing `final_response`
  - otherwise `draft_response`
- optionally perform one bounded polish pass
- preserve order IDs, ticket IDs, email addresses, and instructions exactly
- avoid adding new claims

### guardrail_checker_node

Keep deterministic.

Responsibilities:

- ensure the output is not empty
- ensure required operational terms remain present
- block internal prompt leakage
- block unsupported fabricated contacts or promises
- fall back safely if needed

### persist_node

Responsibilities:

- write assistant message to `chat_messages`
- write compatibility metadata back to `chat_sessions.metadata`
- keep old and new state representations aligned during migration
- preserve current API return shape

Do not change the frontend response contract in phase 1.

---

## FastAPI Integration

Refactor `backend/app/routers/chat.py` carefully.

Requirements:

- keep the current request/response models if possible
- keep `message` in the response, not `response`
- keep `timestamp`
- preserve existing dependency injection for tests

Preferred approach:

- add a graph-backed service path behind the current router contract
- do not immediately hard-wire an untestable singleton graph

The REST and WebSocket routes should both remain compatible.

---

## Database Guidance

Phase 1 should not require a schema rewrite.

Already available and reusable:

- `chat_sessions.metadata`
- `chat_messages.intent`
- `chat_messages.sentiment_score`
- `tickets.category`
- `tickets.customer_email`
- `tickets.order_id`

Only add new DB fields if they are truly required and reflected in product requirements.
Do not add speculative fields just because a generic prompt suggested them.

---

## Testing Requirements

The new graph path must preserve current behavior parity.

At minimum, cover these scenarios:

1. empty message is rejected
2. obvious prompt injection is blocked
3. `track my order` asks for order ID
4. `1001` in active order flow continues to the order path
5. strong complaint plus order words routes to complaint, not order
6. `What is your return policy?` routes to FAQ
7. `hi` results in greeting/general behavior
8. explicit ticket creation request goes to complaint flow
9. explicit ticket inquiry request goes to complaint flow
10. email collection flow continues without generic rerouting first
11. topic switch during email collection is allowed
12. duplicate ticket prevention still works
13. ticket can still be created when user skips email, matching current product behavior
14. frontend response still uses `message` and `timestamp`
15. legacy `session_metadata` compatibility persists across requests
16. REST and WebSocket behavior remain compatible

Also preserve or port the existing multi-turn regression tests before deleting the orchestrator.

---

## Migration Strategy

1. Build the graph path alongside the current orchestrator
2. Keep DB metadata compatibility
3. Reconstruct graph state from legacy metadata in `context_loader`
4. Persist compatibility metadata in `persist_node`
5. Run parity tests
6. Cut over only after parity is acceptable
7. Remove old orchestrator only after stable rollout

Do not do a big-bang rewrite with no compatibility layer.

---

## Final Build Instruction

Implement the LangGraph refactor using this v3 design.

Priority order:

1. Hybrid routing, not LLM-first everywhere
2. Insert `active_flow_router` before `supervisor_router`
3. Preserve current order, complaint, ticket, and email flow correctness
4. Use the real repo interfaces and message models
5. Keep API compatibility with the current frontend
6. Preserve compatibility metadata until migration is complete
7. Achieve test parity before removing the old orchestrator

If there is a conflict between a generic architecture idea and the current repo's protected business behavior, preserve the current protected business behavior.
