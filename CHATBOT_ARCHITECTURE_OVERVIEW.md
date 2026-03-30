# Support Chatbot Architecture Overview

## 1. What This Project Currently Does

This project is a customer-support chatbot system with:

- FAQ answering using vector search plus LLM-generated responses
- order tracking using seeded order data stored in SQLite
- complaint intake with sentiment analysis, ticket creation, and escalation
- a standalone chat page in Next.js
- an embeddable website widget
- REST and WebSocket chat endpoints on the backend

The current product is built as a full-stack application:

- `backend/` runs the API, orchestration logic, data storage, FAQ retrieval, and ticketing
- `frontend/` provides the landing page, full chat UI, and embeddable widget entrypoint

The main focus of the system is not just "chat", but routing each user message into the correct support workflow:

- FAQ workflow
- order workflow
- complaint/ticket workflow
- general/greeting workflow


## 2. High-Level Architecture

```text
User
  -> Next.js chat UI or embeddable widget
  -> POST /api/chat
  -> FastAPI chat router
  -> ChatOrchestrator
  -> intent detection + state machine + domain services
  -> FAQ store / orders DB / ticket service / LLM provider
  -> response persisted to chat session
  -> response returned to UI
```

Current major layers:

1. Frontend channel layer
   - standalone chat page
   - floating widget
   - embeddable `widget.js`

2. API layer
   - FastAPI routers for chat, FAQ, orders, complaints, health, widget config

3. Orchestration layer
   - `ChatOrchestrator` is the core brain of the app
   - handles session state, routing, persistence, and flow transitions

4. Domain services
   - FAQ vector search
   - order lookup
   - complaint sentiment and ticket creation
   - pluggable LLM providers

5. Persistence layer
   - SQLite via SQLAlchemy async engine
   - FAISS persisted index for FAQ embeddings


## 3. Tech Stack

### Backend

- Python
- FastAPI
- Pydantic v2
- SQLAlchemy async
- `aiosqlite`
- FAISS
- `sentence-transformers`
- VADER Sentiment
- Google Gemini / OpenAI / Anthropic via provider abstraction
- pytest + pytest-asyncio

### Frontend

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Sonner for toast notifications
- DOMPurify for sanitizing chat message HTML

### Infrastructure / Packaging

- Dockerfiles for backend and frontend
- environment-driven configuration through `.env`
- frontend rewrite proxy to backend API


## 4. Repository Structure

```text
support-chatbot/
  backend/
    app/
      config.py
      dependencies.py
      db/
      models/
      routers/
      services/
        chat/
        complaints/
        faq/
        llm/
        orders/
    data/faqs.json
    tests/
    main.py
    requirements.txt
    support.db

  frontend/
    public/widget.js
    src/
      app/
      components/
      lib/
    package.json
    next.config.ts
```


## 5. Backend Runtime Flow

### Startup Flow

When the backend starts:

1. FastAPI app is created in `backend/main.py`
2. CORS is configured
3. database tables are created
4. FAQ vector store is initialized from the configured provider
5. FAQs are seeded from `backend/data/faqs.json` if the store is empty
6. sample orders are seeded into SQLite if the orders table is empty

This means the app is designed to be usable immediately in local development with demo data.

### Registered Routers

- `/api/health`
- `/api/chat`
- `/api/chat/ws`
- `/api/faq`
- `/api/orders`
- `/api/complaints`
- `/api/widget/config`


## 6. Core Chatbot Brain

The main chatbot logic lives in:

- `backend/app/services/chat/orchestrator.py`

This file is the most important part of the project. It combines:

- session management
- intent routing
- state-machine transitions
- message persistence
- ticket handling
- order lookup flow
- FAQ retrieval flow
- LLM response generation
- guarded LLM-based reply polishing

### Core responsibilities of `ChatOrchestrator`

- create or reuse a chat session
- validate input
- load and normalize session metadata
- detect if the user is in a special state like:
  - awaiting order id
  - choosing support options
  - email collection
- classify the message intent
- route to the right handler
- store user and assistant messages in the database


## 7. Chat Session and Stored Data

### Database Tables

The backend stores these main entities:

#### `orders`

Contains:

- order id
- customer email
- status
- items
- total
- tracking number
- estimated delivery

#### `tickets`

Contains:

- ticket id
- session id
- category
- description
- sentiment score and label
- priority
- status
- assigned user
- customer email
- order id

#### `chat_sessions`

Contains:

- session id
- channel
- JSON metadata
- created timestamp

#### `chat_messages`

Contains:

- session id
- role
- content
- detected intent
- sentiment score
- created timestamp

### Session Metadata Role

The chatbot state is primarily stored inside `chat_sessions.metadata`.

Important metadata keys used by the current implementation:

- `state`
- `awaiting_order_id`
- `active_order_id`
- `active_order_confirmed`
- `offered_ticket_options`
- `awaiting_email`
- `awaiting_email_confirmation`
- `pending_email`
- `email_attempts`
- `original_complaint_message`
- `original_sentiment_score`
- `last_provided_email`
- `customer_email`
- `has_open_ticket`
- `ticket_id`


## 8. Conversation States Used Now

The current orchestrator defines these explicit states:

- `NORMAL_CHAT`
- `AWAITING_ORDER_ID`
- `SUPPORT_OPTIONS`
- `EMAIL_COLLECTION`

### What each state means

#### `NORMAL_CHAT`

Default state. The user is not in the middle of a forced follow-up flow.

#### `AWAITING_ORDER_ID`

The user asked about an order, but no valid order id was given yet.

#### `SUPPORT_OPTIONS`

The user appears frustrated or asked about tickets, so the bot is waiting for:

- `1` = create a support ticket
- `2` = continue resolving in chat

#### `EMAIL_COLLECTION`

The bot is trying to collect an email for ticket follow-up, or confirm a detected email.


## 9. End-to-End Chat Request Lifecycle

For a normal UI message, the current flow is:

1. user types a message in `ChatWindow`
2. frontend calls `sendMessage()` in `frontend/src/lib/api.ts`
3. request goes to `POST /api/chat`
4. `chat.py` creates `ChatOrchestrator`
5. orchestrator validates and routes the message
6. orchestrator stores chat session updates and chat messages
7. backend returns `ChatResponse`
8. frontend appends the assistant message to local UI state
9. frontend keeps the returned `session_id` for later messages

The session id is what allows the backend to preserve chat flow across turns.


## 10. Intent Detection Logic

Intent detection is hybrid:

- first: keyword/rule-based classification
- fallback: LLM classification only for ambiguous cases

### Supported intents

- `greeting`
- `faq`
- `order_tracking`
- `complaint`
- `general`

### Rule-based classification includes

- regex patterns for greetings
- regex patterns for explicit order lookup
- regex patterns for complaint/frustration language
- regex patterns for FAQ/policy/how-to questions
- frustration level detection
- sarcasm detection
- polite-tone detection

### Important classification rules

- strong frustration or sarcasm pushes the message to `complaint`
- polite transactional wording can keep a message in `faq`
- order tracking is used only for explicit/actionable order lookup
- vague or unclear text can fall back to `general`
- for some ambiguous `general` messages, the LLM is asked to classify

### LLM intent classification is intentionally limited

The orchestrator avoids LLM disambiguation for short vague follow-ups like:

- "thanks"
- "never mind"
- "answer my question"

That prevents the LLM from over-forcing messages into order or complaint flows.


## 11. Input Guardrails and Validation

Before routing, the chatbot performs input validation:

- rejects empty/whitespace-only messages
- rejects messages with no alphanumeric content
- blocks prompt-injection style phrases such as:
  - "ignore previous instructions"
  - "reveal your system prompt"
  - fake `system:` style content

If prompt-injection-like content is detected, the bot returns a safe support-oriented response instead of following the attack.


## 12. FAQ Flow

### How FAQ answers work now

1. intent is classified as `faq`
2. FAQ vector store searches top 3 similar entries
3. if there is no strong result, the LLM is told not to guess and to direct the user to support
4. if there are matches, the matched FAQ entries are inserted into a grounded prompt
5. the LLM writes the final natural-language answer using only the FAQ context

### FAQ retrieval details

- default vector store provider: FAISS
- embedding model: `all-MiniLM-L6-v2`
- FAQ similarity threshold in the orchestrator: `0.55`
- FAQ data is seeded from `backend/data/faqs.json`

### Important implementation detail

The FAISS store embeds only the FAQ questions for search, not `question + answer`, which improves matching against user query phrasing.

### Current FAQ behavior

- semantic FAQ search
- top match gating
- source questions returned in response metadata when matched
- graceful fallback when no answer is grounded


## 13. Order Tracking Flow

### Order flow summary

The order flow is designed to be helpful without overcommitting:

1. detect explicit order-tracking intent
2. if no order id is present, ask for one
3. accept formats like:
   - `ORD-1001`
   - plain numeric follow-up like `1001` while already awaiting an id
4. look up the order from SQLite
5. generate a response based on order state

### Current order states handled

- cancelled
- processing / pending / confirmed without tracking number
- all normal summary states via `OrderService.get_status_summary()`

### Current order flow behaviors

- if the user gave an order id, it is stored as `active_order_id`
- follow-up order questions can reuse the active order id
- if the user switches topics while the bot is waiting for an order id, the bot exits the order flow
- strong negative/legal-threat messages override order flow and go to complaint flow

### If order is not found

The bot responds with a deterministic message telling the user:

- the order id was not found
- they should re-check the format
- example format: `ORD-1001`


## 14. Complaint and Ticket Flow

This is the richest conversation flow in the project.

### Step 1: Complaint detection

A message enters complaint handling when:

- intent classifier marks it as `complaint`
- or the user explicitly asks to create a ticket
- or the user asks whether a ticket already exists

### Step 2: Frustration handling

The orchestrator checks:

- sentiment score from VADER
- frustration level from rules
- sarcasm detection

### Two complaint entry modes

#### A. Very severe complaints

If sentiment is very negative (`<= -0.9`), the bot skips the support-options step and goes directly into email collection for rapid escalation.

#### B. Normal complaint flow

If the complaint is negative but not in that extreme band, the bot offers:

1. create a support ticket
2. try to resolve it in chat

### Important threshold distinction

There are two separate negative thresholds in the current implementation:

- complaint flow shortcut threshold in orchestrator: `-0.9`
- ticket priority escalation threshold in ticket service: `-0.5`

So the bot may still create an `urgent` ticket even if the conversation did not skip directly to email collection.


## 15. Email Collection Logic for Tickets

The ticket flow is email-first.

### What that means now

Before creating most tickets, the bot tries to collect a follow-up email.

Supported behaviors:

- accept a clean email directly
- extract an email from free-form text like "my email is test@gmail.com"
- confirm extracted emails before use
- reuse an email already mentioned earlier in the chat
- allow the user to type `skip`
- allow natural-language refusal like "I do not want to share my email"

### Confirmation flow

If the bot extracts an email from natural language, it does not immediately create the ticket.

It asks:

- confirm with `yes`
- or change it with `change`

### Invalid email handling

- the bot tracks `email_attempts`
- after 3 failed attempts, it pauses ticket creation
- the flow returns to normal chat
- user can start again later by asking to create a ticket

### Topic switching during email collection

If the user changes topic to:

- FAQ
- order tracking
- greeting

the bot cancels the ticket email flow and routes the new topic normally.


## 16. Ticket Creation Logic

Ticket creation is handled by `TicketService`.

### Current ticket creation steps

1. check whether the session already has a ticket
2. if yes, return the existing ticket instead of creating a duplicate
3. analyze sentiment
4. map sentiment to priority
5. derive ticket status
6. insert ticket into database
7. if escalated, log a critical escalation event

### Current priority mapping

- `score <= sentiment_escalation_threshold` -> `urgent`
- `score <= negative threshold` -> `high`
- `score <= 0.05` -> `medium`
- otherwise -> `low`

### Current status mapping

- `urgent` -> `escalated`
- all others -> `open`

### Duplicate-ticket behavior

The current system is session-idempotent:

- one chat session should not create repeated tickets for the same conversation
- if a ticket already exists, the chatbot acknowledges that ticket instead of creating another one

### Ticket response style

After creation, the bot returns:

- ticket short id
- priority
- status
- whether email was captured

If no email exists, the bot asks the user to reference the ticket when contacting support directly.


## 17. Ticket Inquiry Flow

If the user asks something like:

- "do I have any ticket"
- "check my ticket status"

the bot checks the latest ticket for the session.

### If a ticket exists

The bot responds with:

- short ticket id
- whether it already has an email
- a reassurance that the team is already working on it

### If no ticket exists

The bot enters `SUPPORT_OPTIONS` state and offers:

1. yes, create a support ticket
2. no, continue without a ticket


## 18. Existing Ticket Behavior

The current UX does not hard-lock the chat after ticket creation.

Instead:

- session metadata is returned to `NORMAL_CHAT`
- `has_open_ticket` is stored
- if the user tries to create another ticket or complains again in the same session, the bot usually references the existing ticket

This is an important current behavior because many tests are explicitly protecting against duplicate tickets and broken session state.


## 19. Greeting and General Flow

### Greeting flow

Greeting messages return a capability-oriented welcome reply describing that the bot can help with:

- FAQs
- order tracking
- support tickets

### General flow

Messages that are not FAQ/order/complaint/greeting go to the general LLM path.

That path:

- loads recent history
- sends the user message to the configured LLM provider
- returns the generated answer


## 20. How the LLM Is Used

The chatbot does not use the LLM for everything equally.

### Main current LLM use cases

1. generating grounded FAQ responses
2. answering general chat messages
3. rewording operational replies to sound more natural
4. classifying ambiguous intents when rule-based logic is not enough
5. writing special-case order responses for cancelled or not-yet-shipped orders

### Providers supported

- Gemini
- OpenAI
- Anthropic

### Provider selection

Provider choice is controlled by environment variable:

- `LLM_PROVIDER=gemini`
- `LLM_PROVIDER=openai`
- `LLM_PROVIDER=anthropic`

The code uses a factory + shared interface, so the orchestrator is provider-agnostic.


## 21. LLM Guardrails in Reply Generation

One of the strongest design ideas in this project is that many important support replies start as deterministic drafts, then optionally get polished by the LLM.

### Current pattern

1. app builds a factually safe draft reply
2. app sends draft + facts to the LLM composer prompt
3. app checks whether the rewrite preserved required terms
4. if safe, use the polished version
5. if unsafe, fall back to the original deterministic draft

### Information that must survive rewrites

Examples:

- order ids
- ticket ids
- email addresses
- option numbers like `1` and `2`
- words like `skip`
- required follow-up instructions

This makes the bot sound more natural without letting the LLM remove operational details.


## 22. Conversation History Handling

The orchestrator loads the last 6 messages for context.

History is used in:

- FAQ response generation
- general chat generation
- operational reply composition
- some order replies

This means the bot has short memory inside a session, but not unlimited memory.


## 23. Frontend Flow

The main frontend chat UI lives in:

- `frontend/src/components/ChatWindow.tsx`

### Current UI behavior

- stores local message list in React state
- stores `sessionId` from backend responses
- sends messages through REST, not WebSocket
- shows loading indicator
- shows quick replies
- shows toast when a ticket is created
- supports resetting the conversation in the browser

### Initial quick replies

- "What is your return policy?"
- "Track my order"
- "I have a complaint"
- "What payment methods do you accept?"

### Intent-based quick-reply updates

- greeting responses reset suggestions to the initial set
- order-tracking responses show sample ids like `ORD-1001`
- other intents clear suggestions


## 24. Frontend Pages and Channels

### Landing page

`frontend/src/app/page.tsx`

Provides:

- hero section
- feature summary
- API/widget example snippet
- floating chat widget on the page

### Standalone chat page

`frontend/src/app/chat/page.tsx`

Provides:

- full-screen or centered chat layout
- optional embedded mode via `?embed=true`

### Floating widget component

`frontend/src/components/ChatWidget.tsx`

Provides:

- open/close floating launcher
- embedded `ChatWindow`

### Public embeddable script

`frontend/public/widget.js`

Provides:

- DOM-injected floating button
- iframe to `/chat?embed=true`
- simple embed story for other websites


## 25. REST vs WebSocket in the Current Product

The backend exposes both:

- REST chat endpoint
- WebSocket chat endpoint

However, the current Next.js frontend uses REST only.

### What this means right now

- chat UI sends one request per message through `POST /api/chat`
- WebSocket support exists on the backend but is not the primary UI path today
- the `stream()` methods in LLM providers are implemented, but the current frontend does not consume streaming tokens


## 26. FAQ Store Design

### Default mode

The default FAQ retrieval stack is:

- `SentenceTransformer`
- normalized embeddings
- FAISS `IndexFlatIP`
- persisted local metadata JSON and `.index` file

### Alternative mode

The project can switch to Pinecone with environment variables.

### Factory-based design

`get_vector_store()` returns a singleton store instance for the configured provider.

This makes the retrieval system pluggable without changing chatbot logic.


## 27. Order Service Design

`OrderService` provides:

- get order by id
- get orders by email
- get order by tracking number
- build a human-readable status summary

The chatbot currently uses the service mostly for:

- direct order lookup by id
- generating the base summary used in chat replies


## 28. Complaint Service Design

The complaints area is split cleanly:

- `sentiment.py` handles VADER scoring
- `ticket_service.py` handles ticket persistence, priority, and duplicate prevention

This keeps complaint classification separate from complaint persistence.


## 29. Seeded Demo Data

### Seeded orders

The backend currently seeds 5 sample orders:

- delivered
- in transit
- processing
- shipped
- cancelled

These make the order flow demo-ready.

### Seeded FAQs

The FAQ JSON includes topics such as:

- returns
- shipping
- tracking
- payments
- contact
- account help
- warranty
- gift wrapping


## 30. Current Environment-Driven Configuration

Key environment settings:

- `LLM_PROVIDER`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `VECTOR_STORE_PROVIDER`
- `PINECONE_API_KEY`
- `EMBEDDING_MODEL`
- `FAISS_INDEX_PATH`
- `DATABASE_URL`
- `FRONTEND_URL`
- `ALLOWED_ORIGINS`
- `SENTIMENT_ESCALATION_THRESHOLD`
- `SENTIMENT_NEGATIVE_THRESHOLD`
- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_WS_URL`


## 31. Testing Coverage and What It Tells Us

The tests show what behaviors the project currently considers important:

- intent classification correctness
- order flow correctness
- complaint escalation and email collection
- duplicate-ticket prevention
- state-machine preservation across multi-turn conversations
- prompt-injection rejection
- safe LLM rewrite guardrails
- API-level UI flow behavior

The test suite is especially focused on preventing conversation-state regressions.


## 32. Current Chatbot Logic Summary by Workflow

### FAQ workflow

- classify as FAQ
- search vector store
- if strong match, answer from grounded FAQ context
- otherwise say info is unavailable and direct to support

### Order workflow

- classify as order tracking
- ask for order id if missing
- reuse active order id on follow-ups
- exit order state if user changes topic

### Complaint workflow

- classify frustration/complaint
- if severe, go straight to email collection
- otherwise offer ticket vs in-chat resolution
- collect or confirm email
- create ticket
- prevent duplicate tickets

### General workflow

- send to LLM with recent history


## 33. Important Current Observations

These are useful to know about the current version of the project:

- `ChatOrchestrator` is the true center of the system
- chatbot state is stored in DB metadata, not only in frontend memory
- frontend currently uses REST only, even though WebSocket support exists
- LLMs are used with guardrails, not as the sole decision-maker
- ticket creation is session-idempotent
- complaint handling is more advanced than FAQ and order handling
- the system is demo-friendly because of startup seeding
- the app is designed to be provider-pluggable for both LLM and FAQ storage


## 34. One-Screen Flow Summary

```text
User message
  -> input validation
  -> load session + metadata
  -> check special state
     -> awaiting email? handle email flow
     -> awaiting support option? handle option flow
     -> awaiting order id? continue or allow topic switch
  -> classify intent
  -> analyze sentiment
  -> route
     -> FAQ -> vector search + grounded LLM answer
     -> ORDER -> DB lookup + order summary
     -> COMPLAINT -> frustration logic + email flow + ticket service
     -> GREETING -> welcome message
     -> GENERAL -> LLM response
  -> persist assistant reply
  -> return response + session_id + metadata
```


## 35. Bottom Line

The current project is a structured support assistant, not a generic chatbot.

Its actual design is:

- rule-first for routing
- state-machine-driven for support flows
- database-backed for sessions and tickets
- vector-search-backed for FAQs
- LLM-assisted for language quality and ambiguous cases

The strongest part of the current implementation is the complaint/ticket workflow, especially:

- state transitions
- email confirmation logic
- duplicate ticket prevention
- guarded reply composition
- multi-turn recovery from topic switches and edge cases
