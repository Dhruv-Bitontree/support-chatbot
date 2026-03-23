# Customer Support Chatbot

A production-grade, AI-powered customer support chatbot with FAQ search, order tracking, complaint escalation, and multi-channel deployment.

## Architecture

```
support-chatbot/
├── backend/          # Python/FastAPI API server
│   ├── app/
│   │   ├── services/
│   │   │   ├── llm/          # LLM abstraction (Gemini/OpenAI/Claude)
│   │   │   ├── faq/          # Vector store (FAISS/Pinecone)
│   │   │   ├── orders/       # SQLite order tracking
│   │   │   ├── complaints/   # Sentiment analysis + tickets
│   │   │   └── chat/         # Chat orchestration + intent routing
│   │   ├── routers/          # API endpoints
│   │   ├── models/           # Pydantic schemas
│   │   └── db/               # SQLAlchemy + SQLite
│   └── tests/
├── frontend/         # Next.js 15 + Tailwind CSS
│   ├── src/
│   │   ├── app/              # Pages (landing, standalone chat)
│   │   ├── components/       # Chat UI components
│   │   └── lib/              # API client, types, utilities
│   └── public/
│       └── widget.js         # Embeddable chat widget script
└── docker-compose.yml
```

## Features

### FAQ Search
- **Default**: FAISS vector database with `sentence-transformers` embeddings (`all-MiniLM-L6-v2`)
- **Pluggable**: Swap to Pinecone via `VECTOR_STORE_PROVIDER=pinecone`
- Semantic search with cosine similarity scoring
- LLM-synthesized natural language answers from FAQ matches

### Order Tracking
- SQLite database with async access via `aiosqlite`
- Lookup by order ID, email, or tracking number
- Human-readable status summaries with item details
- Seeded with 5 demo orders for testing

### Complaints & Escalation
- **VADER** sentiment analysis for real-time scoring
- Automatic priority assignment: `LOW` / `MEDIUM` / `HIGH` / `URGENT`
- Urgent complaints auto-escalated (sentiment < -0.5)
- Support tickets with full lifecycle tracking

### Multi-Channel
- **Embeddable Widget**: `<script src="widget.js">` for any website
- **Standalone Page**: Full-page chat at `/chat`
- **REST API**: `POST /api/chat` for programmatic access
- **WebSocket**: `ws://localhost:8000/api/chat/ws` for real-time streaming

### LLM Abstraction Layer
- **Default**: Google Gemini (`gemini-2.0-flash`)
- **Plug-and-play**: Switch via environment variable
  - `LLM_PROVIDER=openai` → GPT-4o-mini
  - `LLM_PROVIDER=anthropic` → Claude Sonnet
- Strategy pattern with ABC interface for custom providers

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- An LLM API key (Gemini, OpenAI, or Anthropic)

### 1. Clone and Configure

```bash
cd support-chatbot
cp .env.example .env
# Edit .env with your API key
```

### 2. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend will:
- Create SQLite tables on startup
- Seed 5 demo orders and 15 FAQ entries
- Download the embedding model (~90MB, first run only)

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit:
- **Landing page**: http://localhost:3000 (with floating widget)
- **Full chat**: http://localhost:3000/chat
- **API health**: http://localhost:8000/api/health

### 4. Docker (Alternative)

```bash
cp .env.example .env
# Edit .env
docker compose up --build
```

## API Reference

### Chat
```bash
# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?", "channel": "api"}'
```

### Orders
```bash
# Get order by ID
curl http://localhost:8000/api/orders/ORD-1001

# Lookup by email
curl -X POST http://localhost:8000/api/orders/lookup \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com"}'
```

### Complaints
```bash
# File a complaint
curl -X POST http://localhost:8000/api/complaints \
  -H "Content-Type: application/json" \
  -d '{"message": "My order arrived damaged!", "customer_email": "user@example.com"}'
```

### FAQs
```bash
# Search FAQs
curl -X POST http://localhost:8000/api/faq/search \
  -H "Content-Type: application/json" \
  -d '{"query": "shipping", "top_k": 3}'
```

### Widget Embedding
```html
<!-- Add to any website -->
<script src="http://localhost:3000/widget.js"
        data-api-url="http://localhost:3000"></script>
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | LLM backend: `gemini`, `openai`, `anthropic` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `VECTOR_STORE_PROVIDER` | `faiss` | Vector DB: `faiss`, `pinecone` |
| `PINECONE_API_KEY` | — | Pinecone API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `DATABASE_URL` | `sqlite+aiosqlite:///./support.db` | Database connection |
| `SENTIMENT_ESCALATION_THRESHOLD` | `-0.5` | Auto-escalation threshold |
| `LOG_LEVEL` | `info` | Logging level |

## Testing

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/ -v
```

Tests cover:
- Intent classification (keyword-based)
- Chat orchestration (greeting, order, complaint flows)
- Order service (CRUD, lookups, status summary)
- Sentiment analysis and ticket creation/escalation
- FAQ vector store operations (mock)

## Design Decisions

- **FAISS over hosted vector DB**: Zero-cost default, no external dependency. Pinecone available when scale demands it.
- **VADER over LLM sentiment**: Sub-millisecond classification without API calls. LLM fallback available for complex cases.
- **SQLite over Postgres**: Zero-config default ideal for development and small deployments. Swap `DATABASE_URL` for production.
- **Keyword intent + LLM fallback**: Fast keyword matching handles 80% of cases; LLM called only for ambiguous inputs.
- **Strategy pattern for LLM/VectorStore**: ABC + factory pattern enables swapping providers with zero code changes.
