# 💊 RxPilot — Multi-Agent AI for Pharmacy Operations

A portfolio-grade multi-agent AI system that extracts structured data from photographed pharmacy bills, validates against historical records, checks drug-safety risk via RAG, and forecasts reorder needs. Built with LangGraph orchestration, Claude Vision, and Langfuse observability.

> **⚠️ Portfolio Project** — This is a demonstration system for AI engineering interviews, not for real clinical use. All drug-interaction logic is research-grade and built from public data sources (OpenFDA, RxNorm). No real patient data is used anywhere in this system.

---

## ✨ What's Built (Phase 1)

### 🧠 Bill Extraction Agent
- **Claude Vision API** extracts structured fields (medicine name, batch number, expiry, quantity, supplier, price) from photographed paper bills
- **Multi-language support**: handles English, Hindi, and Marathi text on bills
- **Structured output validation**: pydantic schema enforcement with auto-retry on parse failure
- **Config flag** for swapping to open-weight VLM (Qwen2-VL) in Phase 4

### 🔗 LangGraph Orchestration
- `StateGraph` with typed `PharmacyState` passed through all agent nodes
- Conditional routing by input type (image → extraction, voice → safety/forecast)
- Designed for Phase 2+ additions: validation → safety → forecast nodes

### 📊 Langfuse Tracing (Day 1)
- Every agent call traced: input, output, token usage, latency, estimated cost
- Self-hosted Langfuse via Docker Compose
- Trace IDs linked in API responses for debugging

### 🐘 PostgreSQL + pgvector
- Bills and extracted line items stored in Postgres
- pgvector extension ready for Phase 2 RAG corpus

### 🖥️ Frontend
- Plain HTML/JS (no React) — drag-and-drop bill upload
- Real-time extraction results with per-item field cards
- Processing animation, error handling, history table
- Dark medical theme with glassmorphism design

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Frontend (nginx + HTML/JS)                 │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐  │
│  │  Upload   │ │  Results  │ │ History  │ │  Status   │  │
│  │  Zone     │ │  Cards    │ │  Table   │ │  Indicator│  │
│  └─────┬────┘ └───────────┘ └──────────┘ └───────────┘  │
│        │                                                   │
└────────┼───────────────────────────────────────────────────┘
         │ HTTP (POST /v1/upload)
┌────────▼───────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────┐     │
│  │         LangGraph StateGraph                      │     │
│  │  ┌──────────┐                                     │     │
│  │  │Extraction│──→ END  (Phase 1)                   │     │
│  │  │  Agent   │──→ Validation → Safety  (Phase 2)   │     │
│  │  └──────────┘                                     │     │
│  └──────────────────────────────────────────────────┘     │
│                         │                                   │
│  ┌──────────┐  ┌───────▼────┐  ┌────────────────────┐    │
│  │ Langfuse │  │  Claude    │  │   PostgreSQL        │    │
│  │ Tracing  │  │  Vision    │  │   + pgvector        │    │
│  └──────────┘  └────────────┘  └────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **Backend** | FastAPI |
| **Orchestration** | LangGraph (StateGraph) |
| **Vision LLM** | Claude (Anthropic API) |
| **Database** | PostgreSQL 16 + pgvector |
| **Observability** | Langfuse (self-hosted) |
| **Frontend** | Vanilla HTML/JS/CSS |
| **Deployment** | Docker Compose |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### 1. Configure Environment

```bash
# Clone and enter project
cd rxpilot

# Copy environment template
cp .env.example .env

# Edit .env and set your ANTHROPIC_API_KEY
```

### 2. Start Services

```bash
docker-compose up --build
```

This starts:
| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **Langfuse** | http://localhost:3001 |

### 3. Upload a Bill

1. Open http://localhost:3000
2. Drag & drop a pharmacy bill image (or click to browse)
3. Click "Extract with Claude Vision"
4. View the extracted medicine items, processing time, and cost
5. Check the Langfuse trace at http://localhost:3001

### Local Development (without Docker)

```bash
# Backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend: open frontend/index.html in browser
# Or serve with: python -m http.server 3000 -d frontend
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/upload` | Upload a bill image, run extraction pipeline |
| `GET` | `/v1/bills` | List recent bills |
| `GET` | `/v1/bills/{id}` | Get a single bill with extracted items |
| `GET` | `/health` | Health check (DB, Langfuse, Claude) |

### Example

```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@pharmacy_bill.jpg"
```

---

## 🧪 Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover:
- **State models** — pydantic validation, date normalization, serialization
- **Extraction agent** — JSON parsing, retry logic, mocked Claude API
- **API endpoints** — health, upload, bills list with mocked dependencies

---

## 📁 Project Structure

```
rxpilot/
├── agents/
│   ├── extraction_agent.py     # Claude vision extraction
│   └── state.py                # PharmacyState pydantic schema
├── graph.py                    # LangGraph StateGraph wiring
├── api/
│   ├── main.py                 # FastAPI app
│   ├── database.py             # Postgres connection + queries
│   └── routes/
│       ├── upload.py           # POST /v1/upload, GET /v1/bills
│       └── health.py           # GET /health
├── observability/
│   └── tracing.py              # Langfuse client wrapper
├── rag/                        # Phase 2: drug interaction RAG
├── voice/                      # Phase 3: ASR + TTS
├── eval/
│   ├── golden_set/             # Test bill images + expected JSON
│   ├── run_eval.py             # Evaluation runner
│   └── metrics.py              # Scoring functions
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_state.py
│   ├── test_extraction.py
│   └── test_api.py
├── scripts/
│   └── init-db.sql             # Postgres schema
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Built | Foundations + extraction agent |
| **Phase 2** | ✅ Built | Validation + safety RAG agent + eval harness |
| **Phase 3** | ✅ Built | Voice interface (ASR/TTS) + CI eval gate |
| **Phase 4** | 🔲 Planned | Forecast agent + VLM cost comparison + deploy |

---

## 📄 License

MIT
