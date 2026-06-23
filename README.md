# 📊 RAG Eval Bench

**A production-grade RAG Evaluation Benchmarking Tool with LLM-as-a-Judge scoring, multi-model comparison, and a full interactive dashboard.**

Most RAG projects stop at "build a chatbot that answers questions from documents." This one answers the harder question: **how do you know if it's actually good?**

RAG Eval Bench ingests documents, auto-generates a test dataset of question/answer pairs via an LLM, runs those questions through any RAG pipeline configuration, and scores every answer across four distinct quality dimensions using a second LLM acting as an impartial judge — with full reasoning shown for every score, not just a number.

---

## Why this exists

Most candidates can build a RAG chatbot. Almost none can answer "how do you know it's good?" with anything more rigorous than "I tried a few questions and it seemed fine." This tool is the answer to that question, built as a real, runnable system rather than a slide.

---

## ✨ Features

- **🧪 Synthetic dataset generation** — Gemini reads your documents and generates realistic Q&A pairs automatically, with manual editing before evaluation.
- **⚖️ LLM-as-a-Judge across 4 metrics** — Faithfulness, Answer Relevance, Context Precision, and Correctness, each with a dedicated prompt, rubric, and chain-of-thought reasoning before scoring.
- **🔍 Reasoning, not just numbers** — every score ships with the judge's full explanation, surfaced directly in the dashboard's drilldown view. This is the feature most benchmarking tools skip.
- **🔀 Multi-model comparison** — run a parameter grid (models × top_k × temperature) concurrently and rank results side-by-side on a radar chart, bar chart, and latency/cost scatter plot.
- **📈 Full run history** — every evaluation and comparison is persisted, filterable, paginated, and exportable to CSV.
- **🆓 Runs entirely on free-tier infrastructure** — Google Gemini (free API key), HuggingFace local embeddings (zero API cost), ChromaDB (local, zero infra), SQLite (zero infra).
- **🏗️ Clean layered architecture** — every component (ingestion, embeddings, RAG, evaluation, comparison, persistence) is built behind an abstract interface, fully swappable without touching calling code.

---

## 🖥️ Demo

*(Screenshots/GIF placeholder — add after a local run)*

---

## 🏗️ Architecture
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐

│  Streamlit   │ HTTP │   FastAPI     │      │  Google Gemini   │

│  Dashboard   │─────▶│   Backend     │─────▶│  (RAG + Judge)   │

└─────────────┘      └──────┬───────┘      └─────────────────┘

│

┌─────────────────────┼─────────────────────┐

▼                     ▼                     ▼

┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐

│   ChromaDB     │   │     SQLite        │   │  HuggingFace      │

│ (vector store)  │   │ (run history)      │   │ (local embeddings) │

└───────────────┘   └──────────────────┘   └──────────────────┘


**Data flow:**
Document → Chunk → Embed → Store (ChromaDB)

│

▼

Sample chunks → Generate Q&A (Gemini)

│

▼

Question → Retrieve context → Generate answer (RAG)

│

▼

Answer + Context → 4 parallel Judge calls → Scores + Reasoning

│

▼

Aggregate → Persist (SQLite) → Dashboard

Every layer depends only on an abstract interface one level below it — `BaseLoader`, `BaseVectorStore`, `BaseEvaluator`, `BaseDatasetGenerator` — never on a concrete implementation. Swapping ChromaDB for Pinecone, or Gemini for GPT-4o, requires changes in exactly one place.

---

## 🧠 The 4 Evaluation Metrics

| Metric | Question it answers | Catches |
|---|---|---|
| **Faithfulness** | Is every claim in the answer grounded in the retrieved context? | Hallucination |
| **Answer Relevance** | Does the answer actually address the question asked? | Topic drift, partial answers |
| **Context Precision** | Of the retrieved chunks, how many were actually useful? | Poor retrieval / noisy chunking |
| **Correctness** | Does the answer match the ground-truth reference? | Factual errors (requires a reference dataset) |

Each metric is scored 1–5 by a dedicated LLM judge call with an explicit rubric, and the judge must produce reasoning *before* the score — a known technique for more reliable, less arbitrary LLM-as-a-Judge scoring.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| RAG + Judge LLM | Google Gemini (`google-genai`) | Free tier, generous context window |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, zero API cost, 384-dim |
| Vector store | ChromaDB | Local, zero infra, persistent |
| Backend | FastAPI + Pydantic v2 | Async, typed, auto-documented |
| Persistence | SQLite + SQLAlchemy (async) | Zero infra, swappable to PostgreSQL via one env var |
| Frontend | Streamlit + Plotly | Fast to build, genuinely interactive charts |
| Config | YAML (`models.yaml`, `eval.yaml`) + Pydantic validation | Tunable without code changes, validated at startup |
| Retry/resilience | `tenacity` | Exponential backoff on Gemini rate limits |
| Logging | `loguru` | Structured, readable |
| Dependency mgmt | Poetry | Reproducible environments |

---

## 📁 Project Structure
rag-eval-bench/

├── config/                 # Settings, model registry, eval prompts (YAML + Pydantic)

├── src/

│   ├── ingestion/           # Document loaders, chunker, ingestion pipeline

│   ├── vectorstore/         # Embedding manager, ChromaDB wrapper

│   ├── dataset/              # Synthetic Q&A generation, file-based dataset store

│   ├── rag/                   # RAG pipeline (retrieve + generate)

│   ├── evaluation/             # LLM-as-a-Judge engine, 4 metric evaluators

│   ├── comparison/              # Multi-model comparison runner

│   ├── storage/                  # SQLAlchemy models, async DB, repository

│   └── api/                       # FastAPI app, routes, dependency injection

├── ui/

│   ├── app.py                      # Streamlit entrypoint

│   ├── pages/                       # Ingest, Dataset, Evaluate, Compare, History

│   └── components/                   # API client, shared sidebar

├── tests/                             # pytest unit + integration tests

├── scripts/                            # CLI utilities (seed demo data, export reports)

├── .env.example

├── pyproject.toml

└── README.md


---

## 🚀 Getting Started

### Prerequisites

- Python 3.11–3.13
- A free [Google Gemini API key](https://aistudio.google.com/app/apikey) (no billing project attached)
- Poetry (`pip install poetry`)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/rag-eval-bench.git
cd rag-eval-bench
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
GEMINI_API_KEY=your_free_tier_key_here
SECRET_KEY=any_random_string_at_least_32_characters_long
APP_ENV=development
```

> **Note:** Gemini model availability varies by account/region. Check which models are currently available to your key and update `JUDGE_MODEL` / `DATASET_GEN_MODEL` in `.env` and the `defaults` block in `config/models.yaml` accordingly.

### 3. Start the backend

```bash
poetry run uvicorn src.api.app:create_app --factory --reload --port 8000
```

Verify it's healthy:

```bash
curl http://localhost:8000/health
```

Browse the interactive API docs at **http://localhost:8000/docs**.

### 4. Start the frontend

In a second terminal:

```bash
poetry run streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## 📖 Usage Walkthrough

1. **Ingest** — Upload a PDF/TXT/HTML/DOCX into a named ChromaDB collection.
2. **Generate Dataset** — Sample chunks from that collection and auto-generate question/answer pairs via Gemini. Edit or remove any pair before evaluating.
3. **Evaluate** — Pick a model, top_k, and temperature; run the dataset through RAG + the 4-metric judge. Drill into any question to see the judge's full reasoning.
4. **Compare** — Run multiple models (and/or parameter combinations) against the same dataset; view the radar chart, score breakdown, and latency/cost trade-off.
5. **History** — Browse, filter, export, or delete any past run or comparison.

---

## 🔌 API Overview

All endpoints are documented live at `/docs`. 
Summary:
POST   /api/v1/ingest/files               Upload + ingest documents

GET    /api/v1/ingest/collections          List ChromaDB collections
POST   /api/v1/datasets/generate            Generate synthetic Q&A dataset

GET    /api/v1/datasets                      List datasets

PATCH  /api/v1/datasets/{id}/pairs/{pid}      Edit a pending pair
POST   /api/v1/evaluate/run                    Run RAG + judge evaluation

GET    /api/v1/evaluate/{run_id}                Full per-question results
POST   /api/v1/compare/run                       Run a multi-model comparison

GET    /api/v1/compare/{matrix_id}                Full comparison matrix

---

## 🧪 Running Tests

```bash
poetry run pytest
```

Coverage report generated at `htmlcov/index.html`.

---

## 🗺️ Roadmap

- [ ] Dockerfile + docker-compose for one-command startup
- [ ] Deploy to Streamlit Cloud / Render
- [ ] PostgreSQL migration path via Alembic (SQLite → Postgres is a one-line `DATABASE_URL` change; models already support both)
- [ ] LangSmith tracing integration for every LLM call
- [ ] Support for OpenAI / Anthropic / Groq as additional RAG and judge providers (registry already supports this in `models.yaml` — provider clients pending)
- [ ] PDF/HTML report export (in addition to existing CSV export)

---

## 🏛️ Design Principles

This project was built with production engineering discipline, not notebook-style scripting:

- **Dependency inversion everywhere** — every concrete implementation (ChromaDB, Gemini, HuggingFace) sits behind an abstract base class. The calling code never knows which implementation it's talking to.
- **Config over code** — chunk size, judge prompts, metric weights, and model registry all live in YAML, validated by Pydantic at startup. Tuning the system never requires touching Python.
- **Fail loud, fail fast** — misconfiguration (bad API key, mismatched embedding dimensions, invalid weights) raises immediately at startup, never silently degrades.
- **Reasoning over scores** — every LLM judgment is required to explain itself before producing a number. A score without reasoning is undebuggable.
- **Free-tier-aware by design** — rate limit semaphores, safety caps on comparison grid size, and cost estimation before execution, because this was built and tested entirely on a free Gemini API key.

---

## 📄 License

MIT

---

## 🙋 Author

Built by [Rohan](https://github.com/rohan-1103) as a portfolio project demonstrating production-grade RAG evaluation engineering — LangChain/LangGraph fundamentals, RAG/Vectorless RAG architecture, LLM Gateways, and rigorous evaluation methodology, applied end-to-end.