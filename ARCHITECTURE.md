# RAG Evaluation Benchmarking Tool - Architecture Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES & APIs                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Google Gemini   │  │   OpenAI GPT-4   │  │  Anthropic Claude│ │
│  │  (Dataset Gen)   │  │  (Optional Multi)│  │  (Optional Multi)│ │
│  │  (RAG + Judge)   │  │     Model)       │  │     Model)       │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              △ △ △
                              │ │ │
                         HTTP Requests
                              │ │ │
        ┌─────────────────────┴─┴─┴─────────────────────┐
        │                                               │
┌───────▼──────────────────────────────────────────────▼───────┐
│                  FASTAPI BACKEND (Port 8000)                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Dependency Injection Container              │ │
│  │  (get_db, get_rag_pipeline, get_eval_engine)         │ │
│  └────────────────────────────────────────────────────────┘ │
│                         △                                    │
│                         │                                    │
│  ┌──────────────┬──────┴──────┬──────────────────────────┐  │
│  │              │             │                          │  │
│  ▼              ▼             ▼                          ▼  │
│ ┌───────────────────────────────────────────────────────────┐│
│ │              API ROUTES (src/api/routes/)                ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ POST   /ingest/files            (Upload PDFs/TXT/HTML) ││
│ │ GET    /ingest/collections      (List ChromaDB stores) ││
│ │ POST   /datasets/generate       (Create test dataset)   ││
│ │ GET    /datasets                (List all datasets)     ││
│ │ PATCH  /datasets/{id}/pairs/{p} (Edit Q&A pair)        ││
│ │ POST   /evaluate/run            (Run RAG + Judge)      ││
│ │ GET    /evaluate/{run_id}       (View full results)    ││
│ │ POST   /compare/run             (Multi-model matrix)   ││
│ │ GET    /compare/{matrix_id}     (View comparisons)     ││
│ └───────────────────────────────────────────────────────────┘│
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
     ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
     │  INGESTION  │  │  RAG CORE    │  │ EVALUATION  │
     │  ENGINE     │  │  PIPELINE    │  │  ENGINE     │
     └─────────────┘  └──────────────┘  └─────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
     ┌──────────────────────────────────────────────────┐
     │       BUSINESS LOGIC LAYER (src/)                │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/ingestion/                              │ │
     │  │  ├─ base.py (ABC: BaseLoader)               │ │
     │  │  ├─ loaders.py (PDF, TXT, HTML extraction) │ │
     │  │  ├─ chunker.py (RecursiveChunker)           │ │
     │  │  └─ pipeline.py (Orchestrator)              │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/vectorstore/                            │ │
     │  │  ├─ base.py (ABC: BaseVectorStore)          │ │
     │  │  ├─ embeddings.py (HuggingFace/Google)     │ │
     │  │  └─ chroma.py (ChromaDB implementation)    │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/dataset/                                │ │
     │  │  ├─ base.py (ABC: BaseDatasetGenerator)    │ │
     │  │  ├─ generator.py (Gemini Q&A generation)   │ │
     │  │  ├─ schema.py (Pydantic models)            │ │
     │  │  └─ store.py (JSON persistence)            │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/rag/                                    │ │
     │  │  ├─ base.py (ABC: BaseRAGPipeline)         │ │
     │  │  ├─ pipeline.py (Retrieve + Generate)      │ │
     │  │  └─ schema.py (RAGResponse model)           │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/evaluation/                             │ │
     │  │  ├─ base.py (ABC: BaseEvaluator)           │ │
     │  │  ├─ prompts.py (Judge prompt templates)    │ │
     │  │  ├─ faithfulness.py (Metric 1)             │ │
     │  │  ├─ answer_relevance.py (Metric 2)         │ │
     │  │  ├─ context_precision.py (Metric 3)        │ │
     │  │  ├─ correctness.py (Metric 4)              │ │
     │  │  ├─ schema.py (MetricScore model)          │ │
     │  │  └─ engine.py (Async executor)             │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/comparison/                             │ │
     │  │  ├─ schema.py (ComparisonMatrix model)     │ │
     │  │  ├─ runner.py (Async matrix executor)      │ │
     │  │  └─ aggregator.py (Statistical aggregator) │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ src/storage/                                │ │
     │  │  ├─ database.py (SQLAlchemy + Session)    │ │
     │  │  ├─ models.py (ORM schemas)                │ │
     │  │  └─ repository.py (CRUD + DAO pattern)    │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     │  ┌─────────────────────────────────────────────┐ │
     │  │ config/                                     │ │
     │  │  ├─ settings.py (Pydantic BaseSettings)    │ │
     │  │  ├─ models.yaml (Model registry + costs)  │ │
     │  │  └─ eval.yaml (Metric weights + prompts)  │ │
     │  └─────────────────────────────────────────────┘ │
     │                                                  │
     └──────────────────────────────────────────────────┘
                     △ △ △
                     │ │ │
        ┌────────────┴─┴─┴────────────┐
        │                             │
        ▼                             ▼
    ┌─────────────────┐          ┌──────────────┐
    │   ChromaDB      │          │   SQLite     │
    │   (Vector Store)│          │  (History &  │
    │   persistent/   │          │   Metadata)  │
    │   vectorstore   │          │  data/       │
    │   (384-dim      │          │  rag_eval.db │
    │    embeddings)  │          │              │
    └─────────────────┘          └──────────────┘
         △ △ △                        △ △ △
         │ │ │                        │ │ │
         └─┼─┼────────────┬───────────┘ │ │
           │ │            │             │ │
           │ │   ┌────────┴─────────────┼─┴───────────┐
           │ │   │                      │             │
           ▼ ▼   ▼                      ▼             ▼
    ┌────────────────────────────────────────────────────────┐
    │         STREAMLIT FRONTEND (Port 8501)                 │
    ├────────────────────────────────────────────────────────┤
    │                                                        │
    │  ui/app.py (Main entry point + page router)           │
    │                                                        │
    │  ┌──────────────────────────────────────────────────┐ │
    │  │ PAGES (Multi-page Streamlit UI)                 │ │
    │  ├──────────────────────────────────────────────────┤ │
    │  │ 01_ingest.py      (Upload + chunk documents)    │ │
    │  │ 02_dataset.py     (Generate/edit test dataset)  │ │
    │  │ 03_evaluate.py    (Run eval + view results)     │ │
    │  │ 04_compare.py     (Multi-model comparison)      │ │
    │  │ 05_history.py     (Past runs + export)          │ │
    │  └──────────────────────────────────────────────────┘ │
    │                                                        │
    │  ┌──────────────────────────────────────────────────┐ │
    │  │ COMPONENTS (Reusable UI widgets)                │ │
    │  ├──────────────────────────────────────────────────┤ │
    │  │ sidebar.py        (Global config sidebar)       │ │
    │  │ charts.py         (Plotly radar/bar/scatter)    │ │
    │  │ tables.py         (Drilldown interactive tables)│ │
    │  └──────────────────────────────────────────────────┘ │
    │                                                        │
    └────────────────────────────────────────────────────────┘
                          △
                          │
                   HTTP GET/POST
                   (JSON payloads)
                          │
              (Calls back to FastAPI Backend)
```

---

## Dependency Injection Flow

```
┌─────────────────────────────────────────────────────────────┐
│          FastAPI Dependency Injection Container             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  get_settings() → Settings (Pydantic BaseSettings)         │
│      ├─ Reads .env file                                   │
│      ├─ Validates all configs                            │
│      └─ Returns singleton                                │
│                                                             │
│  get_db() → AsyncSession (SQLAlchemy)                      │
│      ├─ Creates database session                         │
│      ├─ Registers cleanup hook                          │
│      └─ Yields to route handlers                        │
│                                                             │
│  get_embedding_manager() → EmbeddingManager                │
│      ├─ Loads sentence-transformers model              │
│      ├─ Wraps HuggingFace/Google embedder              │
│      └─ Caches in memory                               │
│                                                             │
│  get_vector_store() → ChromaVectorStore                   │
│      ├─ Connects to ChromaDB                           │
│      ├─ Injects embedding manager                      │
│      └─ Persists to disk                               │
│                                                             │
│  get_rag_pipeline() → RAGPipeline                         │
│      ├─ Injects embedding manager                      │
│      ├─ Injects vector store                           │
│      ├─ Configures Google Gemini API                  │
│      └─ Returns orchestrator                          │
│                                                             │
│  get_eval_engine() → EvaluationEngine                     │
│      ├─ Injects all 4 evaluators                       │
│      ├─ Loads judge prompts from config               │
│      ├─ Configures retry + backoff                    │
│      └─ Returns async executor                        │
│                                                             │
│  get_comparison_runner() → ComparisonRunner               │
│      ├─ Injects RAG pipeline                          │
│      ├─ Injects evaluation engine                     │
│      ├─ Configures async concurrency                 │
│      └─ Returns matrix runner                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Through Layers

```
┌──────────────────┐
│  Raw Document    │
│  (PDF/TXT/HTML)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        INGESTION PIPELINE               │
│  (src/ingestion/pipeline.py)            │
├─────────────────────────────────────────┤
│ 1. Load Document                        │
│    └─ BaseLoader → PDFLoader/TXTLoader  │
│       (src/ingestion/loaders.py)        │
│                                         │
│ 2. Parse & Extract Text                │
│    └─ pypdf / BeautifulSoup             │
│                                         │
│ 3. Split into Chunks                   │
│    └─ RecursiveChunker (configurable)   │
│       (src/ingestion/chunker.py)        │
│                                         │
│ 4. Generate Embeddings                 │
│    └─ EmbeddingManager                  │
│       (src/vectorstore/embeddings.py)   │
│       Wraps HuggingFace or Google       │
│                                         │
│ 5. Store in Vector DB                  │
│    └─ ChromaVectorStore                 │
│       (src/vectorstore/chroma.py)       │
└────────┬────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │      ChromaDB Collection            │
    │  (384-dim embeddings + metadata)    │
    └─────────────────────────────────────┘
         △                      △
         │                      │
    ┌────┴──────────────────────┴─────┐
    │                                  │
    ▼                                  ▼
┌──────────────────────┐      ┌──────────────────────┐
│ DATASET GENERATION   │      │  RAG + EVALUATION    │
│ (Query → Q&A Pairs)  │      │  (Query → Scores)    │
└──────────────────────┘      └──────────────────────┘
    │                              │
    │                              │
    ▼                              ▼
┌──────────────────────────────────────────────┐
│   DATASET GENERATOR (src/dataset/)           │
│   (GeminiDatasetGenerator)                   │
├──────────────────────────────────────────────┤
│ 1. Sample random chunks from collection   │
│ 2. Prompt Gemini: "Generate 3 Q&A pairs"  │
│ 3. Parse structured responses              │
│ 4. Persist to JSON (versioned)             │
└───────┬────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────┐
  │  Synthetic Dataset   │
  │  (QAPair objects)    │
  │  data/datasets/      │
  └────────┬─────────────┘
           │
           ▼
        ┌──────────────────────────────────────────────────┐
        │  RAG PIPELINE (src/rag/pipeline.py)              │
        ├──────────────────────────────────────────────────┤
        │ For each question in dataset:                   │
        │                                                 │
        │ 1. Embed question                              │
        │    └─ EmbeddingManager.embed_query()           │
        │                                                │
        │ 2. Retrieve from ChromaDB                      │
        │    └─ Similarity search (top_k chunks)         │
        │       → RetrievedChunk objects                 │
        │                                                │
        │ 3. Build prompt                                │
        │    └─ System instruction                       │
        │    └─ Retrieved context (numbered)             │
        │    └─ Question                                 │
        │                                                │
        │ 4. Generate answer                             │
        │    └─ Gemini API call                          │
        │    └─ Extract tokens + latency                 │
        │       → RAGResponse                            │
        │                                                │
        └────────┬─────────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │    RAGResponse               │
    │  (answer, context, tokens,   │
    │   latency, cost estimate)    │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────┐
    │  EVALUATION ENGINE (src/evaluation/engine.py)    │
    ├──────────────────────────────────────────────────┤
    │ Async concurrent execution of 4 judges:        │
    │                                                 │
    │ Judge 1: FaithfulnessEvaluator                 │
    │   Q: Is answer grounded in context?           │
    │   Score: 1-5                                  │
    │   Reasoning: [full chain-of-thought]          │
    │   └─ src/evaluation/faithfulness.py           │
    │                                                │
    │ Judge 2: AnswerRelevanceEvaluator              │
    │   Q: Does answer address the question?        │
    │   Score: 1-5                                  │
    │   Reasoning: [full chain-of-thought]          │
    │   └─ src/evaluation/answer_relevance.py       │
    │                                                │
    │ Judge 3: ContextPrecisionEvaluator             │
    │   Q: Are retrieved docs useful?               │
    │   Score: 1-5                                  │
    │   Reasoning: [full chain-of-thought]          │
    │   └─ src/evaluation/context_precision.py      │
    │                                                │
    │ Judge 4: CorrectnessEvaluator                  │
    │   Q: Does answer match ground truth?          │
    │   Score: 1-5                                  │
    │   Reasoning: [full chain-of-thought]          │
    │   └─ src/evaluation/correctness.py            │
    │                                                │
    │ → All results bundled into EvalResult          │
    │                                                │
    └────────┬──────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  EvalResult (per question)    │
    │  ├─ Faithfulness score + why │
    │  ├─ AnswerRelevance + why    │
    │  ├─ ContextPrecision + why   │
    │  └─ Correctness + why        │
    │                               │
    │  Composite Score =            │
    │   (weights summed to 1.0)    │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────┐
    │ STORAGE LAYER (src/storage/repository.py)       │
    ├──────────────────────────────────────────────────┤
    │ Persist to SQLite:                              │
    │  - RunRecord (metadata)                        │
    │  - EvalResultRecord (scores + reasoning)       │
    │  - DatasetRecord (version history)             │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────┐
    │   SQLite Database    │
    │  data/rag_eval.db    │
    └──────────────────────┘
             △
             │
             ▼
    ┌──────────────────────────────────────────────────┐
    │  COMPARISON ENGINE (src/comparison/runner.py)    │
    ├──────────────────────────────────────────────────┤
    │ Execute grid of model configurations:          │
    │                                                 │
    │ For each model in [gemini, gpt-4, claude]:    │
    │   For each top_k in [3, 5, 10]:               │
    │     For each temperature in [0.0, 0.5, 1.0]: │
    │       └─ Run RAG + Evaluate → Store           │
    │                                                │
    │ → ComparisonMatrix (all results)              │
    │ → ResultAggregator (means, std devs, costs)   │
    │                                                │
    └────────┬──────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  DASHBOARD (ui/pages/)       │
    │  Display:                    │
    │  ├─ Radar charts             │
    │  ├─ Score breakdowns          │
    │  ├─ Cost vs. quality          │
    │  ├─ Latency histograms       │
    │  └─ Drilldown tables         │
    └──────────────────────────────┘
```

---

## Configuration Management

```
┌─────────────────────────────────────────────────────────┐
│              CONFIGURATION HIERARCHY                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  .env (Runtime environment variables)                  │
│   ├─ API keys (GEMINI_API_KEY, OPENAI_API_KEY)       │
│   ├─ Paths (DATA_DIR, DATASETS_DIR, CHROMA_DIR)      │
│   ├─ Chunking (DEFAULT_CHUNK_SIZE=1000)              │
│   ├─ Model selection (JUDGE_MODEL, DATASET_GEN_MODEL)│
│   ├─ Metric weights (WEIGHT_FAITHFULNESS=0.30)       │
│   └─ Infrastructure (DATABASE_URL, API_PORT)         │
│        │                                               │
│        ▼                                               │
│  config/settings.py (Pydantic validation)             │
│   ├─ Loads & parses .env                             │
│   ├─ Validates types + constraints                   │
│   ├─ Raises on startup if invalid                    │
│   └─ Returns immutable Settings object               │
│        │                                               │
│        ├─ GeminiConfig                               │
│        ├─ EmbeddingConfig                            │
│        ├─ ChromaConfig                               │
│        ├─ DatasetGenConfig                           │
│        ├─ EvaluationConfig                           │
│        ├─ StorageConfig                              │
│        └─ ComparisonConfig                           │
│        │                                               │
│        ▼                                               │
│  config/models.yaml (Model registry)                  │
│   ├─ Provider definitions (name, type, cost/1k)      │
│   ├─ Model catalog (gemini-1.5-pro, gpt-4, etc.)    │
│   ├─ Rate limits (TPM, RPM)                         │
│   └─ UI visualization settings                       │
│        │                                               │
│        ▼                                               │
│  config/eval.yaml (Evaluation settings)               │
│   ├─ Metric weights (must sum to 1.0)               │
│   ├─ Score thresholds (pass/fail gates)              │
│   ├─ Judge prompts (4 templates)                     │
│   └─ Scoring rubrics (1-5 scale definitions)         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
┌─────────────────┐
│  config/        │  ← ROOT (Everything depends on this)
│  settings.py    │
└────────┬────────┘
         │
    ┌────┴────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
┌──────────────────────────────────┐  ┌──────────────────┐
│ src/vectorstore/                │  │ src/ingestion/   │
│ ├─ embeddings.py                │  │ ├─ base.py       │
│ ├─ base.py                      │  │ ├─ loaders.py    │
│ └─ chroma.py                    │  │ ├─ chunker.py    │
│    (depends on config)          │  │ └─ pipeline.py   │
└────────┬────────────────────────┘  └────────┬─────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
              ┌──────────────────────────┐
              │ src/dataset/             │
              │ ├─ base.py               │
              │ ├─ generator.py          │
              │ ├─ schema.py             │
              │ └─ store.py              │
              │ (Depends on RAG + Croma) │
              └────────┬─────────────────┘
                       │
                       ▼
              ┌──────────────────────────┐
              │ src/rag/                 │
              │ ├─ base.py               │
              │ ├─ pipeline.py           │
              │ └─ schema.py             │
              │ (Depends on vectorstore) │
              └────────┬─────────────────┘
                       │
                       ▼
              ┌──────────────────────────┐
              │ src/evaluation/          │
              │ ├─ base.py               │
              │ ├─ prompts.py            │
              │ ├─ faithfulness.py       │
              │ ├─ answer_relevance.py   │
              │ ├─ context_precision.py  │
              │ ├─ correctness.py        │
              │ ├─ schema.py             │
              │ └─ engine.py             │
              │ (Depends on RAG + config)│
              └────────┬─────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │ src/comparison/                      │
        │ ├─ schema.py                         │
        │ ├─ runner.py                         │
        │ └─ aggregator.py                     │
        │ (Depends on RAG + Evaluation engine) │
        └────────┬─────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────────┐
        │ src/storage/                         │
        │ ├─ database.py                       │
        │ ├─ models.py                         │
        │ └─ repository.py                     │
        │ (Depends on config + schemas)        │
        └────────┬─────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────────┐
        │ src/api/                             │
        │ ├─ app.py                            │
        │ ├─ dependencies.py                   │
        │ └─ routes/                           │
        │    ├─ ingest.py                      │
        │    ├─ datasets.py                    │
        │    ├─ evaluate.py                    │
        │    └─ compare.py                     │
        │ (DI orchestrator — glues all layers) │
        └────────┬─────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────────┐
        │ ui/                                  │
        │ ├─ app.py                            │
        │ ├─ pages/                            │
        │ │  ├─ 01_ingest.py                   │
        │ │  ├─ 02_dataset.py                  │
        │ │  ├─ 03_evaluate.py                 │
        │ │  ├─ 04_compare.py                  │
        │ │  └─ 05_history.py                  │
        │ └─ components/                       │
        │    ├─ sidebar.py                     │
        │    ├─ charts.py                      │
        │    └─ tables.py                      │
        │ (HTTP calls to FastAPI backend)      │
        └──────────────────────────────────────┘
```

---

## Concurrency & Async Flow

```
┌─────────────────────────────────────────────────────────────┐
│              ASYNC EXECUTION PATTERNS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Evaluation Engine Async Flow                           ││
│  ├────────────────────────────────────────────────────────┤│
│  │                                                        ││
│  │  For each QAPair:                                     ││
│  │    1. Call RAGPipeline.answer_pair() ───┐            ││
│  │       (fetch context + generate)        │ sync      ││
│  │                                         │            ││
│  │    2. Dispatch 4 judge tasks ────────────→ async    ││
│  │       ├─ Judge 1: Faithfulness (→ await) │          ││
│  │       ├─ Judge 2: Answer Relevance      │          ││
│  │       ├─ Judge 3: Context Precision     │          ││
│  │       └─ Judge 4: Correctness           │          ││
│  │       (All 4 run concurrently)          │          ││
│  │                                         │          ││
│  │    3. await asyncio.gather(...)        ◄─┘          ││
│  │       → Collect all 4 scores                        ││
│  │                                                        ││
│  │    4. Aggregate + store in DB                        ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Comparison Runner Async Flow                           ││
│  ├────────────────────────────────────────────────────────┤│
│  │                                                        ││
│  │  Grid: 3 models × 3 top_k × 3 temps = 27 configs   ││
│  │                                                        ││
│  │  For each config:                                    ││
│  │    ├─ answer_dataset(model_config)                   ││
│  │    └─ Limited by MAX_CONCURRENT_RUNS                ││
│  │       (e.g., run 3 at a time, not all 27)          ││
│  │                                                        ││
│  │  Within each dataset run:                            ││
│  │    └─ Parallel evaluation of all pairs              ││
│  │       using EvaluationEngine                         ││
│  │                                                        ││
│  │  Result: ComparisonMatrix (all configs × results)   ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Rate Limit Management                                 ││
│  ├────────────────────────────────────────────────────────┤│
│  │                                                        ││
│  │  Gemini Free Tier: ~60 RPM, ~4M TPM                  ││
│  │                                                        ││
│  │  Strategy:                                            ││
│  │  ├─ tenacity.retry with exponential backoff          ││
│  │  ├─ Detect 429 Too Many Requests → wait 60s          ││
│  │  ├─ Respect cost limits per API call                 ││
│  │  └─ Log all retries to observability                 ││
│  │                                                        ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Error Handling & Observability

```
┌──────────────────────────────────────────────────────────────┐
│              ERROR HANDLING & RECOVERY                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Custom Exception Hierarchy:                                │
│  └─ BaseException                                           │
│     ├─ LoaderError (File parsing failed)                   │
│     ├─ ChunkerError (Text splitting failed)                │
│     ├─ EmbeddingError (Embedding generation failed)        │
│     ├─ VectorStoreError (ChromaDB operation failed)        │
│     ├─ DatasetGenerationError (Gemini call failed)         │
│     ├─ RAGPipelineError (Retrieval/generation failed)      │
│     ├─ EvaluationError (Judge call failed)                 │
│     └─ ComparisonError (Matrix execution failed)           │
│                                                              │
│  Retry Strategy (via tenacity):                             │
│  ├─ Max retries: 3-5 depending on failure type            │
│  ├─ Backoff: exponential (2s → 4s → 8s → 16s)            │
│  ├─ Detect: HTTP 429, 503, timeout                        │
│  └─ Log: All attempts + final failure reason              │
│                                                              │
│  Observability Stack:                                       │
│  ├─ loguru (structured logging to file + console)         │
│  ├─ Phoenix (LLM tracing + prompt inspection)             │
│  ├─ LangSmith (optional tracing + cost tracking)          │
│  └─ CustomMetrics (latency, token count, cost per call)   │
│                                                              │
│  Graceful Degradation:                                      │
│  ├─ RAGPipeline never raises—returns error RAGResponse    │
│  ├─ Evaluation skips failed pairs, logs warning          │
│  ├─ Comparison marks failed runs, continues others       │
│  └─ UI shows status badge: ✓ Success, ⚠ Partial, ✗ Failed│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```
