# RAG Evaluation Benchmarking Tool - Workflow Diagram

## Complete End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       START: USER INTERACTION                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  User opens Streamlit UI      │
                    │  (http://localhost:8501)      │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
    ┌─────────┐               ┌─────────┐               ┌─────────┐
    │ INGEST  │               │ DATASET │               │EVALUATE │
    │  FLOW   │               │ GEN     │               │  FLOW   │
    └────┬────┘               └────┬────┘               └────┬────┘
         │                         │                         │
         │ [USER JOURNEY 1]        │ [USER JOURNEY 2]        │ [USER JOURNEY 3]
         │                         │                         │
         │                         │                         │
         ▼                         ▼                         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │         JOURNEY 1: DOCUMENT INGESTION WORKFLOW                 │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Step 1: User selects "Ingest" page                           │
    │  ──────────────────────────────────────────────                │
    │  ├─ Displays file upload widget                              │
    │  ├─ Shows chunking config sliders:                           │
    │  │  ├─ Chunk size (default: 1000)                           │
    │  │  ├─ Chunk overlap (default: 200)                         │
    │  │  └─ Max pages for PDF                                   │
    │  └─ Displays list of existing collections                   │
    │                                                                 │
    │  Step 2: User uploads document (PDF/TXT/HTML)                │
    │  ──────────────────────────────────────────                  │
    │  ├─ HTTP POST /ingest/files                                 │
    │  │  ├─ File: <binary>                                      │
    │  │  ├─ collection_name: "hr_policies"                      │
    │  │  ├─ chunk_size: 1000                                    │
    │  │  └─ chunk_overlap: 200                                  │
    │  │                                                            │
    │  │  ▼                                                        │
    │  │  [FastAPI Route Handler]                                │
    │  │  POST /api/v1/ingest/files                              │
    │  │                                                            │
    │  └─→ Dependency Injection:                                  │
    │      ├─ get_settings()                                     │
    │      ├─ get_vector_store()                                 │
    │      └─ get_db()                                           │
    │                                                                 │
    │  Step 3: File Parsing (IngestionPipeline)                   │
    │  ──────────────────────────────                             │
    │  src/ingestion/pipeline.py:                                 │
    │  │                                                            │
    │  ├─ Select loader based on file extension                  │
    │  │  ├─ .pdf → PDFLoader                                    │
    │  │  ├─ .txt → TXTLoader                                    │
    │  │  └─ .html → HTMLLoader                                  │
    │  │                                                            │
    │  ├─ Extract raw text                                        │
    │  │  ├─ pypdf.PdfReader for PDF                            │
    │  │  ├─ open().read() for TXT                              │
    │  │  └─ BeautifulSoup for HTML                             │
    │  │                                                            │
    │  ├─ Track metadata                                          │
    │  │  ├─ source_file: "hr_policies.pdf"                      │
    │  │  ├─ page_number: 1, 2, 3... (if applicable)            │
    │  │  └─ loader_class: "PDFLoader"                           │
    │  │                                                            │
    │  └─ Log: "Loaded 42 pages, 28,500 characters"              │
    │                                                                 │
    │  Step 4: Text Chunking (RecursiveChunker)                   │
    │  ─────────────────────────────                              │
    │  src/ingestion/chunker.py:                                  │
    │  │                                                            │
    │  ├─ Split text by separators (in order):                   │
    │  │  ├─ Try: "\n\n" (paragraph)                            │
    │  │  ├─ Try: "\n" (newline)                                │
    │  │  ├─ Try: " " (space)                                   │
    │  │  └─ Try: "" (character)                                │
    │  │                                                            │
    │  ├─ Respect max chunk size (1000 chars)                    │
    │  ├─ Add overlap (200 chars) between chunks                │
    │  ├─ Preserve source metadata per chunk                     │
    │  │                                                            │
    │  └─ Result: [                                               │
    │      {                                                        │
    │        "content": "Company policy on leave...",            │
    │        "metadata": {                                        │
    │          "source_file": "hr_policies.pdf",                │
    │          "page_number": 1,                                │
    │          "chunk_index": 0,                                │
    │          "loader_class": "PDFLoader"                      │
    │        }                                                      │
    │      },                                                       │
    │      ... (more chunks)                                     │
    │    ]                                                          │
    │                                                                 │
    │  Step 5: Embedding Generation (EmbeddingManager)           │
    │  ────────────────────────────────                          │
    │  src/vectorstore/embeddings.py:                            │
    │  │                                                            │
    │  ├─ Load model: "all-MiniLM-L6-v2" (HuggingFace)           │
    │  │  └─ First run: Downloads ~27MB                         │
    │  │  └─ Subsequent runs: Cached locally                   │
    │  │                                                            │
    │  ├─ For each chunk:                                        │
    │  │  ├─ Call SentenceTransformer.encode(chunk_text)        │
    │  │  └─ Get 384-dimensional vector                        │
    │  │                                                            │
    │  ├─ Batch mode: Process 32 chunks concurrently            │
    │  │                                                            │
    │  └─ Track: "Embedded 127 chunks in 3.2s"                  │
    │                                                                 │
    │  Step 6: ChromaDB Storage (ChromaVectorStore)              │
    │  ──────────────────────────                                │
    │  src/vectorstore/chroma.py:                                │
    │  │                                                            │
    │  ├─ Connect to ChromaDB (persistent/vectorstore)           │
    │  │                                                            │
    │  ├─ Create/get collection: "hr_policies"                   │
    │  │                                                            │
    │  ├─ For each (chunk, vector, metadata):                   │
    │  │  ├─ Generate unique ID: ulid()                         │
    │  │  ├─ Add to collection:                                 │
    │  │  │  ├─ ids: [unique_id]                               │
    │  │  │  ├─ embeddings: [384-dim vector]                   │
    │  │  │  ├─ documents: [chunk_text]                        │
    │  │  │  └─ metadatas: [metadata_dict]                     │
    │  │  │                                                      │
    │  │  └─ Persist to disk                                   │
    │  │                                                            │
    │  └─ Result: "Added 127 chunks to hr_policies"             │
    │                                                                 │
    │  Step 7: Response to Frontend                              │
    │  ─────────────────────────                                 │
    │  ├─ HTTP 200 OK                                            │
    │  ├─ Response body:                                         │
    │  │  {                                                        │
    │  │    "collection_name": "hr_policies",                   │
    │  │    "chunks_added": 127,                                │
    │  │    "total_chars": 127000,                              │
    │  │    "embedding_time_ms": 3200,                          │
    │  │    "storage_time_ms": 850,                             │
    │  │    "total_time_ms": 4050                               │
    │  │  }                                                        │
    │  │                                                            │
    │  └─ Streamlit updates:                                     │
    │     ├─ ✓ Collection "hr_policies" created                 │
    │     ├─ Stats: 127 chunks, 384-dim embeddings              │
    │     └─ Available for next steps                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
         │
         │                        ┌──────────────────────────────────┐
         │                        │  ↓ (if user continues with same │
         └────────────────────────┤  dataset, go to JOURNEY 2)      │
                                  └──────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │         JOURNEY 2: SYNTHETIC DATASET GENERATION WORKFLOW       │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Prerequisites:                                                │
    │  ├─ Must have at least one ingested collection               │
    │  ├─ Must have GEMINI_API_KEY set in .env                    │
    │  └─ ChromaDB collection populated with chunks               │
    │                                                                 │
    │  Step 1: User navigates to "Dataset" page                   │
    │  ──────────────────────────────────────                     │
    │  ├─ Displays dropdown: "Select collection"                  │
    │  │  └─ Shows "hr_policies" (from JOURNEY 1)                │
    │  │                                                            │
    │  ├─ Displays generation config:                             │
    │  │  ├─ n_samples: 10 (number of Q&A pairs)                │
    │  │  ├─ max_pairs_per_chunk: 3                             │
    │  │  ├─ temperature: 0.4                                   │
    │  │  └─ Dataset name: "hr_eval_v1"                         │
    │  │                                                            │
    │  └─ Button: "Generate Dataset"                             │
    │                                                                 │
    │  Step 2: Backend receives generation request                │
    │  ──────────────────────────────                             │
    │  HTTP POST /api/v1/datasets/generate                        │
    │  {                                                            │
    │    "collection_name": "hr_policies",                        │
    │    "num_samples": 10,                                       │
    │    "max_pairs_per_chunk": 3,                               │
    │    "temperature": 0.4,                                      │
    │    "dataset_name": "hr_eval_v1"                            │
    │  }                                                            │
    │                                                                 │
    │  Step 3: Query ChromaDB for random samples                  │
    │  ────────────────────────────                               │
    │  src/vectorstore/chroma.py:                                 │
    │  │                                                            │
    │  ├─ Get all document IDs from collection                   │
    │  ├─ Randomly sample 10 chunks (without replacement)        │
    │  │  └─ Returns full document + metadata                    │
    │  │                                                            │
    │  └─ Result:                                                 │
    │     [                                                         │
    │       {                                                       │
    │         "id": "01ARZ3NHM...",                              │
    │         "text": "Leave policy: 20 days annual...",         │
    │         "metadata": {...}                                  │
    │       },                                                      │
    │       ... (9 more chunks)                                  │
    │     ]                                                         │
    │                                                                 │
    │  Step 4: Generate Q&A pairs via Gemini                      │
    │  ────────────────────────                                   │
    │  src/dataset/generator.py:                                  │
    │  │                                                            │
    │  ├─ For each sampled chunk:                                │
    │  │  │                                                        │
    │  │  ├─ Build prompt:                                       │
    │  │  │  {                                                     │
    │  │  │    "role": "You are a QA expert...",                │
    │  │  │    "context": "<sampled_chunk>",                    │
    │  │  │    "instruction": "Generate 3 Q&A pairs...",        │
    │  │  │    "format": "JSON: [                              │
    │  │  │      {\"question\": \"...\", \"answer\": \"...\"}   │
    │  │  │    ]"                                                │
    │  │  │  }                                                     │
    │  │  │                                                        │
    │  │  ├─ Call Gemini API:                                   │
    │  │  │  ├─ model: "gemini-1.5-flash" (fast + cheap)       │
    │  │  │  ├─ temperature: 0.4                               │
    │  │  │  ├─ max_tokens: 1000                               │
    │  │  │  └─ Retry on 429 (rate limit)                     │
    │  │  │                                                        │
    │  │  ├─ Parse response (extract JSON):                     │
    │  │  │  [                                                     │
    │  │  │    {                                                   │
    │  │  │      "question": "What is the annual leave...",    │
    │  │  │      "answer": "20 days as per policy..."           │
    │  │  │    },                                                  │
    │  │  │    {...},                                            │
    │  │  │    {...}                                             │
    │  │  │  ]                                                     │
    │  │  │                                                        │
    │  │  └─ Log: "Generated 3 pairs from chunk 1/10"           │
    │  │                                                            │
    │  └─ Total generated: 30 pairs (10 chunks × 3 per chunk)   │
    │                                                                 │
    │  Step 5: Create QAPair objects & Persist to JSON           │
    │  ──────────────────────────────────────                    │
    │  src/dataset/store.py:                                      │
    │  │                                                            │
    │  ├─ Build EvalDataset object:                              │
    │  │  {                                                        │
    │  │    "id": "dataset_01ARZ3NHM...",                       │
    │  │    "name": "hr_eval_v1",                               │
    │  │    "collection_name": "hr_policies",                   │
    │  │    "created_at": "2026-06-26T05:30:00Z",              │
    │  │    "pairs": [                                           │
    │  │      {                                                     │
    │  │        "id": "pair_1",                                 │
    │  │        "question": "What is the annual leave...",      │
    │  │        "answer": "20 days as per policy...",           │
    │  │        "status": "PENDING",                            │
    │  │        "source_chunk_id": "01ARZ3NHM..."               │
    │  │      },                                                    │
    │  │      ... (29 more pairs)                               │
    │  │    ]                                                      │
    │  │  }                                                        │
    │  │                                                            │
    │  ├─ Save to disk:                                          │
    │  │  └─ data/datasets/hr_eval_v1.json                      │
    │  │                                                            │
    │  └─ Also store in SQLite (RunRecord):                      │
    │     INSERT INTO datasets (id, name, collection_name) ...   │
    │                                                                 │
    │  Step 6: Return to Frontend                                │
    │  ─────────────────────────                                 │
    │  HTTP 200 OK                                               │
    │  {                                                            │
    │    "dataset_id": "dataset_01ARZ3NHM...",                  │
    │    "dataset_name": "hr_eval_v1",                          │
    │    "pairs_generated": 30,                                 │
    │    "total_time_ms": 12500,                               │
    │    "cost_estimate_usd": 0.018,                           │
    │    "status": "ready_for_review"                          │
    │  }                                                            │
    │                                                                 │
    │  Step 7: User Reviews & Edits Dataset                      │
    │  ────────────────────────────────                          │
    │  ├─ Streamlit displays all 30 pairs in a table:           │
    │  │  ├─ Column 1: Question                                │
    │  │  ├─ Column 2: Ground-truth answer                    │
    │  │  └─ Column 3: Actions (Edit/Delete)                  │
    │  │                                                           │
    │  ├─ User can:                                             │
    │  │  ├─ Delete bad pairs (e.g., generated nonsense)      │
    │  │  ├─ Modify questions/answers                        │
    │  │  └─ Mark as ready for evaluation                    │
    │  │                                                           │
    │  ├─ PATCH /api/v1/datasets/{dataset_id}/pairs/{pair_id} │
    │  │  Request body: {"question": "...", "answer": "..."}  │
    │  │  Updates: PENDING → REVIEWED                        │
    │  │                                                           │
    │  └─ Final dataset: 28 pairs (removed 2 bad ones)        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
         │
         │                        ┌──────────────────────────────────┐
         │                        │  ↓ (User navigates to Evaluate  │
         └────────────────────────┤  page to run RAG + scoring)     │
                                  └──────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │         JOURNEY 3: EVALUATION & SCORING WORKFLOW               │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Prerequisites:                                                │
    │  ├─ Ingested collection exists (hr_policies)                 │
    │  ├─ Dataset generated & reviewed (hr_eval_v1)               │
    │  ├─ GEMINI_API_KEY set                                      │
    │  └─ Judge model available (gemini-1.5-pro)                 │
    │                                                                 │
    │  Step 1: User navigates to "Evaluate" page                  │
    │  ──────────────────────────────────────────                 │
    │  ├─ Displays evaluation form:                               │
    │  │  ├─ Select dataset: "hr_eval_v1"                        │
    │  │  ├─ Select RAG model: "gemini-1.5-flash"              │
    │  │  ├─ RAG config:                                         │
    │  │  │  ├─ top_k: 5 (retrieve top 5 chunks)               │
    │  │  │  ├─ temperature: 0.2 (low for RAG)                │
    │  │  │  └─ max_output_tokens: 500                        │
    │  │  │                                                        │
    │  │  ├─ Evaluation config:                                 │
    │  │  │  ├─ Judge model: "gemini-1.5-pro"                 │
    │  │  │  ├─ Judge temperature: 0.0 (deterministic)        │
    │  │  │  └─ Metric weights (from config/eval.yaml):       │
    │  │  │     ├─ Faithfulness: 0.30                        │
    │  │  │     ├─ AnswerRelevance: 0.25                    │
    │  │  │     ├─ ContextPrecision: 0.25                  │
    │  │  │     └─ Correctness: 0.20                      │
    │  │  │                                                        │
    │  │  └─ Button: "Run Evaluation"                           │
    │  │                                                            │
    │  └─ Streamlit shows progress bar (0%)                      │
    │                                                                 │
    │  Step 2: Backend receives evaluation request               │
    │  ──────────────────────────────                             │
    │  HTTP POST /api/v1/evaluate/run                            │
    │  {                                                            │
    │    "dataset_id": "dataset_01ARZ3NHM...",                  │
    │    "model_id": "gemini-1.5-flash",                        │
    │    "run_config": {                                         │
    │      "collection_name": "hr_policies",                    │
    │      "top_k": 5,                                          │
    │      "temperature": 0.2                                  │
    │    },                                                        │
    │    "judge_config": {...}                                  │
    │  }                                                            │
    │                                                                 │
    │  Step 3: Create evaluation run record                       │
    │  ────────────────────────────                               │
    │  ├─ Generate run_id: ulid()                               │
    │  ├─ Write RunRecord to SQLite:                            │
    │  │  {                                                        │
    │  │    "run_id": "run_01ARZ3NHM...",                      │
    │  │    "dataset_id": "dataset_01ARZ3NHM...",              │
    │  │    "model_id": "gemini-1.5-flash",                    │
    │  │    "status": "IN_PROGRESS",                           │
    │  │    "started_at": "2026-06-26T05:35:00Z",             │
    │  │    "config": {...}                                    │
    │  │  }                                                        │
    │  │                                                            │
    │  └─ Return run_id to frontend for polling                 │
    │                                                                 │
    │  Step 4: For each QAPair in dataset (28 pairs):           │
    │  ───────────────────────────                               │
    │                                                                 │
    │  Iteration i (PAIR 1: "What is annual leave?"):           │
    │  ───────────────────────────────────────────              │
    │                                                                 │
    │  4.1: RAG PIPELINE EXECUTION                              │
    │  ─────────────────────────                                │
    │  src/rag/pipeline.py → answer():                          │
    │                                                                 │
    │  4.1.1: Embed question                                   │
    │         ├─ Question: "What is the annual leave..."       │
    │         ├─ EmbeddingManager.embed_query()               │
    │         └─ Output: 384-dim vector                       │
    │                                                                 │
    │  4.1.2: Retrieve from ChromaDB                           │
    │         ├─ Query ChromaDB with vector                  │
    │         ├─ top_k=5                                      │
    │         ├─ Returns 5 chunks:                           │
    │         │  Chunk 1: "Leave policy: 20 days..." (0.92) │
    │         │  Chunk 2: "Employee handbook..." (0.85)       │
    │         │  ... (3 more)                                │
    │         │  (similarity score in parentheses)           │
    │         │                                                 │
    │         └─ Output: RetrievedChunk[] objects            │
    │                                                                 │
    │  4.1.3: Build prompt                                    │
    │         ├─ System instruction:                         │
    │         │  "You are a precise Q&A assistant..."       │
    │         │                                                 │
    │         ├─ Context block (formatted):                 │
    │         │  "CONTEXT:                                  │
    │         │   1. [leave_policy.pdf:1] Leave policy...  │
    │         │   2. [handbook.pdf:5] Employee handbook... │
    │         │   3. [handbook.pdf:6] ...                  │
    │         │   4. ...                                    │
    │         │   5. ..."                                  │
    │         │                                                 │
    │         ├─ Question block:                            │
    │         │  "QUESTION: What is the annual leave...    │
    │         │   ANSWER:"                                 │
    │         │                                                 │
    │         └─ Full prompt (~1500 tokens)                 │
    │                                                                 │
    │  4.1.4: Generate answer via Gemini                      │
    │         ├─ Call Gemini (gemini-1.5-flash)             │
    │         │  ├─ prompt: full_prompt                    │
    │         │  ├─ temperature: 0.2                       │
    │         │  ├─ max_output_tokens: 500                │
    │         │  └─ Retry on 429                          │
    │         │                                                 │
    │         ├─ Response:                                   │
    │         │  "According to the policy, the annual     │
    │         │   leave entitlement is 20 days per        │
    │         │   year, as stated in leave_policy.pdf."   │
    │         │                                                 │
    │         ├─ Token counts:                              │
    │         │  ├─ input_tokens: 1450                     │
    │         │  └─ output_tokens: 52                     │
    │         │                                                 │
    │         └─ Latency: 1.2s                             │
    │                                                                 │
    │  4.1.5: Create RAGResponse                               │
    │         {                                                  │
    │           "question": "What is annual leave...",         │
    │           "generated_answer": "According to...",         │
    │           "retrieved_chunks": [5 chunks],               │
    │           "retrieval_latency_ms": 120,                  │
    │           "generation_latency_ms": 1200,               │
    │           "total_latency_ms": 1320,                    │
    │           "input_tokens": 1450,                        │
    │           "output_tokens": 52,                         │
    │           "estimated_cost_usd": 0.0008                │
    │         }                                                  │
    │                                                                 │
    │  4.2: EVALUATION ENGINE EXECUTION (Async Parallel)        │
    │  ──────────────────────────────                           │
    │  src/evaluation/engine.py → evaluate():                   │
    │                                                                 │
    │  Dispatch 4 concurrent judge tasks:                       │
    │  ────────────────────────────────────────                │
    │                                                                 │
    │  ┌─ Judge 1: FaithfulnessEvaluator (async)               │
    │  │  ├─ Prompt to Gemini:                                │
    │  │  │  "Given context and answer, is every claim     │
    │  │  │   grounded in the context?                     │
    │  │  │   Score 1-5."                                  │
    │  │  │                                                  │
    │  │  ├─ Input:                                         │
    │  │  │  context: [5 retrieved chunks]                │
    │  │  │  answer: "According to..."                    │
    │  │  │                                                  │
    │  │  ├─ Gemini reasoning (chain-of-thought):          │
    │  │  │  "The answer cites the leave policy directly.│
    │  │  │   All claims are grounded. Score: 5"         │
    │  │  │                                                  │
    │  │  └─ Output: MetricScore                           │
    │  │     {                                               │
    │  │       "metric": "faithfulness",                   │
    │  │       "score": 5,                                 │
    │  │       "reasoning": "The answer cites..."         │
    │  │     }                                               │
    │  │                                                      │
    │  ├─ Judge 2: AnswerRelevanceEvaluator (async)            │
    │  │  ├─ Question: Does answer address the question?  │
    │  │  ├─ Gemini reasoning:                             │
    │  │  │  "The question asks about annual leave. The  │
    │  │  │   answer directly provides the number (20     │
    │  │  │   days). Highly relevant. Score: 5"          │
    │  │  │                                                  │
    │  │  └─ Output:                                        │
    │  │     {"metric": "answer_relevance", "score": 5}   │
    │  │                                                      │
    │  ├─ Judge 3: ContextPrecisionEvaluator (async)           │
    │  │  ├─ Question: Are retrieved docs useful?         │
    │  │  ├─ Analysis:                                     │
    │  │  │  Chunk 1: Very relevant (directly answers)    │
    │  │  │  Chunk 2: Moderately relevant (company wide)  │
    │  │  │  Chunks 3-5: Less relevant                    │
    │  │  │  Precision: 2 out of 5 = 40%                 │
    │  │  │  Score: 3                                     │
    │  │  │                                                  │
    │  │  └─ Output:                                        │
    │  │     {"metric": "context_precision", "score": 3}  │
    │  │                                                      │
    │  └─ Judge 4: CorrectnessEvaluator (async)                │
    │     ├─ Ground truth answer: "20 days annual leave"  │
    │     ├─ Generated answer: "According to... 20 days..."   │
    │     ├─ Match? YES                                     │
    │     ├─ Confidence: High                              │
    │     └─ Output:                                        │
    │        {"metric": "correctness", "score": 5}        │
    │                                                                 │
    │  await asyncio.gather(*[judge1, judge2, judge3, judge4])  │
    │  → All 4 complete in parallel (fastest ≈ 1.5s total)     │
    │                                                                 │
    │  4.3: Aggregate Results & Calculate Composite Score       │
    │  ─────────────────────────────────────                    │
    │  src/evaluation/schema.py:                                │
    │                                                                 │
    │  Composite Score Formula:                                 │
    │  = (0.30 × faithfulness)                                 │
    │    + (0.25 × answer_relevance)                           │
    │    + (0.25 × context_precision)                          │
    │    + (0.20 × correctness)                                │
    │  = (0.30 × 5) + (0.25 × 5) + (0.25 × 3) + (0.20 × 5)  │
    │  = 1.5 + 1.25 + 0.75 + 1.0                              │
    │  = 4.5 / 5.0 = 90% (Excellent)                          │
    │                                                                 │
    │  4.4: Create EvalResult                                   │
    │  ────────────────────────                                │
    │  {                                                            │
    │    "pair_id": "pair_1",                                 │
    │    "rag_response": {RAGResponse object},               │
    │    "metric_scores": {                                  │
    │      "faithfulness": {"score": 5, "reasoning": "..."},  │
    │      "answer_relevance": {"score": 5, "reasoning": "..."}, │
    │      "context_precision": {"score": 3, "reasoning": "..."}, │
    │      "correctness": {"score": 5, "reasoning": "..."}     │
    │    },                                                      │
    │    "composite_score": 4.5,                             │
    │    "quality_band": "Excellent",                       │
    │    "total_time_ms": 2520                              │
    │  }                                                          │
    │                                                                 │
    │  4.5: Persist to SQLite                                  │
    │  ────────────────────                                   │
    │  INSERT INTO eval_results                              │
    │  (run_id, pair_id, faithfulness_score,                │
    │   answer_relevance_score, context_precision_score,    │
    │   correctness_score, composite_score,                 │
    │   reasoning_json, total_time_ms)                      │
    │  VALUES (...)                                           │
    │                                                                 │
    │  Step 5: Repeat for remaining 27 pairs                   │
    │  ─────────────────────────                               │
    │  ├─ PAIR 2: Execute same flow                          │
    │  ├─ PAIR 3: Execute same flow                          │
    │  │  ...                                                 │
    │  └─ PAIR 28: Execute same flow                         │
    │                                                                 │
    │  Async concurrency within pairs:                         │
    │  ├─ All 4 judges run in parallel per pair              │
    │  ├─ Pairs process sequentially (28 iterations)         │
    │  └─ Total time ≈ 28 pairs × 2.5s/pair = 70s           │
    │                                                                 │
    │  Step 6: Calculate batch aggregate statistics            │
    │  ────────────────────────────                            │
    │  src/comparison/aggregator.py:                           │
    │                                                                 │
    │  Across all 28 pairs:                                    │
    │  ├─ Average faithfulness: 4.7 / 5.0 (94%)              │
    │  ├─ Average answer_relevance: 4.6 / 5.0 (92%)          │
    │  ├─ Average context_precision: 3.8 / 5.0 (76%)         │
    │  ├─ Average correctness: 4.9 / 5.0 (98%)               │
    │  ├─ Average composite: 4.5 / 5.0 (90%)                 │
    │  ├─ Total latency: 70,000 ms                           │
    │  ├─ Total tokens: 45,320 input, 1,456 output          │
    │  └─ Total cost: $0.0234                                │
    │                                                                 │
    │  Step 7: Create RAGBatchResult                            │
    │  ──────────────────────────                              │
    │  {                                                            │
    │    "run_id": "run_01ARZ3NHM...",                        │
    │    "dataset_name": "hr_eval_v1",                        │
    │    "rag_model": "gemini-1.5-flash",                     │
    │    "collection_name": "hr_policies",                    │
    │    "top_k": 5,                                          │
    │    "temperature": 0.2,                                 │
    │    "total_pairs": 28,                                  │
    │    "answered_pairs": 28,                               │
    │    "failed_pairs": 0,                                  │
    │    "avg_faithfulness": 4.7,                            │
    │    "avg_answer_relevance": 4.6,                        │
    │    "avg_context_precision": 3.8,                       │
    │    "avg_correctness": 4.9,                             │
    │    "avg_composite": 4.5,                               │
    │    "total_latency_ms": 70000,                          │
    │    "avg_latency_ms": 2500,                             │
    │    "total_tokens": {                                   │
    │      "input": 45320,                                   │
    │      "output": 1456                                    │
    │    },                                                      │
    │    "total_cost_usd": 0.0234,                           │
    │    "completed_at": "2026-06-26T05:37:10Z"             │
    │  }                                                          │
    │                                                                 │
    │  Step 8: Update run status in SQLite                     │
    │  ─────────────────────────                               │
    │  UPDATE runs                                             │
    │  SET status='COMPLETED',                               │
    │      results_summary={...},                            │
    │      completed_at=NOW()                                │
    │  WHERE run_id='run_01ARZ3NHM...'                       │
    │                                                                 │
    │  Step 9: Stream results back to Streamlit               │
    │  ──────────────────────────                             │
    │  GET /api/v1/evaluate/{run_id}                         │
    │  → Returns full EvalResult[] for all 28 pairs         │
    │                                                                 │
    │  Step 10: Streamlit Dashboard Display                   │
    │  ──────────────────────────────                         │
    │  ├─ Summary cards:                                      │
    │  │  ├─ Overall score: 4.5/5.0 (90%) 🟢                │
    │  │  ├─ Total time: 70s                                │
    │  │  ├─ Cost: $0.0234                                  │
    │  │  └─ Success rate: 28/28 (100%)                    │
    │  │                                                        │
    │  ├─ Metric breakdown table:                            │
    │  │  Metric               │ Avg  │ Min │ Max │ Std  │  │
    │  │  Faithfulness         │ 4.7  │ 4.0 │ 5.0 │ 0.32 │  │
    │  │  Answer Relevance     │ 4.6  │ 3.5 │ 5.0 │ 0.45 │  │
    │  │  Context Precision    │ 3.8  │ 2.0 │ 5.0 │ 0.88 │  │
    │  │  Correctness          │ 4.9  │ 4.0 │ 5.0 │ 0.28 │  │
    │  │                                                        │
    │  ├─ Distribution histograms:                           │
    │  │  [Plotly histograms for each metric]               │
    │  │                                                        │
    │  ├─ Drilldown table (for each pair):                  │
    │  │  Q & A, Scores, Reasoning (expandable)             │
    │  │                                                        │
    │  └─ Export button:                                     │
    │     "Download as CSV"                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │     BONUS: MULTI-MODEL COMPARISON WORKFLOW                     │
    │     (User clicks "Compare Models" page)                        │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Step 1: User defines comparison matrix                       │
    │  ────────────────────────                                     │
    │  ├─ RAG models: [gemini-1.5-flash, gpt-4, claude-3]         │
    │  ├─ Retrieval top_k: [3, 5, 10]                             │
    │  ├─ Temperature: [0.0, 0.5, 1.0]                            │
    │  ├─ Result: 3 × 3 × 3 = 27 configurations                  │
    │  └─ Button: "Run Comparison"                               │
    │                                                                 │
    │  Step 2: Backend queues comparison runs                       │
    │  ──────────────────────                                       │
    │  src/comparison/runner.py:                                    │
    │  ├─ MAX_CONCURRENT_RUNS = 3 (respects rate limits)          │
    │  ├─ Queue 27 tasks                                           │
    │  └─ Execute in batches of 3                                 │
    │                                                                 │
    │  Step 3: For each configuration (27 iterations):             │
    │  ───────────────────────────                                 │
    │  ├─ Run RAGPipeline.answer_dataset() with config           │
    │  ├─ Run EvaluationEngine on all pairs                      │
    │  ├─ Collect RAGBatchResult                                 │
    │  └─ Store in DB                                            │
    │                                                                 │
    │  Step 4: Aggregate across all configurations                 │
    │  ─────────────────────                                       │
    │  src/comparison/aggregator.py:                              │
    │  ├─ For each model:                                         │
    │  │  ├─ Average scores across all (top_k, temp) combos     │
    │  │  ├─ Calculate cost efficiency                          │
    │  │  └─ Rank by composite score                           │
    │  │                                                            │
    │  └─ Result: ComparisonMatrix                               │
    │     {                                                         │
    │       "gemini-1.5-flash": {                                 │
    │         "avg_score": 4.5,                                  │
    │         "cost": $0.0234,                                   │
    │         "latency": 2500,                                   │
    │         "rank": 1                                          │
    │       },                                                      │
    │       "gpt-4": {                                            │
    │         "avg_score": 4.7,                                  │
    │         "cost": $0.12,                                     │
    │         "latency": 1800,                                   │
    │         "rank": 2                                          │
    │       },                                                      │
    │       "claude-3": {                                         │
    │         "avg_score": 4.3,                                  │
    │         "cost": $0.15,                                     │
    │         "latency": 2100,                                   │
    │         "rank": 3                                          │
    │       }                                                        │
    │     }                                                          │
    │                                                                 │
    │  Step 5: Visualize comparison                                │
    │  ───────────────────────                                     │
    │  ├─ Radar chart:                                            │
    │  │  Each axis = metric                                     │
    │  │  Each shape = model                                    │
    │  │  Overlay all 3 to compare                             │
    │  │                                                            │
    │  ├─ Bar chart:                                              │
    │  │  Model vs. composite score                            │
    │  │  Color-coded by rank                                  │
    │  │                                                            │
    │  ├─ Scatter plot:                                           │
    │  │  X-axis: Cost (USD)                                   │
    │  │  Y-axis: Quality (composite score)                    │
    │  │  Bubble size: Latency                                │
    │  │                                                            │
    │  └─ Table:                                                  │
    │     Model | Score | Cost  | Latency | Rank              │
    │     ───────────────────────────────────────────          │
    │     GPT4  │ 4.7   │ 0.12  │ 1800ms  │ #1                │
    │     Gemini│ 4.5   │ 0.02  │ 2500ms  │ #2                │
    │     Claude│ 4.3   │ 0.15  │ 2100ms  │ #3                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

```
┌─────────────────────────────────────────────────┐
│        ERROR DETECTION & RECOVERY               │
├─────────────────────────────────────────────────┤
│                                                 │
│  Scenario 1: API Rate Limit (429)              │
│  ────────────────────────                      │
│  ├─ Detected: Gemini returns 429               │
│  ├─ tenacity.retry triggers                   │
│  ├─ Wait: exponential backoff                  │
│  │  ├─ Attempt 1: wait 2s                     │
│  │  ├─ Attempt 2: wait 4s                     │
│  │  ├─ Attempt 3: wait 8s                     │
│  │  └─ Attempt 4: wait 16s                    │
│  ├─ Log: "Rate limited, retrying in 2s"      │
│  └─ Resume: Continue evaluation               │
│                                                 │
│  Scenario 2: Missing Chunk in Retrieval       │
│  ────────────────────────────                  │
│  ├─ Detected: ChromaDB query returns 0         │
│  ├─ Flag: empty_context=True                   │
│  ├─ Generate: Answer without context           │
│  ├─ Log: "No context found, answering anyway"  │
│  ├─ Score: Faithfulness will be low           │
│  └─ Continue: Still evaluate (don't fail)      │
│                                                 │
│  Scenario 3: Invalid Embedding Vector        │
│  ─────────────────────────                    │
│  ├─ Detected: EmbeddingManager returns None   │
│  ├─ Raise: EmbeddingError                      │
│  ├─ Catch: In api/routes/evaluate.py          │
│  ├─ Return: HTTP 500 + error message          │
│  ├─ Log: Full exception traceback              │
│  └─ Frontend: Show error toast notification    │
│                                                 │
│  Scenario 4: Corrupted JSON Response          │
│  ────────────────────────                     │
│  ├─ Detected: json.JSONDecodeError            │
│  ├─ Fallback: Return raw string               │
│  ├─ Log: "Could not parse JSON, using raw"    │
│  ├─ Mark: quality_flag="UNPARSED"             │
│  └─ Continue: Proceed with degraded result    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Database Persistence Flow

```
┌──────────────────────────────────────────────────────────┐
│         SQLite Schema & CRUD Operations                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Tables:                                                │
│  ├─ datasets                                            │
│  │  ├─ id (UUID)                                       │
│  │  ├─ name (str)                                      │
│  │  ├─ collection_name (str)                           │
│  │  ├─ created_at (timestamp)                          │
│  │  └─ num_pairs (int)                                 │
│  │                                                       │
│  ├─ runs                                                │
│  │  ├─ run_id (UUID)                                   │
│  │  ├─ dataset_id (FK)                                 │
│  │  ├─ model_id (str)                                  │
│  │  ├─ status (enum: IN_PROGRESS, COMPLETED, FAILED)   │
│  │  ├─ started_at (timestamp)                          │
│  │  ├─ completed_at (timestamp)                        │
│  │  └─ config_json (JSON)                              │
│  │                                                       │
│  ├─ eval_results                                        │
│  │  ├─ result_id (UUID)                                │
│  │  ├─ run_id (FK)                                     │
│  │  ├─ pair_id (str)                                   │
│  │  ├─ faithfulness_score (float)                      │
│  │  ├─ answer_relevance_score (float)                  │
│  │  ├─ context_precision_score (float)                 │
│  │  ├─ correctness_score (float)                       │
│  │  ├─ composite_score (float)                         │
│  │  ├─ reasoning_json (JSON)                           │
│  │  └─ created_at (timestamp)                          │
│  │                                                       │
│  └─ comparison_matrices                                 │
│     ├─ matrix_id (UUID)                                │
│     ├─ dataset_id (FK)                                 │
│     ├─ runs_config (JSON)                              │
│     ├─ aggregated_results (JSON)                       │
│     └─ created_at (timestamp)                          │
│                                                          │
│  Operations:                                            │
│  ├─ CREATE: Insert new run record                      │
│  ├─ READ: Query history, fetch run details             │
│  ├─ UPDATE: Mark as completed, update status           │
│  ├─ DELETE: Soft-delete old runs (archive)            │
│  └─ EXPORT: Stream to CSV for external analysis        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

This comprehensive workflow diagram covers all three main journeys and bonus features. Each step is traced end-to-end with actual API calls, database operations, and async patterns.

**Your project architecture is production-grade with clear separation of concerns, dependency injection, abstract interfaces, and full observability.**
