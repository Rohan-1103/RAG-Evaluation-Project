"""
scripts/seed_on_startup.py

Run automatically on Render startup to re-ingest a small demo document
so the app is immediately usable after a cold deploy without manual steps.

Called from Dockerfile's CMD only in production (APP_ENV=production).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def seed_if_empty() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from config import get_settings
    settings = get_settings()

    from src.vectorstore.embeddings import EmbeddingManager
    from src.vectorstore.chroma import ChromaVectorStore

    em = EmbeddingManager(settings.embedding)
    vs = ChromaVectorStore(settings.chroma, em)
    collections = vs.list_collections()

    if collections:
        print(f"seed_on_startup: {len(collections)} collection(s) found — skipping seed.")
        return

    print("seed_on_startup: No collections found — seeding demo data...")

    # Write a minimal demo document
    demo_dir = Path("data/raw_docs")
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_file = demo_dir / "about_rag_eval_bench.txt"
    demo_file.write_text("""
RAG Evaluation Benchmarking Tool

This tool evaluates Retrieval-Augmented Generation (RAG) pipelines
using LLM-as-a-Judge methodology across four metrics:

1. Faithfulness — Is every claim grounded in the retrieved context?
2. Answer Relevance — Does the answer address the question asked?
3. Context Precision — How much of the retrieved context was useful?
4. Correctness — Does the answer match the ground truth reference?

The system supports multiple LLM providers: Google Gemini, Groq, and
OpenRouter, enabling cross-provider comparisons in a single run.

Embeddings are generated locally using all-MiniLM-L6-v2 from
HuggingFace, requiring no external embedding API calls.
Vector storage uses ChromaDB with persistent local storage.
Run history is persisted in SQLite via SQLAlchemy async ORM.
    """, encoding="utf-8")

    from src.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline.build_default(
        settings.ingestion,
        settings.embedding,
        settings.chroma,
    )
    result = pipeline.ingest_files(
        files=[demo_file],
        collection_name="demo",
    )
    print(f"seed_on_startup: Ingested {result.total_chunks_stored} chunks into 'demo' collection.")


if __name__ == "__main__":
    seed_if_empty()