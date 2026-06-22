"""
ui/app.py

Streamlit application entrypoint.

Run with:
    streamlit run ui/app.py

Architecture rule enforced here and in every page under ui/pages/:
this layer NEVER imports from src/ or constructs anything from the
business logic layer directly — every action goes through
ui.components.api_client.APIClient, which talks to the FastAPI server
started separately via:
    uvicorn src.api.app:create_app --factory --port 8000

Top-level code in this file (render_sidebar(), page config) executes
on every single rerun AND every navigation, because Streamlit re-runs
this entrypoint script top-to-bottom each time — only the content
inside the currently-selected st.Page changes underneath it. This is
what makes the sidebar "shared chrome" without needing to duplicate
its rendering call inside every individual page file.

home() is defined inline here (not as a separate ui/pages/00_home.py
file) since st.Page() accepts a plain function reference, and the
landing dashboard is lightweight enough that a dedicated file would
be unnecessary indirection for what's fundamentally part of the app
shell, not a workflow step.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Anchor project root onto sys.path BEFORE any local import, exactly
# like config/settings.py's _ENV_FILE pattern — `streamlit run ui/app.py`
# does not reliably put the project root (parent of ui/) on sys.path,
# only the script's own directory, which would otherwise break every
# `from config...` / `from src...` / `from ui...` import below.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.api_client import APIError, get_api_client
from ui.components.sidebar import render_sidebar

st.set_page_config(
    page_title="RAG Eval Bench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

def home() -> None:
    """
    Landing dashboard: quick stats pulled live from the backend, plus
    a 4-step workflow guide linking to the corresponding page.

    Every backend call below is individually wrapped — one endpoint
    being unreachable (e.g. zero collections ingested yet, a fresh
    install) should degrade that one stat card to "—", not blank the
    entire dashboard.
    """
    st.title("📊 RAG Evaluation Benchmarking Tool")
    st.caption(
        "Production-grade RAG evaluation with LLM-as-a-Judge scoring "
        "across Faithfulness, Answer Relevance, Context Precision, "
        "and Correctness."
    )

    client = get_api_client()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            collections = client.list_collections()
            st.metric("Collections", collections["total_collections"])
        except Exception:
            st.metric("Collections", "—")

    with col2:
        try:
            collections = client.list_collections()
            st.metric("Documents indexed", collections["total_documents"])
        except Exception:
            st.metric("Documents indexed", "—")

    with col3:
        try:
            datasets = client.list_datasets(limit=1)
            st.metric("Datasets", datasets["total"])
        except Exception:
            st.metric("Datasets", "—")

    with col4:
        try:
            runs = client.list_runs(limit=1)
            st.metric("Evaluation runs", runs["total"])
        except Exception:
            st.metric("Evaluation runs", "—")

    st.divider()
    st.subheader("Workflow")

    steps = [
        ("📄", "1. Ingest", "Upload documents into a ChromaDB collection.", "ingest"),
        ("🧪", "2. Generate Dataset", "Create synthetic Q&A pairs from your documents.", "dataset"),
        ("⚖️", "3. Evaluate", "Run RAG + LLM-as-a-Judge scoring on one model.", "evaluate"),
        ("📊", "4. Compare", "Benchmark multiple models side-by-side.", "compare"),
    ]

    cols = st.columns(4)
    for col, (icon, title, desc, page_key) in zip(cols, steps):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(desc)

    st.divider()
    st.subheader("About this tool")
    st.markdown(
        """
        This benchmark evaluates RAG pipelines using **LLM-as-a-Judge**
        across four metrics:

        - **Faithfulness** — Is every claim grounded in retrieved context?
        - **Answer Relevance** — Does the answer address the question?
        - **Context Precision** — Was retrieval signal-to-noise good?
        - **Correctness** — Does the answer match the ground truth?

        All scores include the judge's full reasoning — not just a number.
        """
    )

pg = st.navigation(
    [
        st.Page(home, title="Home", icon="🏠", default=True),
        st.Page("pages/01_ingest.py", title="Ingest Documents", icon="📄"),
        st.Page("pages/02_dataset.py", title="Generate Dataset", icon="🧪"),
        st.Page("pages/03_evaluate.py", title="Run Evaluation", icon="⚖️"),
        st.Page("pages/04_compare.py", title="Compare Models", icon="📊"),
        st.Page("pages/05_history.py", title="Run History", icon="🕒"),
    ]
)

pg.run()

__all__ = ["home"]