"""
ui/components/sidebar.py

Shared sidebar rendered identically on every page. Code here runs at
the top level of ui/app.py (not inside any st.Page function), which
in Streamlit's execution model means it persists across navigation —
the same script file (app.py) re-executes top-to-bottom on every
rerun and every page switch; only the selected st.Page's own content
changes underneath it.
"""

from __future__ import annotations

import streamlit as st

from ui.components.api_client import APIError, get_api_client


def render_sidebar() -> None:
    """Render the persistent sidebar: branding, live health status, quick links."""
    with st.sidebar:
        st.markdown("## 📊 RAG Eval Bench")
        st.caption("LLM-as-a-Judge benchmarking tool")
        st.divider()

        _render_health_status()

        st.divider()
        st.caption("Built with FastAPI + Streamlit + Gemini")


def _render_health_status() -> None:
    """
    Live backend health indicator.

    Calls GET /health on every sidebar render (i.e. every page load/
    rerun) — deliberately cheap per src/api/app.py's health_check
    docstring: it checks the DB connection only, never makes a live
    Gemini call, so polling it on every rerun costs no API quota.
    """
    client = get_api_client()
    try:
        health = client.health()
    except Exception:
        st.error("🔴 Backend unreachable")
        st.caption(
            "Start the API server: `uvicorn src.api.app:create_app "
            "--factory --port 8000`"
        )
        return

    status = health.get("status", "unknown")
    if status == "healthy":
        st.success("🟢 Backend connected")
    elif status == "degraded":
        st.warning("🟡 Backend degraded")
    else:
        st.info(f"⚪ Backend status: {status}")

    with st.expander("System info", expanded=False):
        st.text(f"Embedding model: {health.get('embedding_model', '—')}")
        st.text(f"Embedding dim: {health.get('embedding_dim', '—')}")
        metrics = health.get("evaluation_metrics", [])
        st.text(f"Metrics: {', '.join(metrics) if metrics else '—'}")


__all__ = ["render_sidebar"]