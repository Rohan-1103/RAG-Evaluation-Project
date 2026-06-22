"""
ui/pages/03_evaluate.py

STUB — full implementation pending. Will provide: single-model run
configuration (dataset, model, top_k, temperature), live progress
during RAG + judge scoring, and the per-question drilldown table with
judge reasoning.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("⚖️ Run Evaluation")
st.info(
    "🚧 Coming soon: run RAG + LLM-as-a-Judge on a dataset and inspect "
    "per-question scores with full judge reasoning."
)