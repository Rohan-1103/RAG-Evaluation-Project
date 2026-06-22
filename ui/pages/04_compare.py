"""
ui/pages/04_compare.py

STUB — full implementation pending. Will provide: multi-model grid
configuration, the radar chart + bar chart + comparison table, and
CSV export of the comparison matrix.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("📊 Compare Models")
st.info(
    "🚧 Coming soon: compare multiple RAG models side-by-side with "
    "radar charts, latency/cost tables, and ranked results."
)