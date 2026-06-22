"""
ui/pages/02_dataset.py

STUB — full implementation pending. Will provide: dataset generation
form (collection -> sample chunks -> Gemini Q&A pairs), dataset
browser/list, and an inline pair editor (question/ground truth).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("🧪 Generate Test Dataset")
st.info(
    "🚧 Coming soon: generate synthetic Q&A pairs from a collection "
    "and preview/edit them before running evaluation."
)