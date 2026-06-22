"""
ui/pages/05_history.py

STUB — full implementation pending. Will provide: browsable run/
comparison history tables, filtering, and CSV export of historical
results.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("🕒 Run History")
st.info(
    "🚧 Coming soon: browse all past evaluation runs and comparisons, "
    "filter by model/dataset, and re-export results."
)