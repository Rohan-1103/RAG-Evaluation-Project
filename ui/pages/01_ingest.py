"""
ui/pages/01_ingest.py

STUB — full implementation pending. Will provide: file upload form,
collection name input, chunking config, live ingestion progress, and
the collections browser table (list/view/delete).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("📄 Ingest Documents")
st.info(
    "🚧 Coming soon: upload PDFs/TXT/HTML/DOCX, configure chunking, "
    "and browse existing ChromaDB collections."
)