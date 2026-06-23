"""
ui/pages/01_ingest.py

Document ingestion page — upload files into a ChromaDB collection,
browse existing collections, and delete them.

Talks exclusively to the FastAPI backend via ui.components.api_client.
APIClient — see that module's docstring for why this is a hard
architecture rule, not a style preference. Every action here maps
directly to one of src/api/routes/ingest.py's endpoints:

    Upload form          -> POST /api/v1/ingest/files
    Collections table     -> GET  /api/v1/ingest/collections
    "View" expander        -> GET  /api/v1/ingest/collections/{name}
    "Delete" button         -> DELETE /api/v1/ingest/collections/{name}
    Supported formats caption -> GET /api/v1/ingest/supported-formats

Why st.session_state holds the last ingestion result rather than
re-rendering it inline immediately after the POST call:
  Streamlit reruns the entire script top-to-bottom on every widget
  interaction. If the result table were rendered only inside the
  `if submitted:` block, clicking ANY other widget on the page
  afterward (e.g. expanding a collection's details, or just the
  natural rerun Streamlit performs after any interaction) would wipe
  it from view, even though the ingestion itself already succeeded
  and the data is sitting safely in the backend. Storing it in
  session_state and rendering it unconditionally on every rerun keeps
  "what just happened" visible exactly until the user does something
  that should replace it (a new ingestion).

Why the collections table is fetched fresh on every rerun with no
caching:
  Collection state (document_count especially) changes as a direct
  result of actions taken ON THIS PAGE (ingesting more files, deleting
  a collection). Caching it would mean the user has to manually
  refresh to see the effect of their own last action — the one thing
  a page like this must never get wrong. The health check's
  st.cache_resource pattern in sidebar.py is the wrong tool here
  specifically because that data does NOT change as a result of user
  actions on the page it's rendered on.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
from typing import Any

import pandas as pd
import streamlit as st

from ui.components.api_client import APIError, get_api_client
from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("📄 Ingest Documents")
st.caption(
    "Upload documents into a ChromaDB collection. Re-uploading a file "
    "with the same name to the same collection updates its chunks in "
    "place rather than duplicating them."
)

client = get_api_client()

# Mirrors src/api/routes/ingest.py's _COLLECTION_NAME_PATTERN exactly —
# duplicated here (not imported from src/) so this page never imports
# from src/ directly, only from ui.components.api_client. Validating
# client-side gives an instant inline error instead of a round trip to
# the backend just to learn the name was malformed.
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")


# ===========================================================================
# UPLOAD FORM
# ===========================================================================

st.subheader("Upload files")

try:
    formats = client.supported_formats()
    extensions = formats.get("extensions", [])
    st.caption(f"Supported formats: {', '.join(f'.{e}' for e in extensions)}")
except APIError as exc:
    extensions = []
    st.caption("Supported formats: (could not reach backend to confirm)")

with st.form("ingest_form", clear_on_submit=False):
    col1, col2 = st.columns([3, 1])

    with col1:
        collection_name = st.text_input(
            "Collection name",
            placeholder="e.g. hr_policies, q3_report",
            help=(
                "3-63 characters: letters, numbers, underscores, "
                "hyphens. Must start and end with a letter or number."
            ),
        )

    with col2:
        upsert = st.checkbox(
            "Upsert",
            value=True,
            help=(
                "If checked, re-uploading identical content updates "
                "existing chunks in place. If unchecked, duplicate "
                "chunks are skipped rather than updated."
            ),
        )

    uploaded_files = st.file_uploader(
        "Files",
        accept_multiple_files=True,
        type=extensions if extensions else None,
        help="Select one or more files to ingest.",
    )

    submitted = st.form_submit_button("Ingest", type="primary")

if submitted:
    if not collection_name:
        st.error("Collection name is required.")
    elif not _COLLECTION_NAME_PATTERN.match(collection_name):
        st.error(
            "Invalid collection name. Must be 3-63 characters, "
            "letters/numbers/underscores/hyphens only, and start/end "
            "with a letter or number."
        )
    elif not uploaded_files:
        st.error("Select at least one file to upload.")
    else:
        files_payload = [(f.name, f.getvalue()) for f in uploaded_files]
        with st.spinner(
            f"Ingesting {len(files_payload)} file(s) into "
            f"'{collection_name}'... this may take a moment."
        ):
            try:
                result = client.ingest_files(
                    collection_name=collection_name,
                    files=files_payload,
                    upsert=upsert,
                )
                st.session_state["last_ingestion_result"] = result
            except APIError as exc:
                st.error(f"Ingestion failed: {exc.detail}")
                st.session_state.pop("last_ingestion_result", None)


# ===========================================================================
# LAST INGESTION RESULT — see module docstring for why this lives in
# session_state and renders unconditionally on every rerun
# ===========================================================================

if "last_ingestion_result" in st.session_state:
    result: dict[str, Any] = st.session_state["last_ingestion_result"]

    st.divider()
    st.subheader(f"Result — collection '{result['collection_name']}'")

    if result["failed_files"] == 0:
        st.success(
            f"✅ {result['succeeded_files']}/{result['total_files_attempted']} "
            f"file(s) ingested successfully."
        )
    elif result["succeeded_files"] > 0:
        st.warning(
            f"⚠️ {result['succeeded_files']}/{result['total_files_attempted']} "
            f"file(s) succeeded, {result['failed_files']} failed."
        )
    else:
        st.error(
            f"❌ All {result['total_files_attempted']} file(s) failed to ingest."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents loaded", result["total_documents_loaded"])
    m2.metric("Chunks produced", result["total_chunks_produced"])
    m3.metric("Chunks stored", result["total_chunks_stored"])
    m4.metric("Latency", f"{result['total_latency_ms']:.0f} ms")

    files_df = pd.DataFrame(
        [
            {
                "File": f["filename"],
                "Status": f["status"],
                "Loader": f["loader_class"] or "—",
                "Documents": f["document_count"],
                "Chunks": f["chunk_count"],
                "Stored": f["added_to_store"],
                "Latency (ms)": round(f["latency_ms"], 1),
                "Error": f["error_message"] or "",
            }
            for f in result["files"]
        ]
    )

    def _status_color(status_value: str) -> str:
        return {
            "success": "background-color: #d4edda",
            "failed": "background-color: #f8d7da",
            "skipped": "background-color: #fff3cd",
        }.get(status_value, "")

    st.dataframe(
        files_df.style.applymap(_status_color, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear result"):
        del st.session_state["last_ingestion_result"]
        st.rerun()


# ===========================================================================
# COLLECTIONS BROWSER
# ===========================================================================

st.divider()
st.subheader("Existing collections")

try:
    collections_resp = client.list_collections()
except APIError as exc:
    st.error(f"Could not load collections: {exc.detail}")
    collections_resp = None

if collections_resp is not None:
    collections = collections_resp["collections"]

    if not collections:
        st.info("No collections yet. Upload files above to create one.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total collections", collections_resp["total_collections"])
        c2.metric("Total documents", collections_resp["total_documents"])

        for col in collections:
            with st.expander(
                f"📁 **{col['name']}** — {col['document_count']} chunks",
                expanded=False,
            ):
                left, right = st.columns([3, 1])

                with left:
                    st.text(f"Embedding dimension: {col['embedding_dimension'] or '—'}")
                    model_name = col.get("metadata", {}).get("embedding_model")
                    if model_name:
                        st.text(f"Embedding model: {model_name}")
                    if col["is_empty"]:
                        st.caption("This collection is currently empty.")

                with right:
                    delete_key = f"delete_confirm_{col['name']}"
                    if st.session_state.get(delete_key, False):
                        st.warning("Delete permanently?")
                        confirm_col, cancel_col = st.columns(2)
                        with confirm_col:
                            if st.button(
                                "Yes, delete",
                                key=f"confirm_{col['name']}",
                                type="primary",
                            ):
                                try:
                                    client.delete_collection(col["name"])
                                    st.session_state.pop(delete_key, None)
                                    st.success(f"Deleted '{col['name']}'.")
                                    st.rerun()
                                except APIError as exc:
                                    st.error(f"Delete failed: {exc.detail}")
                        with cancel_col:
                            if st.button("Cancel", key=f"cancel_{col['name']}"):
                                st.session_state.pop(delete_key, None)
                                st.rerun()
                    else:
                        if st.button(
                            "🗑️ Delete", key=f"delete_{col['name']}"
                        ):
                            st.session_state[delete_key] = True
                            st.rerun()