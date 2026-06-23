"""
ui/pages/02_dataset.py

Dataset generation page — sample chunks from a collection, generate
synthetic Q&A pairs via Gemini, browse existing datasets, and edit or
delete individual pairs before evaluation.

Maps directly onto src/api/routes/datasets.py's endpoints:

    Generate form       -> POST   /api/v1/datasets/generate
    Dataset browser      -> GET    /api/v1/datasets
    "View" detail panel    -> GET    /api/v1/datasets/{dataset_id}
    Pair edit form         -> PATCH  /api/v1/datasets/{dataset_id}/pairs/{pair_id}
    Pair delete            -> DELETE /api/v1/datasets/{dataset_id}/pairs/{pair_id}
    "Export CSV" button     -> GET    /api/v1/datasets/{dataset_id}/export.csv
    "Delete dataset" button -> DELETE /api/v1/datasets/{dataset_id}

Why the "source collection" field is a selectbox populated from
GET /api/v1/ingest/collections, not a free-text input:

  GenerateDatasetRequest.collection_name must reference a collection
  that genuinely exists in ChromaDB — _sample_chunks_sync() in
  src/api/routes/datasets.py calls vector_store.query() against it
  directly, and an unrecognised name surfaces as a 404 only AFTER the
  user has already filled in every other field and clicked Generate.
  A selectbox sourced from the live collections list makes that whole
  class of error structurally impossible from this page, the same way
  ingest.py's _COLLECTION_NAME_PATTERN catches a malformed name before
  any backend round trip — different mechanism, same goal.

Why editing/deleting a pair is only OFFERED in the UI for
status == "pending" (rather than always shown, then erroring on
submit):

  PATCH /pairs/{pair_id} already enforces this server-side with a 409
  (see edit_pair's docstring in src/api/routes/datasets.py — "Only
  PENDING pairs ... can be edited"). Showing a disabled-looking,
  read-only view for already-answered/evaluated pairs instead of an
  editable form that would just bounce off a 409 on submit is a small
  but real UX difference: the constraint is communicated by what's
  rendered, not discovered by clicking and getting an error.

Why "selected_dataset_id" lives in st.session_state rather than this
page re-fetching+rendering full detail for every dataset in the list
simultaneously:

  A dataset can hold up to 100 pairs (GeneratorConfig.max_pairs_total's
  cap). Rendering every dataset's full pair list at once on a page
  that also lists N datasets would mean N x (up to 100) pair rows on
  screen simultaneously. Storing exactly one "currently expanded"
  dataset_id and fetching ITS full detail on demand keeps this page's
  rendering cost proportional to "one dataset's pairs", not "every
  dataset's pairs combined" — same proportional-cost reasoning behind
  DatasetStore's own metadata.json/dataset.json split on the backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any

import pandas as pd
import streamlit as st

from ui.components.api_client import APIError, get_api_client
from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("🧪 Generate Test Dataset")
st.caption(
    "Sample chunks from a collection and generate synthetic "
    "question-answer pairs via Gemini, ready to run through evaluation."
)

client = get_api_client()

_STATUS_OPTIONS = ["All", "draft", "ready", "running", "completed", "partial"]
_METHOD_OPTIONS = ["All", "synthetic", "manual", "imported", "mixed"]
_SORT_OPTIONS = ["created_at", "updated_at", "name", "total_pairs", "completion_rate"]


# ===========================================================================
# GENERATE FORM
# ===========================================================================

st.subheader("Generate a new dataset")

try:
    collections_resp = client.list_collections()
    collection_names = [c["name"] for c in collections_resp["collections"]]
except APIError:
    collection_names = []

if not collection_names:
    st.warning(
        "No collections found. Ingest documents first before "
        "generating a dataset."
    )
    st.page_link("pages/01_ingest.py", label="Go to Ingest Documents", icon="📄")
else:
    with st.form("generate_dataset_form"):
        col1, col2 = st.columns(2)

        with col1:
            collection_name = st.selectbox(
                "Source collection",
                options=collection_names,
                help="The ChromaDB collection to sample chunks from.",
            )
            dataset_name = st.text_input(
                "Dataset name",
                placeholder="e.g. Q3 Financial Report Eval",
            )
            seed_query = st.text_input(
                "Sampling query (optional)",
                placeholder="Leave blank for a generic sample",
                help=(
                    "Biases chunk sampling toward a topic via "
                    "similarity search. Leave blank to sample broadly "
                    "representative chunks."
                ),
            )
            tags_raw = st.text_input(
                "Tags (comma-separated, optional)",
                placeholder="e.g. finance, q3, baseline",
            )

        with col2:
            sample_size = st.slider(
                "Chunks to sample", min_value=1, max_value=50, value=10
            )
            n_pairs_per_chunk = st.slider(
                "Pairs per chunk", min_value=1, max_value=5, value=1
            )
            max_pairs_total = st.slider(
                "Max total pairs", min_value=1, max_value=100, value=20
            )
            temperature = st.slider(
                "Generation temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.4,
                step=0.1,
                help=(
                    "Higher values produce more varied questions; "
                    "0.0 is more deterministic but formulaic."
                ),
            )

        description = st.text_area(
            "Description (optional)", placeholder="Notes about this dataset..."
        )

        submitted = st.form_submit_button("Generate dataset", type="primary")

    if submitted:
        if not dataset_name.strip():
            st.error("Dataset name is required.")
        else:
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            payload: dict[str, Any] = {
                "collection_name": collection_name,
                "dataset_name": dataset_name.strip(),
                "description": description.strip() or None,
                "seed_query": seed_query.strip() or None,
                "sample_size": sample_size,
                "n_pairs_per_chunk": n_pairs_per_chunk,
                "max_pairs_total": max_pairs_total,
                "temperature": temperature,
                "tags": tags,
            }
            with st.spinner(
                "Sampling chunks and generating Q&A pairs via Gemini... "
                "this may take a moment."
            ):
                try:
                    result = client.generate_dataset(payload)
                    st.session_state["last_generation_result"] = result
                    st.session_state["selected_dataset_id"] = result["dataset_id"]
                except APIError as exc:
                    st.error(f"Generation failed: {exc.detail}")
                    st.session_state.pop("last_generation_result", None)


# ===========================================================================
# LAST GENERATION RESULT
# ===========================================================================

if "last_generation_result" in st.session_state:
    result = st.session_state["last_generation_result"]
    gen = result["generation"]

    st.divider()
    st.subheader(f"Result — '{result['name']}'")

    if gen["pairs_generated"] > 0:
        st.success(
            f"✅ Generated {gen['pairs_generated']} pair(s) from "
            f"{gen['chunks_succeeded']}/{gen['chunks_attempted']} chunks."
        )
    else:
        st.warning("No pairs were generated. Try a different collection or seed query.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pairs generated", gen["pairs_generated"])
    m2.metric("Rejected/deduped", gen["pairs_rejected"] + gen["pairs_deduplicated"])
    m3.metric("Tokens used", gen["total_input_tokens"] + gen["total_output_tokens"])
    m4.metric("Latency", f"{gen['total_latency_ms']:.0f} ms")

    if st.button("Clear result"):
        del st.session_state["last_generation_result"]
        st.rerun()


# ===========================================================================
# DATASET BROWSER
# ===========================================================================

st.divider()
st.subheader("Existing datasets")

f1, f2, f3, f4 = st.columns(4)
with f1:
    status_choice = st.selectbox("Status", _STATUS_OPTIONS, key="ds_filter_status")
with f2:
    method_choice = st.selectbox("Method", _METHOD_OPTIONS, key="ds_filter_method")
with f3:
    sort_by = st.selectbox("Sort by", _SORT_OPTIONS, key="ds_sort_by")
with f4:
    descending = st.checkbox("Newest first", value=True, key="ds_descending")

try:
    list_resp = client.list_datasets(
        status=None if status_choice == "All" else status_choice,
        generation_method=None if method_choice == "All" else method_choice,
        sort_by=sort_by,
        descending=descending,
        limit=50,
    )
except APIError as exc:
    st.error(f"Could not load datasets: {exc.detail}")
    list_resp = None

if list_resp is not None:
    datasets = list_resp["datasets"]

    if not datasets:
        st.info("No datasets yet. Generate one above.")
    else:
        st.caption(f"{list_resp['total']} dataset(s) total.")

        for ds in datasets:
            status_emoji = {
                "draft": "📝",
                "ready": "🟢",
                "running": "🟡",
                "completed": "✅",
                "partial": "⚠️",
            }.get(ds["status"], "⚪")

            header = (
                f"{status_emoji} **{ds['name']}** — {ds['total_pairs']} pairs, "
                f"{ds['evaluated_pairs']} evaluated ({ds['completion_rate']:.0%})"
            )

            with st.expander(header, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.text(f"Source: {ds['source_collection'] or '—'}")
                c2.text(f"Generator: {ds['generator_model'] or '—'}")
                c3.text(f"Status: {ds['status']}")
                if ds["description"]:
                    st.caption(ds["description"])
                if ds["tags"]:
                    st.caption("Tags: " + ", ".join(ds["tags"]))

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("👁️ View / Edit pairs", key=f"view_{ds['id']}"):
                        st.session_state["selected_dataset_id"] = ds["id"]
                        st.rerun()
                with b2:
                    try:
                        csv_bytes = None
                        if st.button("📥 Export CSV", key=f"export_{ds['id']}"):
                            csv_bytes = client.export_dataset_csv(ds["id"])
                        if csv_bytes:
                            st.download_button(
                                "Download",
                                data=csv_bytes,
                                file_name=f"{ds['name']}.csv",
                                mime="text/csv",
                                key=f"download_{ds['id']}",
                            )
                    except APIError as exc:
                        st.error(f"Export failed: {exc.detail}")
                with b3:
                    delete_key = f"delete_confirm_ds_{ds['id']}"
                    if st.session_state.get(delete_key, False):
                        st.warning("Delete permanently?")
                        yes_col, no_col = st.columns(2)
                        with yes_col:
                            if st.button("Yes", key=f"yes_{ds['id']}", type="primary"):
                                try:
                                    client.delete_dataset(ds["id"])
                                    st.session_state.pop(delete_key, None)
                                    if st.session_state.get("selected_dataset_id") == ds["id"]:
                                        st.session_state.pop("selected_dataset_id", None)
                                    st.rerun()
                                except APIError as exc:
                                    st.error(f"Delete failed: {exc.detail}")
                        with no_col:
                            if st.button("No", key=f"no_{ds['id']}"):
                                st.session_state.pop(delete_key, None)
                                st.rerun()
                    else:
                        if st.button("🗑️ Delete", key=f"delete_ds_{ds['id']}"):
                            st.session_state[delete_key] = True
                            st.rerun()


# ===========================================================================
# SELECTED DATASET DETAIL — pairs preview + inline editing
# ===========================================================================

selected_id = st.session_state.get("selected_dataset_id")

if selected_id:
    st.divider()
    try:
        detail = client.get_dataset(selected_id)
    except APIError as exc:
        st.error(f"Could not load dataset: {exc.detail}")
        detail = None

    if detail is not None:
        st.subheader(f"📋 {detail['name']}")
        if st.button("✖️ Close detail view"):
            del st.session_state["selected_dataset_id"]
            st.rerun()

        pairs = detail["pairs"]
        st.caption(f"{len(pairs)} pair(s) — status: {detail['status']}")

        for pair in pairs:
            status_badge = {
                "pending": "⚪ pending",
                "answered": "🟡 answered",
                "evaluated": "🟢 evaluated",
                "failed": "🔴 failed",
            }.get(pair["status"], pair["status"])

            with st.container(border=True):
                st.markdown(f"**{pair['question']}**  ·  {status_badge}")
                st.caption(
                    f"Source: {pair['source_file']}"
                    + (f" (p.{pair['source_page']})" if pair["source_page"] else "")
                )

                if pair["status"] == "pending":
                    with st.form(f"edit_pair_form_{pair['id']}"):
                        new_question = st.text_area(
                            "Question",
                            value=pair["question"],
                            key=f"q_{pair['id']}",
                            height=70,
                        )
                        new_answer = st.text_area(
                            "Ground truth answer",
                            value=pair["ground_truth_answer"],
                            key=f"a_{pair['id']}",
                            height=70,
                        )
                        save_col, delete_col = st.columns([1, 1])
                        with save_col:
                            save_clicked = st.form_submit_button("💾 Save changes")
                        with delete_col:
                            delete_clicked = st.form_submit_button("🗑️ Delete pair")

                    if save_clicked:
                        try:
                            client.edit_pair(
                                selected_id,
                                pair["id"],
                                {
                                    "question": new_question,
                                    "ground_truth_answer": new_answer,
                                },
                            )
                            st.success("Pair updated.")
                            st.rerun()
                        except APIError as exc:
                            st.error(f"Update failed: {exc.detail}")

                    if delete_clicked:
                        try:
                            client.delete_pair(selected_id, pair["id"])
                            st.success("Pair deleted.")
                            st.rerun()
                        except APIError as exc:
                            st.error(f"Delete failed: {exc.detail}")

                else:
                    st.text_area(
                        "Ground truth answer",
                        value=pair["ground_truth_answer"],
                        disabled=True,
                        height=70,
                        key=f"ro_a_{pair['id']}",
                    )
                    if pair["generated_answer"]:
                        st.text_area(
                            "Generated answer",
                            value=pair["generated_answer"],
                            disabled=True,
                            height=70,
                            key=f"ro_gen_{pair['id']}",
                        )
                    if pair["composite_score"] is not None:
                        st.metric("Composite score", f"{pair['composite_score']:.2f} / 5.0")
                    st.caption(
                        "Editing is disabled — this pair has already been "
                        "answered or evaluated."
                    )