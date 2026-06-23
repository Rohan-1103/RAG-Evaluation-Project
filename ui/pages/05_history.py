"""
ui/pages/05_history.py

Unified history browser — filterable, paginated tables of every past
single-model run and every past multi-model comparison, independent of
which dataset or page originally produced them.

Maps onto the same read/delete endpoints already used inline on
03_evaluate.py and 04_compare.py, just with richer filtering and
pagination exposed:

    Runs tab          -> GET /api/v1/evaluate            (filtered, paginated)
                       -> GET /api/v1/evaluate/summary.csv (bulk export)
                       -> DELETE /api/v1/evaluate/{run_id}
    Comparisons tab    -> GET /api/v1/compare              (filtered, paginated)
                       -> DELETE /api/v1/compare/{matrix_id}

Why this page does NOT re-implement the per-question drilldown or the
radar/bar/scatter charts itself:

  03_evaluate.py and 04_compare.py already contain the ONLY renderers
  for those two views in the entire codebase, each driven by reading
  st.session_state["selected_run_detail"] / ["selected_comparison_detail"]
  unconditionally on every rerun (see those files' own docstrings for
  why that pattern was chosen). "View full results" here does exactly
  what those pages' own inline history sections do: fetch the full
  detail object once, store it under that same session_state key, then
  st.switch_page() to the page that actually owns the rendering. The
  alternative — copy-pasting ~150 lines of chart/expander code into a
  THIRD location — would mean three places to keep in sync the next
  time a metric is added or a chart's layout changes. There is exactly
  one renderer per detail type; this page is a second DOOR into it, not
  a second IMPLEMENTATION of it.

Why the Runs tab shows a real "Page X of Y" but the Comparisons tab
only shows "Showing N results" with a bare Next/Previous toggle:

  RunListOut.total is backed by RunRepository.count_runs() — an actual
  COUNT(*) query, so true pagination math is possible. MatrixListOut
  deliberately has NO total field; src/api/routes/compare.py's own
  docstring on list_comparisons explains why: no
  count_comparison_matrices() method exists yet on RunRepository, and
  fabricating a number here (e.g. assuming there are no more results
  past the current page) would misrepresent what the backend can
  currently guarantee. This page surfaces that exact same honest gap
  rather than papering over it.

Why the model filter is a plain text input requiring an EXACT match,
not a dropdown or substring search:

  RunRepository.list_runs()'s rag_model filter
  (src/storage/repository.py) is a SQL `==` comparison, not a LIKE —
  and as already explained in 04_compare.py's own docstring, no
  /api/v1/models endpoint exists to populate a dropdown from the live
  model catalogue without this UI layer importing config/models.yaml
  directly, which it is architecturally forbidden from doing. The text
  input's help caption says "exact match" explicitly so the absence of
  fuzzy matching is communicated, not discovered through a confusing
  empty result.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math
from typing import Any

import pandas as pd
import streamlit as st

from ui.components.api_client import APIError, get_api_client
from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("🕒 Run History")
st.caption(
    "Browse every past evaluation run and model comparison, filter by "
    "dataset/model/status, and export or delete historical results."
)

client = get_api_client()

_PAGE_SIZE = 20

# Mirrors src/evaluation/schema.py's RunStatus enum — duplicated here
# rather than imported, per this codebase's established UI-layer
# convention of small, locally-defined option lists instead of
# importing from src/ (see 02_dataset.py's _STATUS_OPTIONS for the
# same pattern applied to DatasetStatus).
_RUN_STATUS_OPTIONS = ["All", "pending", "running", "completed", "partial", "failed"]
_RUN_SORT_OPTIONS = [
    "started_at",
    "composite_mean",
    "total_cost_usd",
    "avg_total_latency_ms",
]

_STATUS_EMOJI = {
    "pending": "⚪",
    "running": "🟡",
    "completed": "🟢",
    "partial": "⚠️",
    "failed": "🔴",
}


def _score_color(score: float | None) -> str:
    """Same traffic-light thresholds duplicated across 03_evaluate.py and
    04_compare.py — see those files for why this is intentionally
    repeated per-page rather than shared."""
    if score is None:
        return ""
    if score >= 4.5:
        return "background-color: #d4edda"
    if score >= 3.5:
        return "background-color: #e2f0cb"
    if score >= 2.5:
        return "background-color: #fff3cd"
    return "background-color: #f8d7da"


# ===========================================================================
# DATASET FILTER OPTIONS — shared across both tabs
# ===========================================================================

try:
    _datasets_resp = client.list_datasets(limit=100)
    _dataset_options: dict[str, dict[str, Any]] = {
        d["name"]: d for d in _datasets_resp["datasets"]
    }
except APIError:
    _dataset_options = {}


tab_runs, tab_comparisons = st.tabs(["📋 Single-Model Runs", "📊 Comparisons"])


# ===========================================================================
# TAB 1 — SINGLE-MODEL RUNS
# ===========================================================================

with tab_runs:
    st.subheader("Filters")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        run_dataset_label = st.selectbox(
            "Dataset",
            options=["All"] + list(_dataset_options.keys()),
            key="hist_run_dataset",
        )
        run_dataset_id = (
            _dataset_options[run_dataset_label]["id"]
            if run_dataset_label != "All"
            else None
        )
    with f2:
        run_model_filter = st.text_input(
            "Model ID (exact match, optional)",
            key="hist_run_model",
            placeholder="e.g. gemini-2.0-flash",
        )
    with f3:
        run_status_choice = st.selectbox(
            "Status", _RUN_STATUS_OPTIONS, key="hist_run_status"
        )
    with f4:
        run_sort_by = st.selectbox(
            "Sort by", _RUN_SORT_OPTIONS, key="hist_run_sort"
        )

    run_descending = st.checkbox("Newest/highest first", value=True, key="hist_run_desc")

    # Reset pagination to page 1 whenever any filter changes — without
    # this, switching datasets while sitting on page 3 of the old
    # filter set would silently request an out-of-range offset against
    # the new, likely much smaller, result set.
    _run_filter_signature = (
        run_dataset_id,
        run_model_filter.strip() or None,
        run_status_choice,
        run_sort_by,
        run_descending,
    )
    if st.session_state.get("hist_run_filter_sig") != _run_filter_signature:
        st.session_state["hist_run_offset"] = 0
        st.session_state["hist_run_filter_sig"] = _run_filter_signature

    run_offset = st.session_state.get("hist_run_offset", 0)

    try:
        runs_resp = client.list_runs(
            dataset_id=run_dataset_id,
            rag_model=run_model_filter.strip() or None,
            status=None if run_status_choice == "All" else run_status_choice,
            sort_by=run_sort_by,
            descending=run_descending,
            limit=_PAGE_SIZE,
            offset=run_offset,
        )
    except APIError as exc:
        st.error(f"Could not load run history: {exc.detail}")
        runs_resp = None

    if runs_resp is not None:
        runs = runs_resp["runs"]
        total = runs_resp["total"]
        total_pages = max(1, math.ceil(total / _PAGE_SIZE))
        current_page = (run_offset // _PAGE_SIZE) + 1

        st.divider()
        top_col1, top_col2 = st.columns([2, 1])
        with top_col1:
            st.caption(f"{total} run(s) total — page {current_page} of {total_pages}.")
        with top_col2:
            try:
                summary_csv = client.export_runs_summary_csv(dataset_id=run_dataset_id)
                st.download_button(
                    "📥 Export all summaries CSV",
                    data=summary_csv,
                    file_name="runs_summary.csv",
                    mime="text/csv",
                    key="export_runs_summary",
                )
            except APIError:
                pass

        if not runs:
            st.info("No runs match the current filters.")
        else:
            runs_df = pd.DataFrame(
                [
                    {
                        "Status": f"{_STATUS_EMOJI.get(r['status'], '⚪')} {r['status']}",
                        "Dataset": r["dataset_name"],
                        "Model": r["rag_model"],
                        "Composite": round(r["composite_mean"], 2),
                        "Evaluated": f"{r['n_pairs_evaluated']}/{r['n_pairs_total']}",
                        "Cost ($)": round(r["total_cost_usd"], 5),
                        "Latency (ms)": round(r["avg_total_latency_ms"], 0),
                        "Started": r["started_at"][:19].replace("T", " "),
                        "⚠": "⚠️" if r["low_sample_warning"] else "",
                        "_run_id": r["run_id"],
                    }
                    for r in runs
                ]
            )

            display_df = runs_df.drop(columns=["_run_id"])
            st.dataframe(
                display_df.style.applymap(_score_color, subset=["Composite"]),
                use_container_width=True,
                hide_index=True,
            )

            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("⬅️ Previous", disabled=current_page <= 1, key="run_prev"):
                    st.session_state["hist_run_offset"] = max(0, run_offset - _PAGE_SIZE)
                    st.rerun()
            with nav_col3:
                if st.button(
                    "Next ➡️", disabled=current_page >= total_pages, key="run_next"
                ):
                    st.session_state["hist_run_offset"] = run_offset + _PAGE_SIZE
                    st.rerun()

            st.divider()
            selected_run_id = st.selectbox(
                "Select a run to act on",
                options=runs_df["_run_id"].tolist(),
                format_func=lambda rid: next(
                    f"{r['rag_model']} — {r['started_at'][:19].replace('T', ' ')} "
                    f"(composite {r['composite_mean']:.2f}, {r['status']})"
                    for r in runs
                    if r["run_id"] == rid
                ),
                key="hist_run_select",
            )

            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("🔍 Open full drilldown", key="hist_run_open"):
                    try:
                        full_report = client.get_run(selected_run_id)
                        st.session_state["selected_run_detail"] = full_report
                        st.switch_page("pages/03_evaluate.py")
                    except APIError as exc:
                        st.error(f"Could not load run: {exc.detail}")
            with a2:
                try:
                    run_csv = client.export_run_csv(selected_run_id)
                    st.download_button(
                        "📥 Export this run's CSV",
                        data=run_csv,
                        file_name=f"run_{selected_run_id}.csv",
                        mime="text/csv",
                        key="hist_run_export",
                    )
                except APIError:
                    pass
            with a3:
                confirm_key = f"hist_run_delete_confirm_{selected_run_id}"
                if st.session_state.get(confirm_key, False):
                    st.warning("Delete permanently?")
                    yes_col, no_col = st.columns(2)
                    with yes_col:
                        if st.button("Yes", key="hist_run_delete_yes", type="primary"):
                            try:
                                client.delete_run(selected_run_id)
                                st.session_state.pop(confirm_key, None)
                                st.success("Run deleted.")
                                st.rerun()
                            except APIError as exc:
                                st.error(f"Delete failed: {exc.detail}")
                    with no_col:
                        if st.button("No", key="hist_run_delete_no"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                else:
                    if st.button("🗑️ Delete run", key="hist_run_delete"):
                        st.session_state[confirm_key] = True
                        st.rerun()


# ===========================================================================
# TAB 2 — COMPARISONS
# ===========================================================================

with tab_comparisons:
    st.subheader("Filters")

    cmp_dataset_label = st.selectbox(
        "Dataset",
        options=["All"] + list(_dataset_options.keys()),
        key="hist_cmp_dataset",
    )
    cmp_dataset_id = (
        _dataset_options[cmp_dataset_label]["id"] if cmp_dataset_label != "All" else None
    )

    # Reset pagination to page 1 on filter change — same rationale as
    # the Runs tab above.
    _cmp_filter_signature = (cmp_dataset_id,)
    if st.session_state.get("hist_cmp_filter_sig") != _cmp_filter_signature:
        st.session_state["hist_cmp_offset"] = 0
        st.session_state["hist_cmp_filter_sig"] = _cmp_filter_signature

    cmp_offset = st.session_state.get("hist_cmp_offset", 0)

    try:
        matrices_resp = client.list_comparisons(
            dataset_id=cmp_dataset_id, limit=_PAGE_SIZE, offset=cmp_offset
        )
    except APIError as exc:
        st.error(f"Could not load comparison history: {exc.detail}")
        matrices_resp = None

    if matrices_resp is not None:
        matrices = matrices_resp["matrices"]

        st.divider()
        # No total count is available here — see this file's module
        # docstring on why MatrixListOut deliberately has no `total`
        # field. Showing "Showing N results" rather than "Page X of Y"
        # is the honest representation of what this endpoint can
        # currently guarantee.
        st.caption(
            f"Showing {len(matrices)} comparison(s) "
            f"(sorted newest first; no total count available yet)."
        )

        if not matrices:
            st.info("No comparisons match the current filters.")
        else:
            matrices_df = pd.DataFrame(
                [
                    {
                        "Dataset": m["dataset_name"],
                        "Models": m["n_models"],
                        "Best model": m["best_model_id"] or "—",
                        "Best score": (
                            round(m["best_composite_score"], 2)
                            if m["best_composite_score"] is not None
                            else None
                        ),
                        "Judge": m["judge_model"],
                        "Created": m["created_at"][:19].replace("T", " "),
                        "_matrix_id": m["matrix_id"],
                    }
                    for m in matrices
                ]
            )

            display_cmp_df = matrices_df.drop(columns=["_matrix_id"])
            st.dataframe(
                display_cmp_df.style.applymap(_score_color, subset=["Best score"]),
                use_container_width=True,
                hide_index=True,
            )

            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button(
                    "⬅️ Previous", disabled=cmp_offset <= 0, key="cmp_prev"
                ):
                    st.session_state["hist_cmp_offset"] = max(0, cmp_offset - _PAGE_SIZE)
                    st.rerun()
            with nav_col3:
                # No total to compare against, so Next is only disabled
                # when the current page came back short of a full page
                # (the standard signal that no further page exists).
                if st.button(
                    "Next ➡️",
                    disabled=len(matrices) < _PAGE_SIZE,
                    key="cmp_next",
                ):
                    st.session_state["hist_cmp_offset"] = cmp_offset + _PAGE_SIZE
                    st.rerun()

            st.divider()
            selected_matrix_id = st.selectbox(
                "Select a comparison to act on",
                options=matrices_df["_matrix_id"].tolist(),
                format_func=lambda mid: next(
                    f"{m['dataset_name']} — {m['n_models']} models — "
                    f"{m['created_at'][:19].replace('T', ' ')}"
                    for m in matrices
                    if m["matrix_id"] == mid
                ),
                key="hist_cmp_select",
            )

            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("🔍 Open full comparison", key="hist_cmp_open"):
                    try:
                        full_matrix = client.get_comparison(selected_matrix_id)
                        st.session_state["selected_comparison_detail"] = full_matrix
                        st.switch_page("pages/04_compare.py")
                    except APIError as exc:
                        st.error(f"Could not load comparison: {exc.detail}")
            with a2:
                try:
                    cmp_csv = client.export_comparison_csv(selected_matrix_id)
                    st.download_button(
                        "📥 Export this comparison's CSV",
                        data=cmp_csv,
                        file_name=f"comparison_{selected_matrix_id}.csv",
                        mime="text/csv",
                        key="hist_cmp_export",
                    )
                except APIError:
                    pass
            with a3:
                confirm_key = f"hist_cmp_delete_confirm_{selected_matrix_id}"
                if st.session_state.get(confirm_key, False):
                    delete_runs_too = st.checkbox(
                        "Also delete underlying runs",
                        key="hist_cmp_delete_runs_checkbox",
                        help=(
                            "If unchecked, individual model runs remain "
                            "browsable in the Single-Model Runs tab "
                            "above with their matrix link cleared."
                        ),
                    )
                    yes_col, no_col = st.columns(2)
                    with yes_col:
                        if st.button("Yes, delete", key="hist_cmp_delete_yes", type="primary"):
                            try:
                                client.delete_comparison(
                                    selected_matrix_id, delete_runs=delete_runs_too
                                )
                                st.session_state.pop(confirm_key, None)
                                st.success("Comparison deleted.")
                                st.rerun()
                            except APIError as exc:
                                st.error(f"Delete failed: {exc.detail}")
                    with no_col:
                        if st.button("Cancel", key="hist_cmp_delete_no"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                else:
                    if st.button("🗑️ Delete comparison", key="hist_cmp_delete"):
                        st.session_state[confirm_key] = True
                        st.rerun()