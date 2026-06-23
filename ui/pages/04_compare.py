"""
ui/pages/04_compare.py

Multi-model comparison page — configure and run a parameter grid of
RAG models against one dataset, then inspect the resulting comparison
matrix via a radar chart, a metric bar chart, a latency/cost scatter,
and a side-by-side table.

Maps directly onto src/api/routes/compare.py's endpoints:

    Run form              -> POST   /api/v1/compare/run
    Comparison history     -> GET    /api/v1/compare
    Detail panel + charts   -> GET    /api/v1/compare/{matrix_id}
    "Export CSV" button      -> GET    /api/v1/compare/{matrix_id}/export.csv
    "Delete" button             -> DELETE /api/v1/compare/{matrix_id}

Why model_ids is a free-text comma-separated input rather than a
multiselect sourced from a live "list available models" endpoint:

  No /api/v1/models route exists anywhere in src/api/routes/ — the
  model catalogue lives entirely in config/models.yaml and is read
  server-side via config.get_model_registry(), which this UI layer is
  architecturally forbidden from importing (see ui.components.api_client's
  module docstring: this layer talks to FastAPI exclusively, never to
  config/ or src/ directly). Building a dropdown here would require
  either a new backend endpoint just to expose that registry, or a
  second, UI-side hardcoded copy of the model catalogue that would
  silently drift out of sync with models.yaml the first time a model is
  added/disabled there. Free text is the honest choice given the
  current API surface — RunComparisonRequest.model_ids is itself a
  plain list[str] with no enum constraint for the same reason.

Why top_k_values and temperatures are also comma-separated text inputs,
not sliders with multiple handles:

  Streamlit has no native multi-value slider widget, and
  RunComparisonRequest.top_k_values / temperatures are genuinely
  open-ended lists (ComparisonRunner.build_grid_configs() in
  src/comparison/runner.py accepts any list the caller provides, not a
  fixed set of presets) — a comma-separated text field maps directly
  onto that open-ended list shape without inventing artificial preset
  buckets the backend was never designed to expect.

Why this page computes and displays the resulting grid SIZE
(len(model_ids) x len(top_k_values) x len(temperatures)) before
submission, with a soft warning above 12:

  ComparisonRunner._enforce_max_total_runs() (src/comparison/runner.py)
  rejects the request server-side with a 422 if the expanded grid
  exceeds models.yaml's comparison_grid.max_total_runs — but that
  rejection only happens AFTER the user has filled in every field and
  clicked submit. Showing the expected grid size up front, with a
  caption noting the default cap, lets the user catch "3 models x 3
  top_k x 3 temperatures = 27 configs" BEFORE hitting that wall, the
  same anticipate-the-backend-constraint philosophy already applied to
  the collection-name selectbox in 01_ingest.py and 02_dataset.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components.api_client import APIError, get_api_client
from ui.components.sidebar import render_sidebar

render_sidebar()

st.title("📊 Compare Models")
st.caption(
    "Run a parameter grid of RAG models against one dataset and "
    "benchmark them side-by-side on all 4 evaluation metrics, latency, "
    "and cost."
)

client = get_api_client()

_METRIC_NAMES = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "correctness",
]
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "context_precision": "Context Precision",
    "correctness": "Correctness",
}
_SOFT_GRID_WARNING_THRESHOLD = 12  # mirrors models.yaml's default max_total_runs


def _parse_csv_list(raw: str, caster: type) -> list[Any] | None:
    """
    Parse a comma-separated text field into a list of `caster`-typed
    values, or None if the field is blank (meaning "use the backend's
    default" — [5] for top_k, [0.0] for temperature, per
    ComparisonRunner.build_grid_configs()'s own defaults).

    Raises ValueError with a clear message on malformed input, caught
    by the caller and shown as an inline st.error rather than letting
    a raw exception surface.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        return [caster(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Could not parse '{raw}' as a comma-separated list of "
            f"{caster.__name__} values."
        ) from exc


def _validate_grid_inputs(
    model_ids: list[str],
    top_k_values: list[int] | None,
    temperatures: list[float] | None,
) -> list[str]:
    """
    Client-side mirror of RunComparisonRequest's field_validators in
    src/api/routes/compare.py — catches duplicates and out-of-range
    values before a round trip to the backend, exactly as
    01_ingest.py's collection-name regex anticipates the backend's own
    validation locally.
    """
    errors: list[str] = []

    if len(model_ids) != len(set(model_ids)):
        dupes = sorted({m for m in model_ids if model_ids.count(m) > 1})
        errors.append(f"Duplicate model IDs: {', '.join(dupes)}.")

    if top_k_values is not None:
        if len(top_k_values) != len(set(top_k_values)):
            errors.append("Duplicate top_k values are not allowed.")
        for k in top_k_values:
            if not 1 <= k <= 20:
                errors.append(f"top_k value {k} is outside valid range [1, 20].")

    if temperatures is not None:
        if len(temperatures) != len(set(temperatures)):
            errors.append("Duplicate temperature values are not allowed.")
        for t in temperatures:
            if not 0.0 <= t <= 2.0:
                errors.append(f"Temperature {t} is outside valid range [0.0, 2.0].")

    return errors


def _score_color(score: float | None) -> str:
    """Same traffic-light thresholds as 03_evaluate.py's _score_color, duplicated
    intentionally per this codebase's established per-page convention."""
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
# CHART BUILDERS
# ===========================================================================


def _build_radar_chart(entries: list[dict[str, Any]]) -> go.Figure:
    """One Scatterpolar trace per model, all on a shared 1-5 scale."""
    fig = go.Figure()
    labels = [_METRIC_LABELS[m] for m in _METRIC_NAMES]

    for entry in entries:
        radar = entry["radar"]
        values = [radar.get(m, 0.0) for m in _METRIC_NAMES]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=entry["display_name"],
                opacity=0.7,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        showlegend=True,
        height=450,
        margin=dict(t=30, b=30),
    )
    return fig


def _build_metric_bar_chart(entries: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart: one group of bars per metric, one bar colour per model."""
    fig = go.Figure()

    for entry in entries:
        means = [entry[f"{m}_mean"] for m in _METRIC_NAMES]
        stds = [entry[f"{m}_std"] for m in _METRIC_NAMES]
        fig.add_trace(
            go.Bar(
                name=entry["display_name"],
                x=[_METRIC_LABELS[m] for m in _METRIC_NAMES],
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Mean score", range=[0, 5.5]),
        height=400,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _build_latency_cost_scatter(entries: list[dict[str, Any]]) -> go.Figure:
    """
    Composite score (y) vs latency (x), marker size scaled by cost —
    the "sweet spot" view: top-left quadrant is fast + high quality.
    """
    fig = go.Figure()

    for entry in entries:
        fig.add_trace(
            go.Scatter(
                x=[entry["avg_latency_ms"]],
                y=[entry["composite_mean"]],
                mode="markers+text",
                marker=dict(
                    size=max(15, min(60, entry["total_cost_usd"] * 100_000 + 15)),
                ),
                text=[entry["display_name"]],
                textposition="top center",
                name=entry["display_name"],
            )
        )

    fig.update_layout(
        xaxis=dict(title="Avg latency (ms)"),
        yaxis=dict(title="Composite score", range=[0, 5.5]),
        height=400,
        margin=dict(t=30, b=30),
        showlegend=False,
    )
    return fig


# ===========================================================================
# RUN FORM
# ===========================================================================

st.subheader("Configure a comparison")

try:
    datasets_resp = client.list_datasets(limit=100)
    dataset_options = {
        f"{d['name']} ({d['total_pairs']} pairs, {d['status']})": d
        for d in datasets_resp["datasets"]
    }
except APIError:
    dataset_options = {}

try:
    collections_resp = client.list_collections()
    collection_names = [c["name"] for c in collections_resp["collections"]]
except APIError:
    collection_names = []

if not dataset_options:
    st.warning("No datasets found. Generate one first.")
    st.page_link("pages/02_dataset.py", label="Go to Generate Dataset", icon="🧪")
elif not collection_names:
    st.warning("No collections found. Ingest documents first.")
    st.page_link("pages/01_ingest.py", label="Go to Ingest Documents", icon="📄")
else:
    with st.form("run_comparison_form"):
        col1, col2 = st.columns(2)

        with col1:
            dataset_label = st.selectbox(
                "Dataset", options=list(dataset_options.keys())
            )
            selected_dataset = dataset_options[dataset_label]

            default_collection_idx = 0
            if selected_dataset.get("source_collection") in collection_names:
                default_collection_idx = collection_names.index(
                    selected_dataset["source_collection"]
                )
            collection_name = st.selectbox(
                "Collection to retrieve from",
                options=collection_names,
                index=default_collection_idx,
            )

            model_ids_raw = st.text_input(
                "Model IDs (comma-separated)",
                placeholder="gemini-2.0-flash, gemini-1.5-flash-8b",
                help="Up to 10 models, each ID must be unique.",
            )

            dataset_name_override = st.text_input(
                "Comparison label (optional)",
                placeholder="Defaults to the dataset's own name.",
            )

        with col2:
            top_k_raw = st.text_input(
                "Top-K values (comma-separated, optional)",
                placeholder="Defaults to [5]. e.g. 3, 5, 8",
            )
            temperatures_raw = st.text_input(
                "Temperatures (comma-separated, optional)",
                placeholder="Defaults to [0.0]. e.g. 0.0, 0.5",
            )
            score_threshold = st.slider(
                "Min similarity score for retrieval", 0.0, 1.0, 0.0, 0.05
            )
            max_output_tokens = st.slider(
                "Max output tokens", min_value=64, max_value=4096, value=1024
            )

        submitted = st.form_submit_button("▶️ Run comparison", type="primary")

    if submitted:
        try:
            model_ids = [m.strip() for m in model_ids_raw.split(",") if m.strip()]
            top_k_values = _parse_csv_list(top_k_raw, int)
            temperatures = _parse_csv_list(temperatures_raw, float)
        except ValueError as exc:
            st.error(str(exc))
            model_ids = []
            top_k_values = None
            temperatures = None

        if not model_ids:
            st.error("Enter at least one model ID.")
        else:
            errors = _validate_grid_inputs(model_ids, top_k_values, temperatures)
            grid_size = (
                len(model_ids)
                * len(top_k_values or [1])
                * len(temperatures or [1])
            )

            if errors:
                for err in errors:
                    st.error(err)
            else:
                if grid_size > _SOFT_GRID_WARNING_THRESHOLD:
                    st.warning(
                        f"⚠️ This grid expands to {grid_size} model runs, "
                        f"above the typical default cap "
                        f"({_SOFT_GRID_WARNING_THRESHOLD}). The backend may "
                        f"reject this request — consider reducing the number "
                        f"of models, top_k values, or temperatures."
                    )

                payload: dict[str, Any] = {
                    "dataset_id": selected_dataset["id"],
                    "collection_name": collection_name,
                    "model_ids": model_ids,
                    "top_k_values": top_k_values,
                    "temperatures": temperatures,
                    "score_threshold": score_threshold,
                    "max_output_tokens": max_output_tokens,
                    "dataset_name_override": dataset_name_override.strip() or None,
                }

                with st.spinner(
                    f"Running {grid_size} model config(s) against "
                    f"{selected_dataset['total_pairs']} pair(s)... "
                    f"this can take a while for larger grids."
                ):
                    try:
                        matrix = client.run_comparison(payload)
                        st.session_state["selected_comparison_detail"] = matrix
                        st.success(
                            f"✅ Comparison complete — "
                            f"{matrix['n_models']} model(s) ranked. "
                            f"Best: {matrix['best_model_id']} "
                            f"({matrix['best_composite_score']:.2f}/5.0)"
                        )
                    except APIError as exc:
                        st.error(f"Comparison failed: {exc.detail}")


# ===========================================================================
# COMPARISON HISTORY
# ===========================================================================

st.divider()
st.subheader("Comparison history")

dataset_filter_options = ["All"] + list(dataset_options.keys()) if dataset_options else ["All"]
filter_label = st.selectbox("Filter by dataset", dataset_filter_options, key="cmp_ds_filter")
filter_dataset_id = (
    dataset_options[filter_label]["id"]
    if filter_label != "All" and filter_label in dataset_options
    else None
)

try:
    matrices_resp = client.list_comparisons(dataset_id=filter_dataset_id, limit=50)
except APIError as exc:
    st.error(f"Could not load comparison history: {exc.detail}")
    matrices_resp = None

if matrices_resp is not None:
    matrices = matrices_resp["matrices"]
    if not matrices:
        st.info("No comparisons yet. Configure and run one above.")
    else:
        st.caption(f"Showing {len(matrices)} comparison(s).")

        history_df = pd.DataFrame(
            [
                {
                    "Matrix ID": m["matrix_id"][-12:],
                    "Dataset": m["dataset_name"],
                    "Models": m["n_models"],
                    "Best model": m["best_model_id"] or "—",
                    "Best score": (
                        round(m["best_composite_score"], 2)
                        if m["best_composite_score"] is not None
                        else None
                    ),
                    "Created": m["created_at"][:19].replace("T", " "),
                    "_full_matrix_id": m["matrix_id"],
                }
                for m in matrices
            ]
        )

        st.dataframe(
            history_df.drop(columns=["_full_matrix_id"]),
            use_container_width=True,
            hide_index=True,
        )

        matrix_choice = st.selectbox(
            "Select a comparison to inspect",
            options=history_df["_full_matrix_id"].tolist(),
            format_func=lambda mid: next(
                f"{m['dataset_name']} — {m['n_models']} models — "
                f"{m['created_at'][:19].replace('T', ' ')}"
                for m in matrices
                if m["matrix_id"] == mid
            ),
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("👁️ View comparison"):
                try:
                    full_matrix = client.get_comparison(matrix_choice)
                    st.session_state["selected_comparison_detail"] = full_matrix
                except APIError as exc:
                    st.error(f"Could not load comparison: {exc.detail}")
        with b2:
            delete_runs_too = st.checkbox(
                "Also delete underlying runs",
                key="delete_runs_checkbox",
                help=(
                    "If unchecked (default), individual model runs "
                    "remain browsable in the Evaluate page's run history."
                ),
            )
        with b3:
            if st.button("🗑️ Delete comparison"):
                try:
                    client.delete_comparison(matrix_choice, delete_runs=delete_runs_too)
                    if (
                        st.session_state.get("selected_comparison_detail", {}).get("matrix_id")
                        == matrix_choice
                    ):
                        st.session_state.pop("selected_comparison_detail", None)
                    st.success("Comparison deleted.")
                    st.rerun()
                except APIError as exc:
                    st.error(f"Delete failed: {exc.detail}")


# ===========================================================================
# COMPARISON DETAIL — radar, bar, scatter, table
# ===========================================================================

detail = st.session_state.get("selected_comparison_detail")

if detail:
    st.divider()
    st.subheader(f"📋 Comparison results — {detail['dataset_name']}")

    close_col, export_col = st.columns([1, 1])
    with close_col:
        if st.button("✖️ Close comparison"):
            del st.session_state["selected_comparison_detail"]
            st.rerun()
    with export_col:
        try:
            csv_bytes = client.export_comparison_csv(detail["matrix_id"])
            st.download_button(
                "📥 Export CSV",
                data=csv_bytes,
                file_name=f"comparison_{detail['matrix_id']}.csv",
                mime="text/csv",
            )
        except APIError:
            pass

    entries = detail["entries"]

    if not entries:
        st.warning("This comparison has no successful model results.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric(
            "🏆 Best model",
            detail["best_model_id"] or "—",
            f"{detail['best_composite_score']:.2f}/5.0" if detail["best_composite_score"] else None,
        )
        h2.metric(
            "⚡ Fastest model",
            detail["fastest_model_id"] or "—",
            f"{detail['fastest_latency_ms']:.0f} ms" if detail["fastest_latency_ms"] else None,
        )
        h3.metric(
            "💰 Cheapest model",
            detail["cheapest_model_id"] or "—",
            f"${detail['cheapest_cost_usd']:.5f}" if detail["cheapest_cost_usd"] is not None else None,
        )

        st.markdown("#### Score comparison")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(_build_radar_chart(entries), use_container_width=True)
        with chart_col2:
            st.plotly_chart(_build_metric_bar_chart(entries), use_container_width=True)

        st.markdown("#### Latency vs. quality")
        st.caption("Marker size reflects total cost. Top-left is the sweet spot: fast and high quality.")
        st.plotly_chart(_build_latency_cost_scatter(entries), use_container_width=True)

        st.markdown("#### Side-by-side table")
        table_df = pd.DataFrame(
            [
                {
                    "Model": e["display_name"],
                    "Provider": e["provider"],
                    "Composite ↑": round(e["composite_mean"], 3),
                    "Faithfulness ↑": round(e["faithfulness_mean"], 3),
                    "Ans. Relevance ↑": round(e["answer_relevance_mean"], 3),
                    "Ctx. Precision ↑": round(e["context_precision_mean"], 3),
                    "Correctness ↑": round(e["correctness_mean"], 3),
                    "Latency (ms) ↓": round(e["avg_latency_ms"], 1),
                    "Cost (USD) ↓": round(e["total_cost_usd"], 5),
                    "N Evaluated": e["n_evaluated"],
                    "Parse Fail %": round(e["parse_failure_rate"] * 100, 1),
                    "⚠": "⚠️" if e["low_sample_warning"] else "",
                }
                for e in entries
            ]
        )
        st.dataframe(
            table_df.style.applymap(_score_color, subset=["Composite ↑"]),
            use_container_width=True,
            hide_index=True,
        )

        if any(e["low_sample_warning"] for e in entries):
            st.caption(
                "⚠️ One or more models have a low sample size — their "
                "aggregate statistics may not be statistically reliable."
            )