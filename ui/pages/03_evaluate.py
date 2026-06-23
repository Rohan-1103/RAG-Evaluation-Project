"""
ui/pages/03_evaluate.py

Single-model evaluation page — configure and run RAG + LLM-as-a-Judge
scoring on a dataset, then inspect per-question results with full
judge reasoning.

Maps directly onto src/api/routes/evaluate.py's endpoints:

    Run form              -> POST /api/v1/evaluate/run
    Run history table       -> GET  /api/v1/evaluate
    Result detail panel      -> GET  /api/v1/evaluate/{run_id}
    "Export CSV" button       -> GET  /api/v1/evaluate/{run_id}/export.csv
    "Delete" button             -> DELETE /api/v1/evaluate/{run_id}

Why this page has no progress bar ticking pair-by-pair during the run
(unlike, say, a naive "processing 3/20..." spinner):

  POST /api/v1/evaluate/run is a single synchronous-from-the-caller's-
  perspective HTTP request — src/api/routes/evaluate.py's
  run_evaluation awaits the ENTIRE RAGPipeline.answer_dataset() +
  EvaluationEngine.arun() pipeline before returning one complete
  EvalReport. There is no intermediate "pair 3 of 20 done" signal
  exposed over this HTTP boundary today (EvaluationEngine.arun()'s
  on_pair_complete callback parameter exists internally, but no route
  in evaluate.py wires it to a streaming response). A single
  st.spinner with an honest caption about expected duration is the
  correct UI for what is, from this page's perspective, one
  long-running atomic call — adding a fake incrementing progress bar
  with no real signal behind it would be actively misleading.

Why force_rerun is surfaced as an explicit checkbox rather than this
page silently retrying with it set to True after a 409:

  A 409 here specifically means "every pair in this dataset already
  has a result" (see run_evaluation's docstring in
  src/api/routes/evaluate.py) — i.e. force_rerun would DISCARD existing
  generated_answer/metric_scores for every pair and re-spend the full
  RAG + judge token cost redoing work that already produced a result.
  That is a decision with real Gemini API cost attached to it on a
  free-tier key — it must be something the user consciously opts into
  by checking a box, never something this page does on their behalf
  as an invisible retry.

Why the per-question drilldown renders ALL 4 metrics' reasoning inline
per row, rather than behind a second "view reasoning" click:

  This drilldown — full judge reasoning visible per question, not just
  a bare score — was named explicitly as the single differentiating
  feature of this entire project (see the original project plan's
  "🔍 Judge Reasoning ... not just a number" and this codebase's
  MetricScore docstring: "a score without reasoning is undebuggable").
  Hiding it behind an extra click on the one page where evaluation
  results are actually inspected would bury the feature the project
  was built around.
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

st.title("⚖️ Run Evaluation")
st.caption(
    "Run a dataset through RAG answering and LLM-as-a-Judge scoring "
    "across Faithfulness, Answer Relevance, Context Precision, and "
    "Correctness."
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

def _score_color(score: float | None) -> str:
    """Traffic-light colouring matching models.yaml's ui.score_bands thresholds."""
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
# RUN FORM
# ===========================================================================

st.subheader("Configure a run")

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
    with st.form("run_evaluation_form"):
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

            model_id = st.text_input(
                "RAG model ID",
                value="gemini-2.0-flash",
                help="Model ID from models.yaml, e.g. gemini-2.0-flash.",
            )

        with col2:
            top_k = st.slider("Top-K retrieval", min_value=1, max_value=20, value=5)
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1
            )
            max_output_tokens = st.slider(
                "Max output tokens", min_value=64, max_value=4096, value=1024
            )

        force_rerun = st.checkbox(
            "Force re-run (reset all pairs and re-score from scratch)",
            value=False,
            help=(
                "Required if this dataset has already been fully "
                "evaluated. Discards existing answers/scores and "
                "re-spends RAG + judge tokens on every pair."
            ),
        )

        submitted = st.form_submit_button("▶️ Run evaluation", type="primary")

    if submitted:
        payload = {
            "dataset_id": selected_dataset["id"],
            "model_id": model_id,
            "collection_name": collection_name,
            "top_k": top_k,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "force_rerun": force_rerun,
        }
        with st.spinner(
            f"Running RAG + judge evaluation on "
            f"{selected_dataset['total_pairs']} pair(s)... "
            f"this typically takes a few seconds per pair."
        ):
            try:
                report = client.run_evaluation(payload)
                st.session_state["last_run_id"] = report["run_id"]
                st.session_state["selected_run_detail"] = report
                st.success(
                    f"✅ Run complete — composite score "
                    f"{report['avg_composite_score']:.2f} / 5.0"
                )
            except APIError as exc:
                if exc.status_code == 409:
                    st.error(
                        f"{exc.detail} Check 'Force re-run' above if you "
                        f"want to re-score this dataset anyway."
                    )
                else:
                    st.error(f"Evaluation failed: {exc.detail}")

# ===========================================================================
# RUN HISTORY
# ===========================================================================

st.divider()
st.subheader("Run history")

dataset_filter_options = ["All"] + list(dataset_options.keys()) if dataset_options else ["All"]
filter_label = st.selectbox("Filter by dataset", dataset_filter_options, key="run_ds_filter")
filter_dataset_id = (
    dataset_options[filter_label]["id"]
    if filter_label != "All" and filter_label in dataset_options
    else None
)

try:
    runs_resp = client.list_runs(dataset_id=filter_dataset_id, limit=50)
except APIError as exc:
    st.error(f"Could not load run history: {exc.detail}")
    runs_resp = None

if runs_resp is not None:
    runs = runs_resp["runs"]
    if not runs:
        st.info("No runs yet. Configure and run an evaluation above.")
    else:
        st.caption(f"{runs_resp['total']} run(s) total.")

        runs_df = pd.DataFrame(
            [
                {
                    "Run ID": r["run_id"][-12:],
                    "Dataset": r["dataset_name"],
                    "Model": r["rag_model"],
                    "Status": r["status"],
                    "Composite": round(r["composite_mean"], 2),
                    "Evaluated": f"{r['n_pairs_evaluated']}/{r['n_pairs_total']}",
                    "Cost ($)": round(r["total_cost_usd"], 5),
                    "Latency (ms)": round(r["avg_total_latency_ms"], 0),
                    "Started": r["started_at"][:19].replace("T", " "),
                    "_full_run_id": r["run_id"],
                }
                for r in runs
            ]
        )

        display_df = runs_df.drop(columns=["_full_run_id"])
        st.dataframe(
            display_df.style.applymap(
                lambda v: _score_color(v) if isinstance(v, (int, float)) and v <= 5 else "",
                subset=["Composite"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        run_choice = st.selectbox(
            "Select a run to inspect",
            options=runs_df["_full_run_id"].tolist(),
            format_func=lambda rid: next(
                f"{r['rag_model']} — {r['started_at'][:19].replace('T', ' ')} "
                f"(composite {r['composite_mean']:.2f})"
                for r in runs
                if r["run_id"] == rid
            ),
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("👁️ View full results"):
                try:
                    detail = client.get_run(run_choice)
                    st.session_state["selected_run_detail"] = detail
                except APIError as exc:
                    st.error(f"Could not load run: {exc.detail}")
        with b2:
            if st.button("🗑️ Delete run"):
                try:
                    client.delete_run(run_choice)
                    if st.session_state.get("last_run_id") == run_choice:
                        st.session_state.pop("selected_run_detail", None)
                    st.success("Run deleted.")
                    st.rerun()
                except APIError as exc:
                    st.error(f"Delete failed: {exc.detail}")


# ===========================================================================
# RUN DETAIL — aggregate stats + per-question drilldown with full reasoning
# ===========================================================================

detail = st.session_state.get("selected_run_detail")

if detail:
    st.divider()
    st.subheader(f"📋 Run results — {detail['rag_model']}")

    close_col, export_col = st.columns([1, 1])
    with close_col:
        if st.button("✖️ Close results"):
            del st.session_state["selected_run_detail"]
            st.rerun()
    with export_col:
        try:
            csv_bytes = client.export_run_csv(detail["run_id"])
            st.download_button(
                "📥 Export CSV",
                data=csv_bytes,
                file_name=f"run_{detail['run_id']}.csv",
                mime="text/csv",
            )
        except APIError:
            pass

    agg = detail.get("aggregate")
    if agg:
        if agg.get("low_sample_warning"):
            st.warning(
                f"⚠️ Only {agg['n_pairs_evaluated']} pair(s) evaluated — "
                f"aggregate statistics may not be statistically reliable."
            )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Composite", f"{agg['composite_mean']:.2f} ± {agg['composite_std']:.2f}")
        for col, metric_name in zip([m2, m3, m4, m5], _METRIC_NAMES):
            stats = agg.get(metric_name)
            if stats:
                col.metric(_METRIC_LABELS[metric_name], f"{stats['mean']:.2f}")
            else:
                col.metric(_METRIC_LABELS[metric_name], "—")

        p1, p2, p3 = st.columns(3)
        p1.metric("Evaluated", f"{agg['n_pairs_evaluated']}/{agg['n_pairs_total']}")
        p2.metric("Avg latency", f"{agg['avg_total_latency_ms']:.0f} ms")
        p3.metric("Total cost", f"${agg['total_cost_usd']:.5f}")

        if agg["overall_parse_failure_rate"] > 0:
            st.caption(
                f"⚠️ {agg['overall_parse_failure_rate']:.0%} of judge "
                f"calls had parse failures — those scores use the "
                f"fallback value and are excluded from the means above."
            )

    st.divider()
    st.markdown("#### Per-question results")
    st.caption(
        "Click a question to see the judge's full reasoning for every metric."
    )

    results = detail.get("results", [])
    if not results:
        st.info("No per-question results available for this run.")
    else:
        for r in results:
            score_str = f"{r['composite_score']:.2f}"
            reliability_flag = "" if r["is_reliable"] else " ⚠️"
            header = f"**{score_str}/5.0**{reliability_flag} — {r['question'][:80]}"

            with st.expander(header, expanded=False):
                st.markdown("**Question**")
                st.write(r["question"])

                qa_col1, qa_col2 = st.columns(2)
                with qa_col1:
                    st.markdown("**Generated answer**")
                    st.write(r["generated_answer"] or "_(empty)_")
                with qa_col2:
                    st.markdown("**Ground truth answer**")
                    st.write(r["ground_truth_answer"])

                if r["retrieved_chunks"]:
                    with st.expander(
                        f"📚 Retrieved context ({len(r['retrieved_chunks'])} chunks)",
                        expanded=False,
                    ):
                        for i, (chunk, source) in enumerate(
                            zip(r["retrieved_chunks"], r["retrieved_chunk_sources"]), 1
                        ):
                            st.caption(f"{i}. [{source}]")
                            st.text(chunk)

                st.markdown("---")
                st.markdown("**Metric scores & judge reasoning**")

                for metric_name in _METRIC_NAMES:
                    ms = r["metric_scores"].get(metric_name)
                    if ms is None:
                        st.text(f"{_METRIC_LABELS[metric_name]}: not evaluated")
                        continue

                    flag = ""
                    if ms["parse_failed"]:
                        flag = " 🔴 parse failed"
                    elif ms["low_confidence"]:
                        flag = " 🟡 low confidence"

                    st.markdown(
                        f"**{_METRIC_LABELS[metric_name]}: {ms['score']:.1f}/5.0**{flag}"
                    )
                    st.caption(ms["reasoning"])

                st.markdown("---")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.caption(f"RAG latency: {r['rag_latency_ms']:.0f} ms")
                perf_col2.caption(f"Eval latency: {r['eval_latency_ms']:.0f} ms")
                perf_col3.caption(f"Cost: ${r['estimated_cost_usd']:.6f}")