"""
src/api/routes/evaluate.py

Single-model evaluation routes — trigger a RAG + LLM-as-a-Judge run on
a dataset, browse run history, and export results.

Endpoints:
    POST   /api/v1/evaluate/run                — run RAG + evaluation for one model
    GET    /api/v1/evaluate                     — list run summaries (filtered, paginated)
    GET    /api/v1/evaluate/summary.csv         — download run summaries as CSV
    GET    /api/v1/evaluate/{run_id}            — full EvalReport, all per-pair results
    GET    /api/v1/evaluate/{run_id}/export.csv — download per-question results as CSV
    DELETE /api/v1/evaluate/{run_id}            — delete a run from history

This route is the single-model counterpart to src/api/routes/compare.py's
multi-model comparison. The distinction that drives every design
decision below: ComparisonRunner._fresh_dataset_copy() (src/comparison/
runner.py) deliberately deep-copies the dataset before each model run,
specifically because running N models against the SAME dataset object
would corrupt QAPair's one-way PENDING -> ANSWERED lifecycle after the
first model. This route has no such N-model concern — there is exactly
one RAG model per call — so it deliberately does the opposite: it
mutates the loaded EvalDataset IN PLACE and persists that mutation back
through DatasetStore.save(). After a successful run, re-fetching the
same dataset_id via GET /api/v1/datasets/{dataset_id} shows pairs with
status=EVALUATED, generated answers, and scores — not the original
PENDING state. This is intentional: an ad-hoc single-model run is
meant to durably advance the dataset's own lifecycle, not produce an
isolated, throwaway result the way a comparison run's per-model copies
do.

Why a dataset with zero PENDING pairs requires force_rerun=true rather
than silently no-op'ing or silently erroring:

  RAGPipeline.answer_dataset() only processes pairs with
  status == PENDING (src/rag/pipeline.py); EvaluationEngine._select_pairs()
  only evaluates pairs with status == ANSWERED (src/evaluation/engine.py).
  If every pair in a dataset is already EVALUATED from a prior run,
  calling this endpoint again does nothing useful by default — silently
  returning an empty-looking "success" would be confusing. force_rerun
  resets every pair back to PENDING (mirroring ComparisonRunner's own
  per-pair reset logic, applied in place rather than to a copy) so the
  exact same dataset can be cleanly re-run, e.g. against a different
  model or different top_k, without needing to regenerate the
  underlying Q&A pairs from scratch.

Why this same force_rerun mechanism also enables a genuinely useful
incremental workflow with NO special-case code: if a dataset has a MIX
of EVALUATED pairs (from an earlier run) and newly-added PENDING pairs
(e.g. a user appended more generated pairs to an existing dataset),
calling this endpoint WITHOUT force_rerun processes only the PENDING
subset. The resulting EvalReport's n_pairs_total reflects only that
subset — previously evaluated pairs keep their existing scores
untouched in the dataset, and are not re-counted in this run's
aggregate. This fell out naturally from respecting QAPair's existing
status-based filtering rather than working around it.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import (
    get_dataset_store,
    get_evaluation_engine,
    get_rag_pipeline,
    get_run_repository,
)
from src.dataset.schema import DatasetStatus, EvalDataset, QAPairStatus
from src.dataset.store import DatasetNotFoundError, DatasetStore, DatasetStoreError
from src.evaluation.engine import EvaluationEngine
from src.evaluation.schema import (
    AggregatedResult,
    EvalReport,
    EvalResult,
    MetricStats,
    RunStatus,
)
from src.rag.pipeline import RAGPipeline
from src.rag.schema import ModelRunConfig
from src.storage.repository import RunRepository, RunSummary

router = APIRouter()

# ===========================================================================
# REQUEST SCHEMAS
# ===========================================================================

class RunEvaluationRequest(BaseModel):
    """Request body for POST /run."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1)
    model_id: str = Field(
        min_length=1,
        description="RAG model ID from models.yaml, e.g. 'gemini-2.0-flash'.",
    )
    collection_name: str = Field(min_length=3, max_length=63)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    force_rerun: bool = Field(
        default=False,
        description=(
            "If true, reset ALL pairs in the dataset back to PENDING "
            "before running, even if some were already evaluated. "
            "Required when the dataset has zero PENDING pairs."
        ),
    )

# ===========================================================================
# RESPONSE SCHEMAS
# ===========================================================================

class MetricScoreOut(BaseModel):
    """API-facing view of one metric's score + judge reasoning."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    score: float
    reasoning: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    parse_failed: bool
    low_confidence: bool

class EvalResultOut(BaseModel):
    """API-facing view of one QAPair's complete evaluation result."""

    model_config = ConfigDict(frozen=True)

    id: str
    pair_id: str
    question: str
    ground_truth_answer: str
    generated_answer: str
    retrieved_chunks: list[str]
    retrieved_chunk_sources: list[str]
    metric_scores: dict[str, MetricScoreOut]
    composite_score: float
    rag_latency_ms: float
    eval_latency_ms: float
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    any_parse_failed: bool
    correctness_skipped: bool
    is_reliable: bool

    @classmethod
    def from_domain(cls, result: EvalResult) -> EvalResultOut:
        return cls(
            id=result.id,
            pair_id=result.pair_id,
            question=result.question,
            ground_truth_answer=result.ground_truth_answer,
            generated_answer=result.generated_answer,
            retrieved_chunks=result.retrieved_chunks,
            retrieved_chunk_sources=result.retrieved_chunk_sources,
            metric_scores={
                name: MetricScoreOut(
                    metric_name=ms.metric_name,
                    score=ms.score,
                    reasoning=ms.reasoning,
                    latency_ms=round(ms.latency_ms, 1),
                    input_tokens=ms.input_tokens,
                    output_tokens=ms.output_tokens,
                    parse_failed=ms.parse_failed,
                    low_confidence=ms.low_confidence,
                )
                for name, ms in result.metric_scores.items()
            },
            composite_score=result.composite_score,
            rag_latency_ms=round(result.rag_latency_ms, 1),
            eval_latency_ms=round(result.eval_latency_ms, 1),
            total_latency_ms=round(result.total_latency_ms, 1),
            total_tokens=result.total_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            any_parse_failed=result.any_parse_failed,
            correctness_skipped=result.correctness_skipped,
            is_reliable=result.is_reliable,
        )

class MetricStatsOut(BaseModel):
    """API-facing view of per-metric aggregate statistics."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    sample_size: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    p25: float
    p75: float
    parse_failure_rate: float
    is_reliable: bool
    ci_95_low: float
    ci_95_high: float

    @classmethod
    def from_domain(cls, stats: MetricStats) -> MetricStatsOut:
        ci_low, ci_high = stats.confidence_interval_95
        return cls(
            metric_name=stats.metric_name,
            sample_size=stats.sample_size,
            mean=stats.mean,
            std=stats.std,
            median=stats.median,
            min=stats.min,
            max=stats.max,
            p25=stats.p25,
            p75=stats.p75,
            parse_failure_rate=stats.parse_failure_rate,
            is_reliable=stats.is_reliable,
            ci_95_low=ci_low,
            ci_95_high=ci_high,
        )

class AggregatedResultOut(BaseModel):
    """API-facing view of run-level aggregate statistics."""

    model_config = ConfigDict(frozen=True)

    n_pairs_total: int
    n_pairs_evaluated: int
    n_pairs_failed: int
    evaluation_rate: float
    faithfulness: MetricStatsOut | None
    answer_relevance: MetricStatsOut | None
    context_precision: MetricStatsOut | None
    correctness: MetricStatsOut | None
    composite_mean: float
    composite_std: float
    composite_median: float
    avg_rag_latency_ms: float
    avg_eval_latency_ms: float
    avg_total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    overall_parse_failure_rate: float
    correctness_skip_rate: float
    low_sample_warning: bool

    @classmethod
    def from_domain(cls, agg: AggregatedResult) -> AggregatedResultOut:
        return cls(
            n_pairs_total=agg.n_pairs_total,
            n_pairs_evaluated=agg.n_pairs_evaluated,
            n_pairs_failed=agg.n_pairs_failed,
            evaluation_rate=round(agg.evaluation_rate, 4),
            faithfulness=(
                MetricStatsOut.from_domain(agg.faithfulness)
                if agg.faithfulness else None
            ),
            answer_relevance=(
                MetricStatsOut.from_domain(agg.answer_relevance)
                if agg.answer_relevance else None
            ),
            context_precision=(
                MetricStatsOut.from_domain(agg.context_precision)
                if agg.context_precision else None
            ),
            correctness=(
                MetricStatsOut.from_domain(agg.correctness)
                if agg.correctness else None
            ),
            composite_mean=agg.composite_mean,
            composite_std=agg.composite_std,
            composite_median=agg.composite_median,
            avg_rag_latency_ms=round(agg.avg_rag_latency_ms, 1),
            avg_eval_latency_ms=round(agg.avg_eval_latency_ms, 1),
            avg_total_latency_ms=round(agg.avg_total_latency_ms, 1),
            total_input_tokens=agg.total_input_tokens,
            total_output_tokens=agg.total_output_tokens,
            total_cost_usd=agg.total_cost_usd,
            overall_parse_failure_rate=agg.overall_parse_failure_rate,
            correctness_skip_rate=agg.correctness_skip_rate,
            low_sample_warning=agg.low_sample_warning,
        )

class EvalReportOut(BaseModel):
    """API-facing view of a complete EvalReport."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    dataset_id: str
    dataset_name: str
    rag_model: str
    judge_model: str
    prompt_version: str
    top_k: int
    temperature: float
    collection_name: str
    started_at: str
    completed_at: str | None
    total_latency_ms: float
    status: str
    error_message: str | None
    n_results: int
    avg_composite_score: float
    aggregate: AggregatedResultOut | None
    results: list[EvalResultOut]

    @classmethod
    def from_domain(cls, report: EvalReport) -> EvalReportOut:
        return cls(
            run_id=report.run_id,
            dataset_id=report.dataset_id,
            dataset_name=report.dataset_name,
            rag_model=report.rag_model,
            judge_model=report.judge_model,
            prompt_version=report.prompt_version,
            top_k=report.top_k,
            temperature=report.temperature,
            collection_name=report.collection_name,
            started_at=report.started_at.isoformat(),
            completed_at=(
                report.completed_at.isoformat()
                if report.completed_at else None
            ),
            total_latency_ms=round(report.total_latency_ms, 1),
            status=report.status.value,
            error_message=report.error_message,
            n_results=report.n_results,
            avg_composite_score=report.avg_composite_score,
            aggregate=(
                AggregatedResultOut.from_domain(report.aggregate)
                if report.aggregate else None
            ),
            results=[
                EvalResultOut.from_domain(r) for r in report.results
            ],
        )

class RunSummaryOut(BaseModel):
    """API-facing view of a RunSummary — for the history list view."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    dataset_id: str
    dataset_name: str
    matrix_id: str | None
    rag_model: str
    judge_model: str
    status: str
    composite_mean: float
    composite_std: float
    n_pairs_evaluated: int
    n_pairs_total: int
    total_cost_usd: float
    avg_total_latency_ms: float
    started_at: str
    completed_at: str | None
    low_sample_warning: bool

    @classmethod
    def from_domain(cls, summary: RunSummary) -> RunSummaryOut:
        return cls(
            run_id=summary.run_id,
            dataset_id=summary.dataset_id,
            dataset_name=summary.dataset_name,
            matrix_id=summary.matrix_id,
            rag_model=summary.rag_model,
            judge_model=summary.judge_model,
            status=summary.status,
            composite_mean=summary.composite_mean,
            composite_std=summary.composite_std,
            n_pairs_evaluated=summary.n_pairs_evaluated,
            n_pairs_total=summary.n_pairs_total,
            total_cost_usd=summary.total_cost_usd,
            avg_total_latency_ms=round(summary.avg_total_latency_ms, 1),
            started_at=summary.started_at.isoformat(),
            completed_at=(
                summary.completed_at.isoformat()
                if summary.completed_at else None
            ),
            low_sample_warning=summary.low_sample_warning,
        )

class RunListOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    runs: list[RunSummaryOut]
    total: int
    limit: int
    offset: int

class DeleteRunOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    deleted: bool

# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

def _parse_run_status_filter(raw: str | None) -> RunStatus | None:
    """Validate a status query param, raising 400 with valid options on error."""
    if raw is None:
        return None
    try:
        return RunStatus(raw)
    except ValueError as exc:
        valid = [s.value for s in RunStatus]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{raw}'. Valid options: {valid}.",
        ) from exc

def _load_dataset_or_404(store: DatasetStore, dataset_id: str) -> EvalDataset:
    """
    Load a dataset by ID, converting DatasetNotFoundError/DatasetStoreError
    into the appropriate HTTPException.

    Identical pattern to src/api/routes/datasets.py's helper of the same
    name — duplicated rather than shared because route-local validation
    helpers are intentionally kept self-contained per route module in
    this codebase (see ingest.py's _validate_collection_name for the
    same convention), avoiding a premature shared "route_utils" module
    for two small functions.
    """
    try:
        return store.load(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except DatasetStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

def _reset_pairs_to_pending(dataset: EvalDataset) -> None:
    """
    Reset every pair in the dataset back to PENDING, in place.

    Mirrors ComparisonRunner._fresh_dataset_copy()'s per-pair reset
    block (src/comparison/runner.py) exactly, EXCEPT it operates
    directly on the given dataset rather than a deep copy — intentional
    here, since this route's entire purpose is to mutate-and-persist
    the one dataset it was given, not isolate N parallel copies for N
    different models the way the comparison runner must.
    """
    for pair in dataset.pairs:
        pair.status = QAPairStatus.PENDING
        pair.generated_answer = None
        pair.retrieved_chunks = []
        pair.retrieved_chunk_sources = []
        pair.rag_latency_ms = 0.0
        pair.rag_input_tokens = 0
        pair.rag_output_tokens = 0
        pair.rag_model = None
        pair.metric_scores = {}
        pair.composite_score = None
        pair.error_message = None

def _map_run_status_to_dataset_status(run_status: RunStatus) -> DatasetStatus:
    """
    Translate a completed run's RunStatus into the dataset's own
    DatasetStatus, applied to the dataset after a run finishes.

    FAILED maps to PARTIAL rather than back to READY: a FAILED
    EvalReport (produced by EvaluationEngine._make_failed_report, see
    src/evaluation/engine.py) only occurs when literally zero pairs
    could be evaluated — but in this route's flow, that pre-flight
    check (batch_result.answered_pairs == 0) already raises a 422
    BEFORE evaluation_engine.arun() is ever called, so FAILED should be
    unreachable here in practice. PARTIAL is the safer default for
    this branch regardless, since it signals "an evaluation attempt
    was made on this dataset" without falsely claiming full success.
    """
    mapping = {
        RunStatus.COMPLETED: DatasetStatus.COMPLETED,
        RunStatus.PARTIAL: DatasetStatus.PARTIAL,
        RunStatus.FAILED: DatasetStatus.PARTIAL,
        RunStatus.RUNNING: DatasetStatus.RUNNING,
        RunStatus.PENDING: DatasetStatus.READY,
    }
    return mapping.get(run_status, DatasetStatus.PARTIAL)

# ===========================================================================
# ROUTES
# ===========================================================================

@router.post(
    "/run",
    response_model=EvalReportOut,
    status_code=status.HTTP_201_CREATED,
    summary="Run RAG answering + LLM-as-a-Judge evaluation for one model",
)
async def run_evaluation(
    request: RunEvaluationRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    evaluation_engine: EvaluationEngine = Depends(get_evaluation_engine),
    dataset_store: DatasetStore = Depends(get_dataset_store),
    repo: RunRepository = Depends(get_run_repository),
) -> EvalReportOut:
    """
    Run the dataset's PENDING pairs through RAGPipeline.answer_dataset()
    then EvaluationEngine.arun(), persist the updated dataset state back
    through DatasetStore, and save the resulting EvalReport via
    RunRepository for history browsing.

    RAGPipeline.answer_dataset() is fully synchronous internally and is
    wrapped in run_in_executor so it never blocks the event loop for
    other concurrent requests — the same pattern already established in
    src/api/routes/ingest.py and src/api/routes/datasets.py.
    EvaluationEngine.arun() is natively async (see src/evaluation/
    engine.py) and is awaited directly with no executor needed.
    """
    loop = asyncio.get_event_loop()

    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, dataset_store, request.dataset_id
    )

    if request.force_rerun:
        _reset_pairs_to_pending(dataset)
        logger.info(
            f"run_evaluation: force_rerun=true — reset all "
            f"{len(dataset.pairs)} pairs in '{dataset.id}' to PENDING."
        )
    else:
        pending_count = sum(
            1 for p in dataset.pairs if p.status == QAPairStatus.PENDING
        )
        if pending_count == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Dataset '{request.dataset_id}' has no PENDING "
                    f"pairs — every pair has already been answered or "
                    f"evaluated. Set force_rerun=true to reset and "
                    f"re-run the entire dataset, e.g. against a "
                    f"different model or retrieval configuration."
                ),
            )

    run_config = ModelRunConfig(
        model_id=request.model_id,
        collection_name=request.collection_name,
        top_k=request.top_k,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        score_threshold=request.score_threshold,
    )

    logger.info(
        f"run_evaluation: Running RAG for dataset '{dataset.id}' "
        f"with model '{request.model_id}'..."
    )

    batch_result = await loop.run_in_executor(
        None, lambda: rag_pipeline.answer_dataset(dataset, run_config)
    )

    if batch_result.answered_pairs == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"RAG pipeline answered 0/{batch_result.total_pairs} "
                f"pairs. Check that collection_name "
                f"'{request.collection_name}' exists and contains "
                f"documents, and that model_id '{request.model_id}' "
                f"is available."
            ),
        )

    logger.info(
        f"run_evaluation: RAG complete "
        f"({batch_result.answered_pairs}/{batch_result.total_pairs} "
        f"answered). Starting evaluation with judge..."
    )

    report = await evaluation_engine.arun(
        dataset=dataset,
        rag_model=request.model_id,
        collection_name=request.collection_name,
        top_k=request.top_k,
        temperature=request.temperature,
    )

    dataset.metadata = dataset.metadata.model_copy(
        update={
            "status": _map_run_status_to_dataset_status(report.status)
        }
    )
    await loop.run_in_executor(None, dataset_store.save, dataset)
    await repo.upsert_dataset_record(
        metadata=dataset.metadata,
        dataset_dir=dataset_store.base_dir / dataset.id,
    )

    await repo.save_run(report)

    logger.info(
        f"run_evaluation: Run '{report.run_id}' complete. "
        f"status={report.status.value}, "
        f"composite_mean={report.avg_composite_score:.2f}, "
        f"n_results={report.n_results}."
    )

    return EvalReportOut.from_domain(report)

@router.get(
    "",
    response_model=RunListOut,
    summary="List evaluation run summaries with filtering and pagination",
)
async def list_runs(
    dataset_id: str | None = Query(default=None),
    rag_model: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    matrix_id: str | None = Query(default=None),
    sort_by: str = Query(default="started_at"),
    descending: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: RunRepository = Depends(get_run_repository),
) -> RunListOut:
    """
    List lightweight run summaries for the history browser.

    Backed by RunRepository.list_runs() (src/storage/repository.py),
    which queries flattened RunRecord columns directly — no full
    EvalReport (with every per-pair EvalResult) is loaded or
    deserialised for this list view. Drilling into one specific run's
    full results is GET /{run_id}, a separate, deliberately heavier
    call.
    """
    parsed_status = _parse_run_status_filter(status_filter)

    summaries = await repo.list_runs(
        dataset_id=dataset_id,
        rag_model=rag_model,
        status=parsed_status,
        matrix_id=matrix_id,
        limit=limit,
        offset=offset,
        order_by=sort_by,
        descending=descending,
    )
    total = await repo.count_runs(
        dataset_id=dataset_id,
        rag_model=rag_model,
        status=parsed_status,
    )

    return RunListOut(
        runs=[RunSummaryOut.from_domain(s) for s in summaries],
        total=total,
        limit=limit,
        offset=offset,
    )

@router.get(
    "/summary.csv",
    summary="Download run summary statistics as CSV",
)
async def export_runs_summary_csv(
    dataset_id: str | None = Query(default=None),
    repo: RunRepository = Depends(get_run_repository),
) -> Response:
    """
    Export run-level aggregate statistics across all (or one dataset's)
    runs as a downloadable CSV — the "Export Comparison" style summary
    table, but for ad-hoc single-model run history rather than a
    ComparisonMatrix.

    Registered BEFORE GET /{run_id} in this file deliberately: Starlette
    matches routes in registration order, and "/summary.csv" must be
    checked as a literal path before "/{run_id}" would otherwise capture
    it as run_id="summary.csv". The same ordering constraint does not
    apply to "/{run_id}/export.csv" below, since that path has an extra
    segment "/{run_id}" alone can never match.
    """
    df = await repo.export_runs_summary_csv(dataset_id=dataset_id)

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No runs found"
                + (f" for dataset_id='{dataset_id}'." if dataset_id else ".")
            ),
        )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    filename = (
        f"runs_summary_{dataset_id}.csv" if dataset_id else "runs_summary.csv"
    )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

@router.get(
    "/{run_id}",
    response_model=EvalReportOut,
    summary="Get a complete evaluation report with all per-pair results",
)
async def get_run(
    run_id: str,
    repo: RunRepository = Depends(get_run_repository),
) -> EvalReportOut:
    """
    Retrieve the full EvalReport, including every EvalResult's question,
    generated answer, retrieved chunks, and per-metric judge reasoning —
    the data source for the dashboard's expandable drilldown table.
    """
    report = await repo.get_run(run_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return EvalReportOut.from_domain(report)

@router.get(
    "/{run_id}/export.csv",
    summary="Download a run's per-question results as CSV",
)
async def export_run_results_csv(
    run_id: str,
    repo: RunRepository = Depends(get_run_repository),
) -> Response:
    """
    Export per-question results (question, answer, all 4 metric scores
    + reasoning) as CSV, matching eval.yaml's report.csv_columns layout.

    RunRepository.export_run_results_csv() already returns a pandas
    DataFrame with no disk write — built directly into an in-memory
    buffer here, consistent with src/api/routes/datasets.py's
    export_dataset_csv side-effect-free download pattern.
    """
    try:
        df = await repo.export_run_results_csv(run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="run_{run_id}_results.csv"'
            )
        },
    )

@router.delete(
    "/{run_id}",
    response_model=DeleteRunOut,
    summary="Delete a run from history",
)
async def delete_run(
    run_id: str,
    repo: RunRepository = Depends(get_run_repository),
) -> DeleteRunOut:
    """
    Permanently delete a RunRecord and all its EvalResultRecords
    (cascade — see src/storage/models.py's ondelete="CASCADE" on
    EvalResultRecord.run_id).

    Does NOT modify the dataset's own state — pairs that were answered
    and scored during this run keep their generated_answer and
    metric_scores in the dataset's own JSON; only this run's entry in
    the SQL-backed history disappears.
    """
    deleted = await repo.delete_run(run_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )
    return DeleteRunOut(run_id=run_id, deleted=True)

__all__ = ["router"]