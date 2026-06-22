"""
src/api/routes/compare.py

Multi-model comparison routes — run a parameter grid of RAG models
against one dataset via ComparisonRunner, then browse and export
comparison history.

Endpoints:
    POST   /api/v1/compare/run                  — run a multi-model comparison
    GET    /api/v1/compare                       — list comparison matrices (paginated)
    GET    /api/v1/compare/{matrix_id}           — full comparison matrix
    GET    /api/v1/compare/{matrix_id}/export.csv — download the comparison table as CSV
    DELETE /api/v1/compare/{matrix_id}           — delete a comparison matrix

This is the multi-model counterpart to src/api/routes/evaluate.py's
single-model run. The distinction that drives this file's design:
ComparisonRunner._fresh_dataset_copy() (src/comparison/runner.py)
deep-copies the dataset once per model config specifically so that N
models can be compared against the SAME underlying question set
without one model's RAGPipeline.answer_dataset() call corrupting
QAPair's one-way PENDING -> ANSWERED lifecycle for the next model.
Because of that internal isolation, this route — unlike evaluate.py's
run_evaluation — never mutates or re-saves the dataset it loads. The
dataset loaded here is purely a read-only template; every durable
result of running this endpoint lives in the returned ComparisonMatrix
and the RunRecords/EvalResultRecords persisted underneath it via
RunRepository, never in the dataset's own JSON file.

Why max_output_tokens is patched onto grid configs via model_copy()
rather than threaded through ComparisonRunner.build_grid_configs()'s
own signature:

  build_grid_configs() (src/comparison/runner.py) was written to expand
  exactly the dimensions defined in models.yaml's comparison_grid
  block — model_ids x top_k_values x temperatures — and deliberately
  leaves every other ModelRunConfig field at its Pydantic default. Add-
  ing a max_output_tokens parameter to that helper's signature would
  couple it to one more axis no comparison_grid config actually varies
  in practice. Since ModelRunConfig is a frozen Pydantic model,
  model_copy(update={...}) is the standard, supported way to derive a
  variant of an already-constructed frozen instance — patching the one
  field this route's request body actually allows the caller to
  override is simpler than widening a helper meant to stay aligned with
  the YAML-driven grid concept it mirrors.

Why model_ids, top_k_values, and temperatures are validated for
uniqueness at the request-schema level (Pydantic field_validators
below) rather than relying on ComparisonRunner._deduplicate_configs():

  _deduplicate_configs() exists in the runner to protect against
  config_hash collisions arising from genuinely independent callers
  (e.g. a Streamlit session and a script both queuing overlapping
  work) — it is a safety net, not a substitute for clean input. If
  this route allowed duplicate model_ids through, the resulting grid
  would silently produce fewer ComparisonMatrix entries than the
  request implied, with no error — confusing for an API caller who
  has no visibility into the runner's internal dedup step. Rejecting
  duplicates with a clear 422 here means the configs list arriving at
  ComparisonRunner.arun_comparison() is already guaranteed dedup-free,
  making the ValueError it raises on "empty after deduplication"
  effectively unreachable through this specific route — by design, the
  same defensive-but-unreachable pattern already used for the FAILED
  RunStatus branch in evaluate.py's _map_run_status_to_dataset_status.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.dependencies import (
    get_comparison_runner,
    get_dataset_store,
    get_run_repository,
)
from src.comparison.runner import ComparisonRunner
from src.dataset.schema import EvalDataset
from src.dataset.store import DatasetNotFoundError, DatasetStore, DatasetStoreError
from src.evaluation.schema import ComparisonMatrix, ModelComparisonEntry
from src.storage.repository import MatrixSummary, RunRepository

router = APIRouter()

# ===========================================================================
# REQUEST SCHEMA
# ===========================================================================

class RunComparisonRequest(BaseModel):
    """
    Request body for POST /run.

    model_ids x top_k_values x temperatures forms a cartesian product
    of ModelRunConfig — mirrors ComparisonRunner.build_grid_configs()'s
    own expansion exactly (see this file's module docstring). With the
    defaults below (single top_k, single temperature), a request with
    3 model_ids produces exactly 3 configs — one per model, the common
    case of "compare these models head-to-head under one fixed
    retrieval setting."
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1)
    collection_name: str = Field(min_length=3, max_length=63)
    model_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Models to compare. Each model_id must be unique.",
    )
    top_k_values: list[int] | None = Field(
        default=None,
        description="Defaults to [5] if omitted. Each value must be unique.",
    )
    temperatures: list[float] | None = Field(
        default=None,
        description="Defaults to [0.0] if omitted. Each value must be unique.",
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)
    dataset_name_override: str | None = Field(default=None, max_length=255)

    @field_validator("model_ids")
    @classmethod
    def unique_model_ids(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            dupes = sorted({m for m in v if v.count(m) > 1})
            raise ValueError(
                f"Duplicate model_ids: {dupes}. Each model may appear "
                f"only once per comparison request."
            )
        return v

    @field_validator("top_k_values")
    @classmethod
    def valid_top_k_values(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        for k in v:
            if not 1 <= k <= 20:
                raise ValueError(
                    f"top_k value {k} is outside valid range [1, 20]."
                )
        if len(v) != len(set(v)):
            raise ValueError("Duplicate top_k values are not allowed.")
        return v

    @field_validator("temperatures")
    @classmethod
    def valid_temperatures(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        for t in v:
            if not 0.0 <= t <= 2.0:
                raise ValueError(
                    f"Temperature {t} is outside valid range [0.0, 2.0]."
                )
        if len(v) != len(set(v)):
            raise ValueError("Duplicate temperature values are not allowed.")
        return v

# ===========================================================================
# RESPONSE SCHEMAS
# ===========================================================================

class ModelComparisonEntryOut(BaseModel):
    """API-facing view of one model's row in the comparison table."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    rag_model: str
    display_name: str
    provider: str
    faithfulness_mean: float
    answer_relevance_mean: float
    context_precision_mean: float
    correctness_mean: float
    composite_mean: float
    faithfulness_std: float
    answer_relevance_std: float
    context_precision_std: float
    correctness_std: float
    composite_std: float
    avg_latency_ms: float
    total_cost_usd: float
    n_evaluated: int
    parse_failure_rate: float
    low_sample_warning: bool
    radar: dict[str, float]

    @classmethod
    def from_domain(cls, entry: ModelComparisonEntry) -> ModelComparisonEntryOut:
        return cls(
            run_id=entry.run_id,
            rag_model=entry.rag_model,
            display_name=entry.display_name,
            provider=entry.provider,
            faithfulness_mean=entry.faithfulness_mean,
            answer_relevance_mean=entry.answer_relevance_mean,
            context_precision_mean=entry.context_precision_mean,
            correctness_mean=entry.correctness_mean,
            composite_mean=entry.composite_mean,
            faithfulness_std=entry.faithfulness_std,
            answer_relevance_std=entry.answer_relevance_std,
            context_precision_std=entry.context_precision_std,
            correctness_std=entry.correctness_std,
            composite_std=entry.composite_std,
            avg_latency_ms=round(entry.avg_latency_ms, 1),
            total_cost_usd=entry.total_cost_usd,
            n_evaluated=entry.n_evaluated,
            parse_failure_rate=round(entry.parse_failure_rate, 4),
            low_sample_warning=entry.low_sample_warning,
            radar=entry.radar_values,
        )

class ComparisonMatrixOut(BaseModel):
    """
    API-facing view of a complete ComparisonMatrix.

    Deliberately omits the embedded EvalReport objects (matrix.reports)
    — exposing run_ids instead, so a client that needs one specific
    model's full per-pair results (questions, generated answers,
    retrieved chunks, judge reasoning) drills in via the existing
    GET /api/v1/evaluate/{run_id} from evaluate.py, rather than this
    endpoint duplicating that entire heavy payload N times over for an
    N-model comparison. The lightweight-summary-vs-heavy-detail split
    already established between DatasetSummaryOut/DatasetDetailOut and
    RunSummaryOut/EvalReportOut is applied identically here.
    """

    model_config = ConfigDict(frozen=True)

    matrix_id: str
    dataset_id: str
    dataset_name: str
    created_at: str
    judge_model: str
    prompt_version: str
    n_models: int
    model_ids: list[str]
    run_ids: list[str]
    entries: list[ModelComparisonEntryOut]
    best_model_id: str | None
    best_composite_score: float | None
    fastest_model_id: str | None
    fastest_latency_ms: float | None
    cheapest_model_id: str | None
    cheapest_cost_usd: float | None
    has_multiple_models: bool

    @classmethod
    def from_domain(cls, matrix: ComparisonMatrix) -> ComparisonMatrixOut:
        best = matrix.best_model_by_composite
        fastest = matrix.fastest_model
        cheapest = matrix.cheapest_model

        return cls(
            matrix_id=matrix.matrix_id,
            dataset_id=matrix.dataset_id,
            dataset_name=matrix.dataset_name,
            created_at=matrix.created_at.isoformat(),
            judge_model=matrix.judge_model,
            prompt_version=matrix.prompt_version,
            n_models=len(matrix.entries),
            model_ids=matrix.model_ids,
            run_ids=[r.run_id for r in matrix.reports],
            entries=[
                ModelComparisonEntryOut.from_domain(e)
                for e in matrix.entries
            ],
            best_model_id=best.rag_model if best else None,
            best_composite_score=best.composite_mean if best else None,
            fastest_model_id=fastest.rag_model if fastest else None,
            fastest_latency_ms=fastest.avg_latency_ms if fastest else None,
            cheapest_model_id=cheapest.rag_model if cheapest else None,
            cheapest_cost_usd=cheapest.total_cost_usd if cheapest else None,
            has_multiple_models=matrix.has_multiple_models,
        )

class MatrixSummaryOut(BaseModel):
    """API-facing view of a MatrixSummary — for the history list view."""

    model_config = ConfigDict(frozen=True)

    matrix_id: str
    dataset_id: str
    dataset_name: str
    judge_model: str
    n_models: int
    best_model_id: str | None
    best_composite_score: float | None
    created_at: str

    @classmethod
    def from_domain(cls, summary: MatrixSummary) -> MatrixSummaryOut:
        return cls(
            matrix_id=summary.matrix_id,
            dataset_id=summary.dataset_id,
            dataset_name=summary.dataset_name,
            judge_model=summary.judge_model,
            n_models=summary.n_models,
            best_model_id=summary.best_model_id,
            best_composite_score=summary.best_composite_score,
            created_at=summary.created_at.isoformat(),
        )

class MatrixListOut(BaseModel):
    """
    Response for GET /compare.

    Deliberately has no `total` field. RunRepository exposes
    count_runs() for the single-run history list (see evaluate.py's
    RunListOut), but has no equivalent count_comparison_matrices()
    method yet — adding one would be the natural follow-up if exact
    total counts become necessary for pagination UI controls. Returning
    a fabricated or expensively-recomputed total here would misrepre-
    sent what this endpoint can currently guarantee.
    """

    model_config = ConfigDict(frozen=True)

    matrices: list[MatrixSummaryOut]
    limit: int
    offset: int

class DeleteMatrixOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    matrix_id: str
    deleted: bool
    runs_also_deleted: bool

# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

def _load_dataset_or_404(store: DatasetStore, dataset_id: str) -> EvalDataset:
    """
    Load a dataset by ID, converting DatasetNotFoundError/DatasetStoreError
    into the appropriate HTTPException.

    Identical pattern to the same-named helper in datasets.py and
    evaluate.py — kept route-local rather than shared, consistent with
    this codebase's established convention of small, self-contained
    per-route validation helpers (see ingest.py's
    _validate_collection_name for the precedent).
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

# ===========================================================================
# ROUTES
# ===========================================================================

@router.post(
    "/run",
    response_model=ComparisonMatrixOut,
    status_code=status.HTTP_201_CREATED,
    summary="Run a multi-model comparison across one or more RAG models",
)
async def run_comparison(
    request: RunComparisonRequest,
    dataset_store: DatasetStore = Depends(get_dataset_store),
    comparison_runner: ComparisonRunner = Depends(get_comparison_runner),
    repo: RunRepository = Depends(get_run_repository),
) -> ComparisonMatrixOut:
    """
    Load the dataset as a read-only template, expand the requested
    model/top_k/temperature grid into ModelRunConfigs, run them all
    concurrently via ComparisonRunner.arun_comparison(), persist the
    resulting ComparisonMatrix (and every constituent EvalReport)
    through RunRepository, and return the comparison table.

    ComparisonRunnerError (raised if every model config fails, or if
    the expanded grid exceeds models.yaml's comparison_grid.max_total_runs
    safety cap) is deliberately NOT caught here — it propagates to the
    handler registered in src/api/app.py, which is the single source
    of truth for that exception's HTTP representation, exactly as
    GenerationError is left to propagate in datasets.py's
    generate_dataset for the identical reason.
    """
    loop = asyncio.get_event_loop()

    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, dataset_store, request.dataset_id
    )

    base_configs = ComparisonRunner.build_grid_configs(
        model_ids=request.model_ids,
        collection_name=request.collection_name,
        top_k_values=request.top_k_values,
        temperatures=request.temperatures,
        score_threshold=request.score_threshold,
    )
    configs = [
        config.model_copy(
            update={"max_output_tokens": request.max_output_tokens}
        )
        for config in base_configs
    ]

    logger.info(
        f"run_comparison: dataset='{dataset.id}', "
        f"n_configs={len(configs)}, models={request.model_ids}."
    )

    matrix = await comparison_runner.arun_comparison(
        dataset=dataset,
        configs=configs,
        dataset_name_override=request.dataset_name_override,
    )

    await repo.save_comparison_matrix(matrix)

    logger.info(
        f"run_comparison: matrix '{matrix.matrix_id}' saved "
        f"({len(matrix.entries)}/{len(configs)} models succeeded). "
        f"Best: '{matrix.best_model_by_composite.rag_model if matrix.best_model_by_composite else None}'."
    )

    return ComparisonMatrixOut.from_domain(matrix)

@router.get(
    "",
    response_model=MatrixListOut,
    summary="List comparison matrices",
)
async def list_comparisons(
    dataset_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: RunRepository = Depends(get_run_repository),
) -> MatrixListOut:
    """
    List comparison matrix summaries, newest first.

    Backed by RunRepository.list_comparison_matrices() — queries the
    ComparisonMatrixRecord table directly via flattened columns
    (best_model_id, n_models, etc.); no embedded EvalReports or
    full_json blobs are deserialised for this list view.
    """
    summaries = await repo.list_comparison_matrices(
        dataset_id=dataset_id, limit=limit, offset=offset
    )
    return MatrixListOut(
        matrices=[MatrixSummaryOut.from_domain(s) for s in summaries],
        limit=limit,
        offset=offset,
    )

@router.get(
    "/{matrix_id}",
    response_model=ComparisonMatrixOut,
    summary="Get a complete comparison matrix",
)
async def get_comparison(
    matrix_id: str,
    repo: RunRepository = Depends(get_run_repository),
) -> ComparisonMatrixOut:
    """
    Retrieve the full comparison table for one matrix — the data
    source for the dashboard's radar chart, bar chart, and side-by-side
    table.

    Reconstructed by RunRepository.get_comparison_matrix() directly
    from the ComparisonMatrixRecord's full_json column (see
    src/storage/repository.py's docstring on why this is the one read
    path that prefers the JSON blob over rebuilding from constituent
    RunRecords).
    """
    matrix = await repo.get_comparison_matrix(matrix_id)
    if matrix is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison matrix '{matrix_id}' not found.",
        )
    return ComparisonMatrixOut.from_domain(matrix)

@router.get(
    "/{matrix_id}/export.csv",
    summary="Download the comparison table as CSV",
)
async def export_comparison_csv(
    matrix_id: str,
    repo: RunRepository = Depends(get_run_repository),
) -> Response:
    """
    Export the comparison table (one row per model, matching
    ModelComparisonEntry.to_table_row()'s column layout) as a
    downloadable CSV.

    Built in-memory from ComparisonMatrix.to_comparison_table_rows() —
    no file is written to the server's filesystem, consistent with the
    side-effect-free CSV download pattern already used in
    datasets.py's export_dataset_csv and evaluate.py's
    export_run_results_csv.
    """
    matrix = await repo.get_comparison_matrix(matrix_id)
    if matrix is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison matrix '{matrix_id}' not found.",
        )

    rows = matrix.to_comparison_table_rows()
    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="comparison_{matrix_id}.csv"'
            )
        },
    )

@router.delete(
    "/{matrix_id}",
    response_model=DeleteMatrixOut,
    summary="Delete a comparison matrix",
)
async def delete_comparison(
    matrix_id: str,
    delete_runs: bool = Query(
        default=False,
        description=(
            "If true, also delete every constituent RunRecord and its "
            "EvalResultRecords. If false (default), runs remain "
            "independently browsable in single-run history with their "
            "matrix_id link cleared — see "
            "RunRecord.matrix_id's ondelete='SET NULL' foreign key in "
            "src/storage/models.py."
        ),
    ),
    repo: RunRepository = Depends(get_run_repository),
) -> DeleteMatrixOut:
    """
    Delete a ComparisonMatrixRecord, optionally cascading to its
    constituent runs.

    Defaults to preserving runs: a comparison job is a view over runs
    that happened to execute together, not the runs' reason for
    existing — clearing comparison history should not silently erase
    individually-valid evaluation runs unless the caller explicitly
    asks for that with delete_runs=true.
    """
    deleted = await repo.delete_comparison_matrix(
        matrix_id, delete_runs=delete_runs
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison matrix '{matrix_id}' not found.",
        )
    return DeleteMatrixOut(
        matrix_id=matrix_id,
        deleted=True,
        runs_also_deleted=delete_runs,
    )

__all__ = ["router"]