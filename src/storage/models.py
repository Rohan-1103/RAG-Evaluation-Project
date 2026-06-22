"""
src/storage/models.py

SQLAlchemy 2.0 declarative ORM models for run persistence.

Owns the declarative Base — database.py (engine + session factory)
imports Base from here, not the other way around. This avoids circular
imports: models define schema, database.py wires schema to a connection.

Three tables:
  DatasetRecord       — lightweight queryable index of datasets.
                        The full EvalDataset JSON lives in DatasetStore
                        (file-based). This table mirrors DatasetMetadata
                        only, enabling SQL filtering/sorting without
                        loading every dataset.json from disk.

  RunRecord           — one row per EvalReport (one model run on one
                        dataset). The primary object for run history,
                        the "Past Runs" browser, and CSV export.

  EvalResultRecord    — one row per QAPair evaluation within a run.
                        Per-pair drilldown data for the dashboard's
                        expandable table. Metric scores are flattened
                        into columns for SQL-level filtering
                        (e.g. "show me all results where faithfulness < 3")
                        while full_json preserves complete fidelity
                        for exact Pydantic reconstruction.

  ComparisonMatrixRecord — one row per multi-model comparison job.
                          Links to N RunRecords via matrix_id FK.

Design rule: ORM models are PURE PERSISTENCE SCHEMA.
  No business logic, no validation beyond column constraints, no
  Pydantic imports. Conversion between ORM rows and Pydantic schemas
  (EvalReport, AggregatedResult, etc.) lives in repository.py — never
  here. This keeps the persistence layer swappable (SQLite → Postgres)
  without touching domain logic, and keeps domain logic testable
  without a database.

Why flatten + store full_json simultaneously:
  Flattened columns (faithfulness_score, rag_model, started_at) enable
  fast SQL queries: WHERE rag_model = 'gemini-2.0-flash' AND
  composite_mean > 4.0 ORDER BY started_at DESC.
  full_json preserves every field of the original Pydantic object
  (including nested MetricScore.reasoning text) so repository.py can
  reconstruct an exact EvalReport/EvalResult without any data loss —
  flattened columns alone would lose the reasoning text and any future
  schema additions.

Naming convention for constraints:
  Explicit naming_convention on MetaData ensures Alembic autogenerate
  produces stable, predictable constraint names (ix_runs_rag_model,
  fk_eval_results_run_id_runs) instead of database-generated random
  names that differ between SQLite and PostgreSQL — required for
  clean migration diffs when moving from SQLite to Postgres later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON

# ===========================================================================
# NAMING CONVENTION — stable constraint names across SQLite/PostgreSQL
# ===========================================================================

_NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class Base(DeclarativeBase):
    """
    Declarative base for all ORM models in this project.

    database.py imports this Base to create tables via
    Base.metadata.create_all(engine) and to bind the session factory.
    repository.py imports the concrete model classes below, never Base
    directly, to keep query code decoupled from schema wiring.
    """

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)

# ===========================================================================
# DATASET RECORD
# ===========================================================================

class DatasetRecord(Base):
    """
    Queryable index of EvalDatasets.

    Mirrors DatasetMetadata exactly. The authoritative full dataset
    (all QAPairs) lives in DatasetStore as JSON on disk — this table
    exists purely so the UI can list/filter/sort datasets via SQL
    without reading every dataset.json file on every page load.

    Kept in sync by DatasetStore.save() calling
    RunRepository.upsert_dataset_record() after every JSON write.
    If this table and the JSON files ever diverge, DatasetStore's
    rebuild_index() remains the source of truth — this table is a
    derived cache, not the system of record for dataset content.
    """

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    generation_method: Mapped[str] = mapped_column(
        String(20), nullable=False
    )
    source_collection: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    source_files: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    generator_model: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )

    total_pairs: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft"
    )
    tags: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    version: Mapped[str] = mapped_column(
        String(20), nullable=False, default="1.0.0"
    )

    # Path to the dataset's directory under DatasetStore.base_dir.
    # Lets repository.py locate the full JSON without recomputing it
    # from the ID (DatasetStore uses id as the directory name, but
    # storing the path explicitly survives any future change to that
    # convention).
    dataset_dir: Mapped[str] = mapped_column(String(500), nullable=False)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    runs: Mapped[list["RunRecord"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_created_at", "created_at"),
        Index("ix_datasets_generation_method", "generation_method"),
    )

    def __repr__(self) -> str:
        return (
            f"DatasetRecord(id='{self.id}', name='{self.name}', "
            f"pairs={self.total_pairs}, status='{self.status}')"
        )

# ===========================================================================
# RUN RECORD
# ===========================================================================

class RunRecord(Base):
    """
    One row per EvalReport — a single model run on a single dataset.

    The primary table for:
      - "Past Runs" browser in the Streamlit UI
      - Run history queries (all runs for a model, all runs for a dataset)
      - CSV export of the summary_csv_columns spec from eval.yaml
      - Re-loading a historical AggregatedResult without re-running eval

    aggregate_json stores the complete AggregatedResult (including all
    4 MetricStats with percentiles) so repository.py can reconstruct
    the exact Pydantic object for the dashboard without recomputing
    statistics from raw EvalResultRecords every page load.

    matrix_id is nullable: a single ad-hoc evaluation run (one model,
    no comparison) has no matrix. Only runs created as part of
    ComparisonRunner.arun_comparison() are linked to a
    ComparisonMatrixRecord.
    """

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)

    dataset_id: Mapped[str] = mapped_column(
        String(40),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)

    matrix_id: Mapped[str | None] = mapped_column(
        String(40),
        ForeignKey("comparison_matrices.matrix_id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Run configuration
    # ------------------------------------------------------------------

    rag_model: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    collection_name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------
    # Aggregate summary (flattened for SQL filtering/sorting)
    # ------------------------------------------------------------------

    n_pairs_total: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    n_pairs_evaluated: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    n_pairs_failed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    faithfulness_mean: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    answer_relevance_mean: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    context_precision_mean: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    correctness_mean: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    composite_mean: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    composite_std: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    avg_rag_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    avg_eval_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    total_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_cost_usd: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    overall_parse_failure_rate: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    low_sample_warning: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ------------------------------------------------------------------
    # Full fidelity payload
    # ------------------------------------------------------------------

    aggregate_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc=(
            "Complete AggregatedResult.model_dump() — full per-metric "
            "MetricStats including percentiles. Reconstructed exactly "
            "by repository.py without recomputing statistics."
        ),
    )

    # ------------------------------------------------------------------
    # Record metadata
    # ------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    dataset: Mapped["DatasetRecord"] = relationship(back_populates="runs")
    matrix: Mapped["ComparisonMatrixRecord | None"] = relationship(
        back_populates="runs"
    )
    results: Mapped[list["EvalResultRecord"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="EvalResultRecord.evaluated_at",
    )

    __table_args__ = (
        Index("ix_runs_dataset_id", "dataset_id"),
        Index("ix_runs_matrix_id", "matrix_id"),
        Index("ix_runs_rag_model", "rag_model"),
        Index("ix_runs_status", "status"),
        Index("ix_runs_started_at", "started_at"),
        Index("ix_runs_composite_mean", "composite_mean"),
        # Composite index for the most common dashboard query:
        # "all runs for this dataset, sorted by recency"
        Index("ix_runs_dataset_started", "dataset_id", "started_at"),
    )

    def __repr__(self) -> str:
        return (
            f"RunRecord(run_id='{self.run_id}', "
            f"model='{self.rag_model}', "
            f"composite={self.composite_mean:.2f}, "
            f"status='{self.status}')"
        )

# ===========================================================================
# EVAL RESULT RECORD
# ===========================================================================

class EvalResultRecord(Base):
    """
    One row per QAPair evaluation within a run.

    The drilldown table data source: clicking a row in the comparison
    dashboard expands to show question, generated answer, retrieved
    chunks, and per-metric reasoning for that specific pair.

    Metric scores AND reasoning are flattened into columns
    (faithfulness_score, faithfulness_reasoning, ...) rather than a
    single JSON blob, because:
      1. SQL filtering: "show me pairs where faithfulness < 3" is a
         single indexed column comparison, not a JSON path query.
      2. CSV export: pandas.read_sql() produces the exact column
         layout specified in eval.yaml's csv_columns without any
         post-processing/unpacking step.
    full_json retains the complete EvalResult (including judge_model,
    prompt_version per metric, parse_failed/low_confidence flags) for
    exact reconstruction when full fidelity is needed.
    """

    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)

    run_id: Mapped[str] = mapped_column(
        String(40),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    pair_id: Mapped[str] = mapped_column(String(40), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(40), nullable=False)

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    rag_model: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)

    # ------------------------------------------------------------------
    # Q&A content
    # ------------------------------------------------------------------

    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_answer: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # ------------------------------------------------------------------
    # Flattened metric scores + reasoning
    # ------------------------------------------------------------------

    faithfulness_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    faithfulness_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    faithfulness_parse_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    answer_relevance_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    answer_relevance_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    answer_relevance_parse_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    context_precision_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    context_precision_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    context_precision_parse_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    correctness_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    correctness_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    correctness_parse_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    composite_score: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    rag_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    eval_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    total_latency_ms: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    rag_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    rag_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    eval_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    eval_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    estimated_cost_usd: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    # ------------------------------------------------------------------
    # Quality flags
    # ------------------------------------------------------------------

    any_parse_failed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    correctness_skipped: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ------------------------------------------------------------------
    # Full fidelity payload
    # ------------------------------------------------------------------

    retrieved_chunks_json: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Full retrieved chunk text list — for drilldown display.",
    )
    retrieved_chunk_sources_json: Mapped[list[str] | None] = (
        mapped_column(
            JSON,
            nullable=True,
            doc="Citation strings parallel to retrieved_chunks_json.",
        )
    )
    full_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc=(
            "Complete EvalResult.model_dump() for exact reconstruction. "
            "Includes per-metric MetricScore objects with all fields."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    run: Mapped["RunRecord"] = relationship(back_populates="results")

    __table_args__ = (
        Index("ix_eval_results_run_id", "run_id"),
        Index("ix_eval_results_pair_id", "pair_id"),
        Index("ix_eval_results_dataset_id", "dataset_id"),
        Index("ix_eval_results_composite_score", "composite_score"),
        Index(
            "ix_eval_results_faithfulness_score", "faithfulness_score"
        ),
        # Common drilldown query: "all results for this run, worst first"
        Index(
            "ix_eval_results_run_composite", "run_id", "composite_score"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"EvalResultRecord(id='{self.id}', "
            f"run_id='{self.run_id}', "
            f"composite={self.composite_score:.2f})"
        )

# ===========================================================================
# COMPARISON MATRIX RECORD
# ===========================================================================

class ComparisonMatrixRecord(Base):
    """
    One row per multi-model comparison job.

    Links N RunRecords (one per model in the comparison) via
    RunRecord.matrix_id. Stores the complete ComparisonMatrix as
    full_json so the dashboard can re-render a historical comparison
    (radar chart, table, best/cheapest/fastest model) without
    re-querying and rebuilding it from constituent RunRecords.

    A UniqueConstraint on (dataset_id, created_at) is deliberately
    NOT applied — re-running the same dataset through the same models
    at a later time is a valid, common workflow (e.g. comparing
    Tuesday's run against Thursday's after a prompt change), and both
    runs must be independently browsable in history.
    """

    __tablename__ = "comparison_matrices"

    matrix_id: Mapped[str] = mapped_column(String(40), primary_key=True)

    dataset_id: Mapped[str] = mapped_column(
        String(40),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    judge_model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)

    n_models: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    best_model_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    best_composite_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    full_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc=(
            "Complete ComparisonMatrix.model_dump() including all "
            "ModelComparisonEntry rows and embedded EvalReports."
        ),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    runs: Mapped[list["RunRecord"]] = relationship(
        back_populates="matrix"
    )

    __table_args__ = (
        Index("ix_comparison_matrices_dataset_id", "dataset_id"),
        Index("ix_comparison_matrices_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"ComparisonMatrixRecord(matrix_id='{self.matrix_id}', "
            f"n_models={self.n_models}, "
            f"best='{self.best_model_id}')"
        )

__all__ = [
    "Base",
    "DatasetRecord",
    "RunRecord",
    "EvalResultRecord",
    "ComparisonMatrixRecord",
]