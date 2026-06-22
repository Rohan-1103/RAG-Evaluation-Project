"""
src/storage/repository.py

RunRepository — the only module permitted to import both ORM models
(src.storage.models) and domain schemas (src.evaluation.schema,
src.dataset.schema) in the same file.

Every other module in the codebase works exclusively with Pydantic
domain objects (EvalReport, AggregatedResult, ComparisonMatrix,
DatasetMetadata). This repository is the translation boundary:
callers pass in / receive Pydantic objects, never SQLAlchemy rows.

Responsibilities:
  - save_run / get_run / list_runs / delete_run
      Persist and retrieve EvalReport objects, reconstructed from
      flattened columns + full_json with zero data loss.
  - save_comparison_matrix / get_comparison_matrix / list_comparison_matrices
      Same pattern for multi-model ComparisonMatrix objects, linking
      constituent RunRecords via matrix_id.
  - upsert_dataset_record / get_dataset_record / list_dataset_records
      Keeps the SQL-queryable dataset index in sync with DatasetStore's
      file-based JSON (called by DatasetStore.save(), not directly by
      UI/API code).
  - export_run_results_csv / export_runs_summary_csv
      pandas DataFrame export matching eval.yaml's csv_columns /
      summary_csv_columns column specs exactly.

Why "create-or-replace" for save_run, not a field-by-field UPDATE:
  An EvalReport is always saved once, atomically, after a complete
  evaluation run finishes (or fails). There is no use case for
  incrementally patching individual fields of a historical run after
  the fact — the run either exists in full or doesn't exist at all.
  Treating save_run as "delete old row + children if present, insert
  fresh" avoids an entire class of partial-update bugs (e.g. stale
  aggregate_json next to updated flattened columns) for a write
  pattern that never needs partial updates in practice.

Why lightweight RunSummary/MatrixSummary dataclasses for list methods:
  Returning full EvalReport objects (with every EvalResult, including
  full retrieved_chunks text) for a "browse last 50 runs" table would
  load and deserialise far more data than the UI ever renders in a
  list view. RunSummary mirrors exactly the columns the dashboard's
  history table displays — get_run() remains the path to the complete
  object when a user drills into one specific run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.dataset.schema import DatasetMetadata, DatasetStatus, GenerationMethod
from src.evaluation.schema import (
    AggregatedResult,
    ComparisonMatrix,
    EvalReport,
    EvalResult,
    ModelComparisonEntry,
    RunStatus,
)
from src.storage.models import (
    ComparisonMatrixRecord,
    DatasetRecord,
    EvalResultRecord,
    RunRecord,
)

# ===========================================================================
# LIGHTWEIGHT SUMMARY TYPES — for list views, decoupled from ORM lifecycle
# ===========================================================================

@dataclass(frozen=True, slots=True)
class RunSummary:
    """
    Lightweight run identity + aggregate metrics for history list views.

    Frozen and slotted — these are read-only display objects with no
    relationship to the originating AsyncSession, safe to hold and
    pass around after the session that produced them has closed.
    """

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
    started_at: datetime
    completed_at: datetime | None
    low_sample_warning: bool

    def __repr__(self) -> str:
        return (
            f"RunSummary(run_id='{self.run_id}', "
            f"model='{self.rag_model}', "
            f"composite={self.composite_mean:.2f}, "
            f"status='{self.status}')"
        )


@dataclass(frozen=True, slots=True)
class MatrixSummary:
    """Lightweight comparison matrix identity for history list views."""

    matrix_id: str
    dataset_id: str
    dataset_name: str
    judge_model: str
    n_models: int
    best_model_id: str | None
    best_composite_score: float | None
    created_at: datetime

    def __repr__(self) -> str:
        return (
            f"MatrixSummary(matrix_id='{self.matrix_id}', "
            f"n_models={self.n_models}, "
            f"best='{self.best_model_id}')"
        )

# ===========================================================================
# RUN REPOSITORY
# ===========================================================================

class RunRepository:
    """
    Persistence boundary for runs, comparison matrices, and the dataset
    index. Wraps a single AsyncSession for the lifetime of one logical
    operation — callers obtain a session from Database.session() /
    session_dependency() and construct a RunRepository(session) per
    request/script invocation. The repository never owns or creates
    its own session.

    Usage (script/Streamlit):
        async with db.session() as session:
            repo = RunRepository(session)
            await repo.save_run(report)
            history = await repo.list_runs(dataset_id="ds_...")

    Usage (FastAPI route):
        @router.get("/runs/{run_id}")
        async def get_run(run_id: str, session: AsyncSession = Depends(get_db)):
            repo = RunRepository(session)
            report = await repo.get_run(run_id)
            if report is None:
                raise HTTPException(404, f"Run '{run_id}' not found")
            return report
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # RUN PERSISTENCE
    # ------------------------------------------------------------------

    async def save_run(self, report: EvalReport) -> str:
        """
        Persist an EvalReport as create-or-replace.

        If run_id already exists, the existing RunRecord and all its
        EvalResultRecords are deleted first (cascade handles results),
        then the report is inserted fresh. This guarantees the
        flattened columns and full_json/aggregate_json blobs can never
        drift out of sync with each other — every save is a complete,
        atomic snapshot.

        Args:
            report: Complete EvalReport, typically with aggregate set.
                    A report with aggregate=None (e.g. a FAILED report
                    from EvaluationEngine's pre-flight checks) is still
                    saved — failed runs belong in history too.

        Returns:
            The run_id that was saved.
        """
        existing = await self._session.get(RunRecord, report.run_id)
        if existing is not None:
            logger.debug(
                f"RunRepository.save_run: '{report.run_id}' exists, "
                f"replacing."
            )
            await self._session.delete(existing)
            await self._session.flush()

        run_record = self._report_to_run_record(report)
        self._session.add(run_record)
        await self._session.flush()

        for result in report.results:
            self._session.add(
                self._eval_result_to_record(result)
            )

        await self._session.flush()

        logger.info(
            f"RunRepository.save_run: Saved run '{report.run_id}' "
            f"({len(report.results)} results, "
            f"status={report.status.value})."
        )
        return report.run_id

    async def get_run(
        self,
        run_id: str,
        include_results: bool = True,
    ) -> EvalReport | None:
        """
        Retrieve a complete EvalReport by run_id.

        Args:
            run_id:          The run to fetch.
            include_results: If False, returns a report with an empty
                             results list — useful when only aggregate
                             stats are needed (e.g. populating a
                             ComparisonMatrix) without the cost of
                             deserialising every EvalResultRecord's
                             full_json.

        Returns:
            EvalReport, or None if run_id does not exist.
        """
        stmt = select(RunRecord).where(RunRecord.run_id == run_id)
        if include_results:
            stmt = stmt.options(selectinload(RunRecord.results))

        record = (await self._session.execute(stmt)).scalar_one_or_none()
        if record is None:
            return None

        return self._run_record_to_report(
            record, results=record.results if include_results else []
        )

    async def list_runs(
        self,
        dataset_id: str | None = None,
        rag_model: str | None = None,
        status: RunStatus | None = None,
        matrix_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "started_at",
        descending: bool = True,
    ) -> list[RunSummary]:
        """
        List run summaries for the history browser, with filtering.

        Args:
            dataset_id: Only runs for this dataset.
            rag_model:  Only runs using this RAG model.
            status:     Only runs with this status.
            matrix_id:  Only runs belonging to this comparison matrix.
            limit:      Max results (pagination).
            offset:     Skip this many results (pagination).
            order_by:   One of "started_at", "composite_mean",
                       "total_cost_usd", "avg_total_latency_ms".
            descending: Sort direction.

        Returns:
            List of RunSummary, lightweight and session-independent.
        """
        stmt = select(RunRecord)

        if dataset_id is not None:
            stmt = stmt.where(RunRecord.dataset_id == dataset_id)
        if rag_model is not None:
            stmt = stmt.where(RunRecord.rag_model == rag_model)
        if status is not None:
            stmt = stmt.where(RunRecord.status == status.value)
        if matrix_id is not None:
            stmt = stmt.where(RunRecord.matrix_id == matrix_id)

        sort_column_map = {
            "started_at": RunRecord.started_at,
            "composite_mean": RunRecord.composite_mean,
            "total_cost_usd": RunRecord.total_cost_usd,
            "avg_total_latency_ms": RunRecord.avg_eval_latency_ms,
        }
        sort_column = sort_column_map.get(
            order_by, RunRecord.started_at
        )
        stmt = stmt.order_by(
            sort_column.desc() if descending else sort_column.asc()
        )
        stmt = stmt.limit(limit).offset(offset)

        records = (await self._session.execute(stmt)).scalars().all()
        return [self._run_record_to_summary(r) for r in records]

    async def count_runs(
        self,
        dataset_id: str | None = None,
        rag_model: str | None = None,
        status: RunStatus | None = None,
    ) -> int:
        """Total run count matching filters, for pagination controls."""
        stmt = select(func.count()).select_from(RunRecord)
        if dataset_id is not None:
            stmt = stmt.where(RunRecord.dataset_id == dataset_id)
        if rag_model is not None:
            stmt = stmt.where(RunRecord.rag_model == rag_model)
        if status is not None:
            stmt = stmt.where(RunRecord.status == status.value)
        return (await self._session.execute(stmt)).scalar_one()

    async def delete_run(self, run_id: str) -> bool:
        """
        Delete a run and all its EvalResultRecords (cascade).

        Returns:
            True if a run was deleted, False if run_id did not exist.
        """
        record = await self._session.get(RunRecord, run_id)
        if record is None:
            return False
        await self._session.delete(record)
        await self._session.flush()
        logger.info(f"RunRepository.delete_run: Deleted '{run_id}'.")
        return True

    async def get_best_run_for_dataset(
        self,
        dataset_id: str,
    ) -> RunSummary | None:
        """
        Return the highest composite_mean run for a dataset.

        Used by the dashboard to highlight "best model so far" without
        requiring the user to manually sort the history table.
        """
        stmt = (
            select(RunRecord)
            .where(RunRecord.dataset_id == dataset_id)
            .where(RunRecord.status == RunStatus.COMPLETED.value)
            .order_by(RunRecord.composite_mean.desc())
            .limit(1)
        )
        record = (await self._session.execute(stmt)).scalar_one_or_none()
        return self._run_record_to_summary(record) if record else None

    # ------------------------------------------------------------------
    # COMPARISON MATRIX PERSISTENCE
    # ------------------------------------------------------------------

    async def save_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
    ) -> str:
        """
        Persist a ComparisonMatrix and link its constituent runs.

        Each EvalReport embedded in matrix.reports is saved via
        save_run() first (create-or-replace, so re-saving a matrix
        that reuses an already-saved run_id is safe), then the
        RunRecord.matrix_id foreign key is set to link it.

        Args:
            matrix: Complete ComparisonMatrix from
                    ComparisonMatrixBuilder.build().

        Returns:
            The matrix_id that was saved.
        """
        existing = await self._session.get(
            ComparisonMatrixRecord, matrix.matrix_id
        )
        if existing is not None:
            logger.debug(
                f"RunRepository.save_comparison_matrix: "
                f"'{matrix.matrix_id}' exists, replacing."
            )
            await self._session.delete(existing)
            await self._session.flush()

        for report in matrix.reports:
            await self.save_run(report)

        best_entry = matrix.best_model_by_composite

        matrix_record = ComparisonMatrixRecord(
            matrix_id=matrix.matrix_id,
            dataset_id=matrix.dataset_id,
            dataset_name=matrix.dataset_name,
            created_at=matrix.created_at,
            judge_model=matrix.judge_model,
            prompt_version=matrix.prompt_version,
            n_models=len(matrix.entries),
            best_model_id=(
                best_entry.rag_model if best_entry else None
            ),
            best_composite_score=(
                best_entry.composite_mean if best_entry else None
            ),
            full_json=matrix.model_dump(mode="json"),
        )
        self._session.add(matrix_record)
        await self._session.flush()

        # Link constituent runs to this matrix
        run_ids = [r.run_id for r in matrix.reports]
        if run_ids:
            stmt = (
                select(RunRecord)
                .where(RunRecord.run_id.in_(run_ids))
            )
            linked_runs = (
                await self._session.execute(stmt)
            ).scalars().all()
            for run_record in linked_runs:
                run_record.matrix_id = matrix.matrix_id
            await self._session.flush()

        logger.info(
            f"RunRepository.save_comparison_matrix: Saved "
            f"'{matrix.matrix_id}' "
            f"({len(matrix.entries)} models linked)."
        )
        return matrix.matrix_id

    async def get_comparison_matrix(
        self,
        matrix_id: str,
    ) -> ComparisonMatrix | None:
        """
        Retrieve a complete ComparisonMatrix by matrix_id.

        Reconstructed directly from full_json — this is the one case
        where the JSON blob is read in preference to rebuilding from
        constituent RunRecords, since the matrix's embedded EvalReports
        (with all per-pair results) are exactly what full_json captured
        and re-querying N runs individually would be both slower and
        redundant.
        """
        record = await self._session.get(
            ComparisonMatrixRecord, matrix_id
        )
        if record is None or record.full_json is None:
            return None
        return ComparisonMatrix.model_validate(record.full_json)

    async def list_comparison_matrices(
        self,
        dataset_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MatrixSummary]:
        """List comparison matrix summaries, newest first."""
        stmt = select(ComparisonMatrixRecord)
        if dataset_id is not None:
            stmt = stmt.where(
                ComparisonMatrixRecord.dataset_id == dataset_id
            )
        stmt = (
            stmt.order_by(ComparisonMatrixRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        records = (await self._session.execute(stmt)).scalars().all()
        return [
            MatrixSummary(
                matrix_id=r.matrix_id,
                dataset_id=r.dataset_id,
                dataset_name=r.dataset_name,
                judge_model=r.judge_model,
                n_models=r.n_models,
                best_model_id=r.best_model_id,
                best_composite_score=r.best_composite_score,
                created_at=r.created_at,
            )
            for r in records
        ]

    async def delete_comparison_matrix(
        self,
        matrix_id: str,
        delete_runs: bool = False,
    ) -> bool:
        """
        Delete a comparison matrix record.

        Args:
            matrix_id:    Matrix to delete.
            delete_runs:  If True, also delete the constituent
                         RunRecords. If False (default), runs remain
                         in history with matrix_id set to NULL
                         (ondelete="SET NULL" on the FK handles this
                         automatically — runs are independently valid).

        Returns:
            True if deleted, False if matrix_id did not exist.
        """
        record = await self._session.get(
            ComparisonMatrixRecord, matrix_id
        )
        if record is None:
            return False

        if delete_runs:
            stmt = delete(RunRecord).where(
                RunRecord.matrix_id == matrix_id
            )
            await self._session.execute(stmt)

        await self._session.delete(record)
        await self._session.flush()
        logger.info(
            f"RunRepository.delete_comparison_matrix: Deleted "
            f"'{matrix_id}' (delete_runs={delete_runs})."
        )
        return True

    # ------------------------------------------------------------------
    # DATASET INDEX SYNC
    # ------------------------------------------------------------------

    async def upsert_dataset_record(
        self,
        metadata: DatasetMetadata,
        dataset_dir: Path,
        evaluated_pairs: int = 0,
    ) -> None:
        """
        Sync the SQL-queryable dataset index with DatasetStore's
        file-based JSON.

        Called by DatasetStore.save() after every JSON write so this
        table never drifts from the authoritative on-disk dataset.
        Upsert semantics: update in place if dataset_id exists,
        insert if new — unlike save_run, datasets ARE updated
        incrementally (status transitions, version bumps) so a
        field-by-field update is correct here, not create-or-replace.
        """
        record = await self._session.get(DatasetRecord, metadata.id)

        if record is None:
            record = DatasetRecord(id=metadata.id)
            self._session.add(record)

        record.name = metadata.name
        record.description = metadata.description
        record.created_at = metadata.created_at
        record.updated_at = metadata.updated_at
        record.generation_method = metadata.generation_method.value
        record.source_collection = metadata.source_collection
        record.source_files = metadata.source_files
        record.generator_model = metadata.generator_model
        record.total_pairs = metadata.total_pairs
        record.status = metadata.status.value
        record.tags = metadata.tags
        record.version = metadata.version
        record.dataset_dir = str(dataset_dir)

        await self._session.flush()
        logger.debug(
            f"RunRepository.upsert_dataset_record: Synced "
            f"'{metadata.id}'."
        )

    async def get_dataset_record(
        self,
        dataset_id: str,
    ) -> DatasetMetadata | None:
        """Retrieve DatasetMetadata reconstructed from the SQL index."""
        record = await self._session.get(DatasetRecord, dataset_id)
        if record is None:
            return None
        return self._dataset_record_to_metadata(record)

    async def list_dataset_records(
        self,
        status: DatasetStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DatasetMetadata]:
        """List dataset metadata from the SQL index, newest first."""
        stmt = select(DatasetRecord)
        if status is not None:
            stmt = stmt.where(DatasetRecord.status == status.value)
        stmt = (
            stmt.order_by(DatasetRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        records = (await self._session.execute(stmt)).scalars().all()
        return [self._dataset_record_to_metadata(r) for r in records]

    async def delete_dataset_record(self, dataset_id: str) -> bool:
        """
        Delete a dataset index entry and all its runs (cascade).

        Does NOT delete the dataset's JSON files on disk — that is
        DatasetStore.delete()'s responsibility. Callers needing a
        complete delete should call both DatasetStore.delete() and
        this method, in either order.
        """
        record = await self._session.get(DatasetRecord, dataset_id)
        if record is None:
            return False
        await self._session.delete(record)
        await self._session.flush()
        return True

    # ------------------------------------------------------------------
    # CSV EXPORT
    # ------------------------------------------------------------------

    async def export_run_results_csv(
        self,
        run_id: str,
    ) -> pd.DataFrame:
        """
        Export per-question results for a run as a DataFrame.

        Column layout matches eval.yaml report.csv_columns exactly.
        Raises ValueError if run_id does not exist or has no results.
        """
        stmt = (
            select(EvalResultRecord)
            .where(EvalResultRecord.run_id == run_id)
            .order_by(EvalResultRecord.evaluated_at)
        )
        records = (await self._session.execute(stmt)).scalars().all()

        if not records:
            raise ValueError(
                f"RunRepository.export_run_results_csv: No results "
                f"found for run_id='{run_id}'. "
                f"Run may not exist or may have produced zero results."
            )

        rows = [
            {
                "question":                  r.question,
                "reference_answer":          r.ground_truth_answer,
                "generated_answer":          r.generated_answer,
                "retrieved_chunks_count":    r.retrieved_chunks_count,
                "faithfulness_score":        r.faithfulness_score,
                "faithfulness_reasoning":    r.faithfulness_reasoning,
                "answer_relevance_score":    r.answer_relevance_score,
                "answer_relevance_reasoning":r.answer_relevance_reasoning,
                "context_precision_score":   r.context_precision_score,
                "context_precision_reasoning": r.context_precision_reasoning,
                "correctness_score":         r.correctness_score,
                "correctness_reasoning":     r.correctness_reasoning,
                "composite_score":           r.composite_score,
                "latency_ms":                r.total_latency_ms,
                "input_tokens":              (
                    r.rag_input_tokens + r.eval_input_tokens
                ),
                "output_tokens":              (
                    r.rag_output_tokens + r.eval_output_tokens
                ),
                "estimated_cost_usd":         r.estimated_cost_usd,
                "parse_failed":               r.any_parse_failed,
                "low_confidence":             (
                    r.faithfulness_score is not None
                    and r.composite_score < 2.0
                ),
            }
            for r in records
        ]
        return pd.DataFrame(rows)

    async def export_runs_summary_csv(
        self,
        dataset_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Export run-level summary statistics as a DataFrame.

        Column layout matches eval.yaml report.summary_csv_columns.
        Used for the "Export Comparison" button on the dashboard.
        """
        stmt = select(RunRecord)
        if dataset_id is not None:
            stmt = stmt.where(RunRecord.dataset_id == dataset_id)
        stmt = stmt.order_by(RunRecord.started_at.desc())

        records = (await self._session.execute(stmt)).scalars().all()

        rows = [
            {
                "model_id":                  r.rag_model,
                "display_name":              r.rag_model,
                "n_questions":                r.n_pairs_evaluated,
                "faithfulness_mean":          r.faithfulness_mean,
                "answer_relevance_mean":      r.answer_relevance_mean,
                "context_precision_mean":     r.context_precision_mean,
                "correctness_mean":           r.correctness_mean,
                "composite_mean":             r.composite_mean,
                "composite_std":              r.composite_std,
                "avg_latency_ms":             r.avg_eval_latency_ms,
                "total_input_tokens":         r.total_input_tokens,
                "total_output_tokens":        r.total_output_tokens,
                "total_cost_usd":             r.total_cost_usd,
            }
            for r in records
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CONVERSION: Pydantic -> ORM
    # ------------------------------------------------------------------

    @staticmethod
    def _report_to_run_record(report: EvalReport) -> RunRecord:
        """Convert EvalReport -> RunRecord (without results children)."""
        agg = report.aggregate

        return RunRecord(
            run_id=report.run_id,
            dataset_id=report.dataset_id,
            dataset_name=report.dataset_name,
            matrix_id=None,  # set later by save_comparison_matrix if applicable
            rag_model=report.rag_model,
            judge_model=report.judge_model,
            prompt_version=report.prompt_version,
            top_k=report.top_k,
            temperature=report.temperature,
            collection_name=report.collection_name,
            started_at=report.started_at,
            completed_at=report.completed_at,
            total_latency_ms=report.total_latency_ms,
            status=report.status.value,
            error_message=report.error_message,
            n_pairs_total=agg.n_pairs_total if agg else 0,
            n_pairs_evaluated=agg.n_pairs_evaluated if agg else 0,
            n_pairs_failed=agg.n_pairs_failed if agg else 0,
            faithfulness_mean=(
                agg.faithfulness.mean
                if agg and agg.faithfulness else None
            ),
            answer_relevance_mean=(
                agg.answer_relevance.mean
                if agg and agg.answer_relevance else None
            ),
            context_precision_mean=(
                agg.context_precision.mean
                if agg and agg.context_precision else None
            ),
            correctness_mean=(
                agg.correctness.mean
                if agg and agg.correctness else None
            ),
            composite_mean=agg.composite_mean if agg else 0.0,
            composite_std=agg.composite_std if agg else 0.0,
            avg_rag_latency_ms=agg.avg_rag_latency_ms if agg else 0.0,
            avg_eval_latency_ms=agg.avg_eval_latency_ms if agg else 0.0,
            total_input_tokens=agg.total_input_tokens if agg else 0,
            total_output_tokens=agg.total_output_tokens if agg else 0,
            total_cost_usd=agg.total_cost_usd if agg else 0.0,
            overall_parse_failure_rate=(
                agg.overall_parse_failure_rate if agg else 0.0
            ),
            low_sample_warning=(
                agg.low_sample_warning if agg else True
            ),
            aggregate_json=(
                agg.model_dump(mode="json") if agg else None
            ),
        )

    @staticmethod
    def _eval_result_to_record(result: EvalResult) -> EvalResultRecord:
        """Convert EvalResult -> EvalResultRecord with flattened scores."""
        faithfulness = result.metric_scores.get("faithfulness")
        answer_relevance = result.metric_scores.get("answer_relevance")
        context_precision = result.metric_scores.get("context_precision")
        correctness = result.metric_scores.get("correctness")

        return EvalResultRecord(
            id=result.id,
            run_id=result.run_id,
            pair_id=result.pair_id,
            dataset_id=result.dataset_id,
            evaluated_at=result.evaluated_at,
            rag_model=result.rag_model,
            judge_model=result.judge_model,
            prompt_version=result.prompt_version,
            question=result.question,
            ground_truth_answer=result.ground_truth_answer,
            generated_answer=result.generated_answer,
            retrieved_chunks_count=len(result.retrieved_chunks),
            faithfulness_score=(
                faithfulness.score if faithfulness else None
            ),
            faithfulness_reasoning=(
                faithfulness.reasoning if faithfulness else None
            ),
            faithfulness_parse_failed=(
                faithfulness.parse_failed if faithfulness else False
            ),
            answer_relevance_score=(
                answer_relevance.score if answer_relevance else None
            ),
            answer_relevance_reasoning=(
                answer_relevance.reasoning if answer_relevance else None
            ),
            answer_relevance_parse_failed=(
                answer_relevance.parse_failed
                if answer_relevance else False
            ),
            context_precision_score=(
                context_precision.score if context_precision else None
            ),
            context_precision_reasoning=(
                context_precision.reasoning
                if context_precision else None
            ),
            context_precision_parse_failed=(
                context_precision.parse_failed
                if context_precision else False
            ),
            correctness_score=(
                correctness.score if correctness else None
            ),
            correctness_reasoning=(
                correctness.reasoning if correctness else None
            ),
            correctness_parse_failed=(
                correctness.parse_failed if correctness else False
            ),
            composite_score=result.composite_score,
            rag_latency_ms=result.rag_latency_ms,
            eval_latency_ms=result.eval_latency_ms,
            total_latency_ms=result.total_latency_ms,
            rag_input_tokens=result.rag_input_tokens,
            rag_output_tokens=result.rag_output_tokens,
            eval_input_tokens=result.eval_input_tokens,
            eval_output_tokens=result.eval_output_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            any_parse_failed=result.any_parse_failed,
            correctness_skipped=result.correctness_skipped,
            retrieved_chunks_json=result.retrieved_chunks,
            retrieved_chunk_sources_json=result.retrieved_chunk_sources,
            full_json=result.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # CONVERSION: ORM -> Pydantic
    # ------------------------------------------------------------------

    @staticmethod
    def _run_record_to_report(
        record: RunRecord,
        results: list[EvalResultRecord],
    ) -> EvalReport:
        """
        Reconstruct EvalReport from RunRecord + its EvalResultRecords.

        aggregate is rebuilt from aggregate_json (exact AggregatedResult,
        including per-metric percentiles that flattened columns don't
        carry) rather than from flattened columns, since aggregate_json
        is the lossless source of truth for that nested object.
        """
        aggregate = (
            AggregatedResult.model_validate(record.aggregate_json)
            if record.aggregate_json
            else None
        )

        eval_results = [
            EvalResult.model_validate(r.full_json)
            for r in results
            if r.full_json is not None
        ]

        return EvalReport(
            run_id=record.run_id,
            dataset_id=record.dataset_id,
            dataset_name=record.dataset_name,
            rag_model=record.rag_model,
            judge_model=record.judge_model,
            prompt_version=record.prompt_version,
            top_k=record.top_k,
            temperature=record.temperature,
            collection_name=record.collection_name,
            started_at=record.started_at,
            completed_at=record.completed_at,
            total_latency_ms=record.total_latency_ms,
            status=RunStatus(record.status),
            error_message=record.error_message,
            results=eval_results,
            aggregate=aggregate,
        )

    @staticmethod
    def _run_record_to_summary(record: RunRecord) -> RunSummary:
        """Reconstruct lightweight RunSummary from RunRecord columns only."""
        return RunSummary(
            run_id=record.run_id,
            dataset_id=record.dataset_id,
            dataset_name=record.dataset_name,
            matrix_id=record.matrix_id,
            rag_model=record.rag_model,
            judge_model=record.judge_model,
            status=record.status,
            composite_mean=record.composite_mean,
            composite_std=record.composite_std,
            n_pairs_evaluated=record.n_pairs_evaluated,
            n_pairs_total=record.n_pairs_total,
            total_cost_usd=record.total_cost_usd,
            avg_total_latency_ms=(
                record.avg_rag_latency_ms + record.avg_eval_latency_ms
            ),
            started_at=record.started_at,
            completed_at=record.completed_at,
            low_sample_warning=record.low_sample_warning,
        )

    @staticmethod
    def _dataset_record_to_metadata(
        record: DatasetRecord,
    ) -> DatasetMetadata:
        """Reconstruct DatasetMetadata from DatasetRecord."""
        return DatasetMetadata(
            id=record.id,
            name=record.name,
            description=record.description,
            created_at=record.created_at,
            updated_at=record.updated_at,
            generation_method=GenerationMethod(record.generation_method),
            source_collection=record.source_collection,
            source_files=record.source_files,
            generator_model=record.generator_model,
            total_pairs=record.total_pairs,
            status=DatasetStatus(record.status),
            tags=record.tags,
            version=record.version,
        )

__all__ = [
    "RunSummary",
    "MatrixSummary",
    "RunRepository",
]