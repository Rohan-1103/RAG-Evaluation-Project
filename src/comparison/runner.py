"""
src/comparison/runner.py

ComparisonRunner — orchestrates multi-model RAG + evaluation comparison.

Flow:
    EvalDataset (template, never mutated directly)
    list[ModelRunConfig]
        ↓ For each config (respecting max_concurrent_runs semaphore):
            ↓ deep copy dataset (pairs reset to PENDING)
            ↓ RAGPipeline.answer_dataset(copy, config)
            ↓ EvaluationEngine.arun(copy, config.model_id, ...)
            ↓ EvalReport
        ↓ ComparisonMatrixBuilder.build(all_reports)
    ComparisonMatrix

Why deep copy is mandatory:
  QAPair.set_answer() enforces a PENDING → ANSWERED transition exactly
  once. Running the SAME dataset object through 3 models would corrupt
  state after the first model run — the second model's answer_pair()
  call would raise ValueError because status is no longer PENDING.
  Each model gets its own independent copy of the dataset so RAG
  answers and eval scores never collide across models.

Concurrency model:
  - Model runs (RAG + eval for one model) run concurrently up to
    max_concurrent_runs (from comparison.max_concurrent_runs in .env).
  - Within one model run, EvaluationEngine internally manages its own
    max_concurrent_eval_calls semaphore for judge calls.
  - This two-level semaphore design prevents the comparison runner
    from overwhelming the API even when comparing many models at once.

Deduplication:
  ModelRunConfig.config_hash detects duplicate (model, top_k, temp,
  collection) combinations before execution. Duplicate configs are
  skipped with a warning rather than wasting API calls on identical work.

Safety cap:
  comparison_grid.max_total_runs from models.yaml caps the number of
  runs that can be queued in a single comparison job. Exceeding it
  raises ComparisonRunnerError before any API calls are made.
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any

from loguru import logger

from config.settings import Settings
from src.dataset.schema import EvalDataset, QAPairStatus
from src.evaluation.engine import ComparisonMatrixBuilder, EvaluationEngine
from src.evaluation.schema import ComparisonMatrix, EvalReport
from src.rag.pipeline import RAGPipeline
from src.rag.schema import ModelRunConfig

# ===========================================================================
# MODEL RUN RESULT
# ===========================================================================

class ModelRunResult:
    """
    Result of running one ModelRunConfig through RAG + evaluation.

    Bundles the EvalReport with execution metadata (success/failure,
    error message) so the ComparisonRunner can report partial failures
    without losing track of which configs succeeded.

    Using __slots__ — this object is created once per config and never
    needs dynamic attributes, so slots save memory at scale.
    """

    __slots__ = (
        "config",
        "report",
        "succeeded",
        "error_message",
        "wall_latency_ms",
    )

    def __init__(
        self,
        config: ModelRunConfig,
        report: EvalReport | None,
        succeeded: bool,
        error_message: str | None = None,
        wall_latency_ms: float = 0.0,
    ) -> None:
        self.config = config
        self.report = report
        self.succeeded = succeeded
        self.error_message = error_message
        self.wall_latency_ms = wall_latency_ms

    def __repr__(self) -> str:
        return (
            f"ModelRunResult("
            f"model='{self.config.model_id}', "
            f"succeeded={self.succeeded}, "
            f"latency={self.wall_latency_ms:.0f}ms)"
        )

# ===========================================================================
# COMPARISON RUNNER
# ===========================================================================

class ComparisonRunner:
    """
    Orchestrates running an EvalDataset through multiple model configs
    and produces a side-by-side ComparisonMatrix.

    Constructor injection — RAGPipeline and EvaluationEngine are passed
    in, not constructed internally. The runner is agnostic to which
    vector store or embedding model backs the pipeline.

    Usage:
        runner = ComparisonRunner.from_settings(settings)
        matrix = await runner.arun_comparison(
            dataset=my_dataset,
            configs=[
                ModelRunConfig(model_id="gemini-2.0-flash", collection_name="docs", top_k=5),
                ModelRunConfig(model_id="gemini-1.5-flash-8b", collection_name="docs", top_k=5),
            ],
        )
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        evaluation_engine: EvaluationEngine,
        settings: Settings,
        matrix_builder: ComparisonMatrixBuilder | None = None,
    ) -> None:
        """
        Initialise the comparison runner.

        Args:
            rag_pipeline:      Shared RAGPipeline instance — reused
                               across all model configs in a comparison.
            evaluation_engine: Shared EvaluationEngine instance —
                               judge model stays constant across the
                               comparison even as the RAG model varies.
            settings:          Application settings for concurrency limits.
            matrix_builder:    Optional override for testing.
        """
        self._rag_pipeline = rag_pipeline
        self._evaluation_engine = evaluation_engine
        self._settings = settings
        self._matrix_builder = matrix_builder or ComparisonMatrixBuilder()

        logger.info(
            f"ComparisonRunner initialised. "
            f"max_concurrent_runs={settings.comparison.max_concurrent_runs}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        rag_pipeline: RAGPipeline | None = None,
        evaluation_engine: EvaluationEngine | None = None,
    ) -> ComparisonRunner:
        """
        Standard factory — builds RAGPipeline and EvaluationEngine
        from Settings if not provided.

        Passing pre-built instances allows reuse across multiple
        comparison runs without reloading the embedding model or
        reconstructing all 4 evaluators each time.
        """
        resolved_rag = rag_pipeline or RAGPipeline.from_settings(settings)
        resolved_engine = (
            evaluation_engine or EvaluationEngine.from_settings(settings)
        )
        return cls(
            rag_pipeline=resolved_rag,
            evaluation_engine=resolved_engine,
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def arun_comparison(
        self,
        dataset: EvalDataset,
        configs: list[ModelRunConfig],
        dataset_name_override: str | None = None,
        on_run_complete: Any | None = None,
    ) -> ComparisonMatrix:
        """
        Run the full comparison across all configs concurrently.

        Args:
            dataset:                Template EvalDataset. NEVER mutated
                                    directly — each config receives its
                                    own deep copy with pairs reset to
                                    PENDING before RAG answering.
            configs:                List of ModelRunConfig to compare.
                                    Deduplicated by config_hash before
                                    execution.
            dataset_name_override:  Optional display name for the matrix.
            on_run_complete:        Optional async callback
                                    (config_idx, total, ModelRunResult)
                                    for live UI progress across runs.

        Returns:
            ComparisonMatrix with one entry per successfully completed
            config, sorted by composite_mean descending.

        Raises:
            ValueError: if configs is empty after deduplication.
            ComparisonRunnerError: if total configs exceed
                                   max_total_runs, or if every config
                                   fails (no matrix can be built).
        """
        deduped_configs = self._deduplicate_configs(configs)

        if not deduped_configs:
            raise ValueError(
                "ComparisonRunner.arun_comparison: configs list is empty "
                "after deduplication. Provide at least one ModelRunConfig."
            )

        self._enforce_max_total_runs(deduped_configs)

        logger.info(
            f"ComparisonRunner: Starting comparison. "
            f"dataset='{dataset.name}', "
            f"n_configs={len(deduped_configs)}, "
            f"models={[c.model_id for c in deduped_configs]}"
        )

        semaphore = asyncio.Semaphore(
            self._settings.comparison.max_concurrent_runs
        )
        total = len(deduped_configs)

        async def run_one(idx: int, config: ModelRunConfig) -> ModelRunResult:
            async with semaphore:
                result = await self._run_single_config(dataset, config)
                logger.info(
                    f"ComparisonRunner: Config {idx + 1}/{total} "
                    f"('{config.display_label}') complete. "
                    f"succeeded={result.succeeded}"
                )
                if on_run_complete is not None:
                    try:
                        await on_run_complete(idx + 1, total, result)
                    except Exception as cb_exc:
                        logger.warning(
                            f"ComparisonRunner: on_run_complete callback "
                            f"failed: {cb_exc}"
                        )
                return result

        tasks = [
            run_one(idx, config)
            for idx, config in enumerate(deduped_configs)
        ]
        results: list[ModelRunResult] = await asyncio.gather(*tasks)

        successful_reports = [
            r.report for r in results
            if r.succeeded and r.report is not None
        ]
        failed = [r for r in results if not r.succeeded]

        if failed:
            logger.warning(
                f"ComparisonRunner: {len(failed)}/{total} configs failed. "
                f"Failures: "
                f"{[(r.config.model_id, r.error_message) for r in failed]}"
            )

        if not successful_reports:
            raise ComparisonRunnerError(
                reason=(
                    f"All {total} model configs failed. "
                    f"No comparison matrix can be built. "
                    f"See logs for per-model error details."
                ),
            )

        from config import get_eval_config

        matrix = self._matrix_builder.build(
            reports=successful_reports,
            dataset_id=dataset.id,
            dataset_name=dataset_name_override or dataset.name,
            judge_model=self._settings.judge.model,
            prompt_version=get_eval_config().active_prompt_version,
        )

        logger.info(
            f"ComparisonRunner: Comparison complete. "
            f"matrix_id='{matrix.matrix_id}', "
            f"successful={len(successful_reports)}/{total}"
        )

        return matrix

    def run_comparison(
        self,
        dataset: EvalDataset,
        configs: list[ModelRunConfig],
        dataset_name_override: str | None = None,
    ) -> ComparisonMatrix:
        """
        Synchronous wrapper around arun_comparison().

        Use for CLI scripts and tests. Streamlit and FastAPI should
        prefer arun_comparison() directly within an async context.
        """
        return asyncio.run(
            self.arun_comparison(dataset, configs, dataset_name_override)
        )

    # ------------------------------------------------------------------
    # Single config execution
    # ------------------------------------------------------------------

    async def _run_single_config(
        self,
        dataset: EvalDataset,
        config: ModelRunConfig,
    ) -> ModelRunResult:
        """
        Run RAG + evaluation for one ModelRunConfig on a fresh dataset copy.

        Never raises — all failures are captured in ModelRunResult so
        one bad model config does not abort the entire comparison.

        RAGPipeline.answer_dataset() is synchronous (wraps blocking
        embedding + ChromaDB + Gemini calls). It is wrapped in
        run_in_executor so it does not block the asyncio event loop
        while other model configs run concurrently.
        """
        wall_start = time.monotonic()

        try:
            dataset_copy = self._fresh_dataset_copy(dataset)

            logger.info(
                f"ComparisonRunner: Running RAG for "
                f"'{config.display_label}' "
                f"({len(dataset_copy.pairs)} pairs)..."
            )

            loop = asyncio.get_event_loop()
            batch_result = await loop.run_in_executor(
                None,
                lambda: self._rag_pipeline.answer_dataset(
                    dataset_copy, config
                ),
            )

            if batch_result.answered_pairs == 0:
                return ModelRunResult(
                    config=config,
                    report=None,
                    succeeded=False,
                    error_message=(
                        f"RAG pipeline answered "
                        f"0/{batch_result.total_pairs} pairs. "
                        f"Check collection_name and model availability."
                    ),
                    wall_latency_ms=(
                        time.monotonic() - wall_start
                    ) * 1000,
                )

            logger.info(
                f"ComparisonRunner: Running evaluation for "
                f"'{config.display_label}'..."
            )

            report = await self._evaluation_engine.arun(
                dataset=dataset_copy,
                rag_model=config.model_id,
                collection_name=config.collection_name,
                top_k=config.top_k,
                temperature=config.temperature,
            )

            wall_latency_ms = (time.monotonic() - wall_start) * 1000

            return ModelRunResult(
                config=config,
                report=report,
                succeeded=True,
                wall_latency_ms=wall_latency_ms,
            )

        except Exception as exc:
            wall_latency_ms = (time.monotonic() - wall_start) * 1000
            logger.error(
                f"ComparisonRunner: Config '{config.display_label}' "
                f"failed: {exc}"
            )
            return ModelRunResult(
                config=config,
                report=None,
                succeeded=False,
                error_message=f"{type(exc).__name__}: {exc}",
                wall_latency_ms=wall_latency_ms,
            )

    # ------------------------------------------------------------------
    # Dataset isolation
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_dataset_copy(dataset: EvalDataset) -> EvalDataset:
        """
        Deep copy the dataset with all pairs reset to PENDING.

        Critical for correctness: QAPair.set_answer() can only be
        called once per pair (PENDING → ANSWERED). Running the same
        dataset through N models requires N independent copies, each
        starting from PENDING, so RAG answers and eval scores never
        collide across models.

        Deep copy preserves question, ground_truth_answer, source_chunk,
        and all identity fields exactly — only the per-run mutable
        fields (generated_answer, scores, status) are reset.
        """
        fresh = copy.deepcopy(dataset)
        for pair in fresh.pairs:
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
        return fresh

    # ------------------------------------------------------------------
    # Deduplication and safety caps
    # ------------------------------------------------------------------

    def _deduplicate_configs(
        self,
        configs: list[ModelRunConfig],
    ) -> list[ModelRunConfig]:
        """
        Remove duplicate configs by config_hash.

        Two configs with identical (model_id, collection_name, top_k,
        temperature, score_threshold) are considered duplicates —
        running both would waste API calls on identical work.
        First occurrence wins; later duplicates are dropped with a
        warning naming both configs for traceability.
        """
        seen: dict[str, ModelRunConfig] = {}
        for config in configs:
            h = config.config_hash
            if h in seen:
                logger.warning(
                    f"ComparisonRunner: Duplicate config skipped: "
                    f"'{config.display_label}' "
                    f"(identical to '{seen[h].display_label}', "
                    f"hash={h})."
                )
                continue
            seen[h] = config
        return list(seen.values())

    def _enforce_max_total_runs(
        self,
        configs: list[ModelRunConfig],
    ) -> None:
        """
        Raise if the number of configs exceeds the safety cap from
        models.yaml comparison_grid.max_total_runs.

        Prevents accidentally queuing 50+ runs against a free-tier
        API in a single comparison job. Falls back to a conservative
        cap of 12 if the model registry is unavailable for any reason.
        """
        try:
            from config import get_model_registry
            registry = get_model_registry()
            max_runs = registry.comparison_grid.max_total_runs
        except Exception:
            max_runs = 12

        if len(configs) > max_runs:
            raise ComparisonRunnerError(
                reason=(
                    f"Comparison requests {len(configs)} model runs but "
                    f"max_total_runs={max_runs} "
                    f"(configured in models.yaml comparison_grid). "
                    f"Reduce the number of models/parameters being "
                    f"compared, or increase max_total_runs if you "
                    f"understand the API quota implications."
                ),
            )

    # ------------------------------------------------------------------
    # Grid builder — convenience factory for the UI
    # ------------------------------------------------------------------

    @staticmethod
    def build_grid_configs(
        model_ids: list[str],
        collection_name: str,
        top_k_values: list[int] | None = None,
        temperatures: list[float] | None = None,
        score_threshold: float = 0.0,
    ) -> list[ModelRunConfig]:
        """
        Build a full parameter grid of ModelRunConfig.

        Convenience factory for the Streamlit UI's "Run Comparison"
        button. Generates one config per (model × top_k × temperature)
        combination.

        Args:
            model_ids:       Models to include in the grid.
            collection_name: ChromaDB collection used by all configs.
            top_k_values:    Defaults to [5] if not provided.
            temperatures:    Defaults to [0.0] if not provided.
            score_threshold: Applied uniformly across the grid.

        Returns:
            List of ModelRunConfig — one per combination. Pass directly
            to arun_comparison(), which deduplicates and enforces
            max_total_runs automatically.
        """
        resolved_top_k = top_k_values or [5]
        resolved_temps = temperatures or [0.0]

        configs: list[ModelRunConfig] = []
        for model_id in model_ids:
            for top_k in resolved_top_k:
                for temperature in resolved_temps:
                    configs.append(
                        ModelRunConfig(
                            model_id=model_id,
                            collection_name=collection_name,
                            top_k=top_k,
                            temperature=temperature,
                            score_threshold=score_threshold,
                            run_label=(
                                f"{model_id} (k={top_k}, t={temperature})"
                            ),
                        )
                    )
        return configs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_concurrent_runs(self) -> int:
        return self._settings.comparison.max_concurrent_runs

    def __repr__(self) -> str:
        return (
            f"ComparisonRunner("
            f"max_concurrent_runs={self.max_concurrent_runs})"
        )

# ===========================================================================
# CUSTOM EXCEPTION
# ===========================================================================

class ComparisonRunnerError(Exception):
    """
    Raised for comparison-level failures that prevent any matrix
    from being built — either the requested grid exceeds the safety
    cap, or every model config failed.

    Distinct from per-model failures (captured in ModelRunResult
    without raising) — this exception means the comparison as a
    whole cannot proceed and the caller (UI/API) must surface it
    directly rather than rendering a partial dashboard.
    """

    def __init__(
        self,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(f"[ComparisonRunner] {reason}")

__all__ = [
    "ModelRunResult",
    "ComparisonRunner",
    "ComparisonRunnerError",
]