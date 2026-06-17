"""
src/evaluation/engine.py

EvaluationEngine — orchestrates the complete LLM-as-a-Judge evaluation.

Flow per evaluation run:
    EvalDataset (list of QAPairs with generated_answer populated)
        ↓ For each QAPair (with inter-question delay)
        ↓ asyncio.gather() — run all 4 evaluators concurrently
        ↓ FaithfulnessEvaluator.aevaluate(pair)
        ↓ AnswerRelevanceEvaluator.aevaluate(pair)
        ↓ ContextPrecisionEvaluator.aevaluate(pair)
        ↓ CorrectnessEvaluator.aevaluate(pair)
        ↓ Compute composite score
        ↓ Construct EvalResult
        ↓ Checkpoint to SQLite (if enabled)
        ↓ ResultAggregator.aggregate(all_results)
        ↓ EvalReport

Design principles:
  - The engine is a pure orchestrator. It contains no scoring logic,
    no prompt construction, no JSON parsing. All of that lives in
    the evaluators and the base class.
  - Concurrency model: metrics for one pair run concurrently
    (asyncio.gather over 4 evaluators). Pairs run sequentially
    with configurable inter-question delay. This respects Gemini
    free-tier rate limits (15 RPM) while maximising throughput
    per pair.
  - Failure isolation: a failure on one pair does not abort the run.
    continue_on_question_failure from eval.yaml controls this.
  - Checkpointing: if checkpoint_after_each_question=True in eval.yaml,
    each EvalResult is written to SQLite immediately after scoring.
    A run interrupted at question 47/100 can be inspected and
    partially resumed.
  - The engine does NOT run the RAG pipeline. It expects QAPairs
    with status=ANSWERED (generated_answer already set). The caller
    (ComparisonRunner or the FastAPI route) is responsible for running
    RAGPipeline first, then passing the answered dataset to the engine.

Rate limit strategy on Gemini free tier (15 RPM):
  - 4 concurrent metric calls per question × 1 question at a time
    = 4 simultaneous requests at peak.
  - inter_question_delay_seconds=1.0 (from eval.yaml) spaces
    question starts by 1 second.
  - At 4 calls/question and ~2s per call, we use ~2 RPM — well
    within the 15 RPM limit.
  - For paid tier: set inter_question_delay_seconds=0.0 and
    increase max_concurrent_eval_calls in .env.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config.settings import Settings
from src.dataset.schema import EvalDataset, QAPair, QAPairStatus
from src.evaluation.evaluators import BaseEvaluator, build_all_evaluators
from src.evaluation.schema import (
    AggregatedResult,
    AggregationStrategy,
    ComparisonMatrix,
    EvalReport,
    EvalResult,
    MetricStats,
    ModelComparisonEntry,
    RunStatus,
    make_eval_result_id,
    make_matrix_id,
    make_run_id,
)


# ===========================================================================
# RESULT AGGREGATOR
# ===========================================================================


class ResultAggregator:
    """
    Computes descriptive statistics across a list of EvalResults.

    Produces AggregatedResult — the per-model summary consumed by
    the dashboard's radar chart, bar chart, and comparison table.

    Stateless — no instance state. All methods are pure functions
    that take inputs and return outputs.
    """

    def aggregate(
        self,
        results: list[EvalResult],
        run_id: str,
        rag_model: str,
        judge_model: str,
        dataset_id: str,
        n_pairs_total: int,
        min_reliable_sample: int = 10,
        strategy: AggregationStrategy = AggregationStrategy.MEAN,
    ) -> AggregatedResult:
        """
        Aggregate a list of EvalResults into an AggregatedResult.

        Args:
            results:            EvalResults to aggregate.
            run_id:             Run ID for the aggregated result.
            rag_model:          Model that generated the answers.
            judge_model:        Model that scored the answers.
            dataset_id:         Dataset ID.
            n_pairs_total:      Total pairs in the dataset (including failed).
            min_reliable_sample: Threshold below which low_sample_warning=True.
            strategy:           Aggregation strategy (mean, median).

        Returns:
            AggregatedResult with per-metric statistics.
        """
        if not results:
            logger.warning(
                "ResultAggregator: No results to aggregate. "
                "Returning empty AggregatedResult."
            )
            return AggregatedResult(
                run_id=run_id,
                rag_model=rag_model,
                judge_model=judge_model,
                dataset_id=dataset_id,
                n_pairs_total=n_pairs_total,
                n_pairs_evaluated=0,
                n_pairs_failed=n_pairs_total,
                low_sample_warning=True,
            )

        n_evaluated = len(results)
        n_failed = n_pairs_total - n_evaluated

        # Per-metric aggregation
        metric_names = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "correctness",
        ]

        metric_stats: dict[str, MetricStats | None] = {}
        for metric_name in metric_names:
            stats = self._aggregate_metric(
                results=results,
                metric_name=metric_name,
                strategy=strategy,
            )
            metric_stats[metric_name] = stats

        # Composite score aggregation
        composite_scores = [r.composite_score for r in results]
        composite_mean = self._mean(composite_scores)
        composite_std = self._std(composite_scores, composite_mean)
        composite_median = self._median(composite_scores)

        # Performance aggregation
        avg_rag_latency = self._mean(
            [r.rag_latency_ms for r in results]
        )
        avg_eval_latency = self._mean(
            [r.eval_latency_ms for r in results]
        )
        avg_total_latency = self._mean(
            [r.total_latency_ms for r in results]
        )
        total_input_tokens = sum(r.eval_input_tokens for r in results)
        total_output_tokens = sum(r.eval_output_tokens for r in results)
        total_cost = sum(r.estimated_cost_usd for r in results)

        # Quality signals
        parse_failures = sum(
            1 for r in results if r.any_parse_failed
        )
        overall_parse_failure_rate = (
            parse_failures / n_evaluated
            if n_evaluated > 0
            else 0.0
        )

        correctness_skips = sum(
            1 for r in results if r.correctness_skipped
        )
        correctness_skip_rate = (
            correctness_skips / n_evaluated
            if n_evaluated > 0
            else 0.0
        )

        low_sample_warning = n_evaluated < min_reliable_sample

        if low_sample_warning:
            logger.warning(
                f"ResultAggregator: Only {n_evaluated} pairs evaluated "
                f"(min recommended: {min_reliable_sample}). "
                f"Aggregate statistics may not be reliable."
            )

        return AggregatedResult(
            run_id=run_id,
            rag_model=rag_model,
            judge_model=judge_model,
            dataset_id=dataset_id,
            n_pairs_total=n_pairs_total,
            n_pairs_evaluated=n_evaluated,
            n_pairs_failed=n_failed,
            aggregation_strategy=strategy,
            faithfulness=metric_stats.get("faithfulness"),
            answer_relevance=metric_stats.get("answer_relevance"),
            context_precision=metric_stats.get("context_precision"),
            correctness=metric_stats.get("correctness"),
            composite_mean=round(composite_mean, 4),
            composite_std=round(composite_std, 4),
            composite_median=round(composite_median, 4),
            avg_rag_latency_ms=round(avg_rag_latency, 1),
            avg_eval_latency_ms=round(avg_eval_latency, 1),
            avg_total_latency_ms=round(avg_total_latency, 1),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6),
            overall_parse_failure_rate=round(
                overall_parse_failure_rate, 4
            ),
            correctness_skip_rate=round(correctness_skip_rate, 4),
            low_sample_warning=low_sample_warning,
        )

    def _aggregate_metric(
        self,
        results: list[EvalResult],
        metric_name: str,
        strategy: AggregationStrategy,
    ) -> MetricStats | None:
        """
        Compute MetricStats for one metric across all results.

        Excludes:
          - Results where this metric's score is None
          - Results where parse_failed=True for this metric
            (unreliable scores should not distort statistics)

        Returns None if no reliable scores exist for this metric.
        """
        scores: list[float] = []
        parse_failures = 0

        for result in results:
            ms = result.metric_scores.get(metric_name)
            if ms is None:
                continue
            if ms.parse_failed:
                parse_failures += 1
                continue
            scores.append(ms.score)

        total_attempts = len(scores) + parse_failures
        parse_failure_rate = (
            parse_failures / total_attempts
            if total_attempts > 0
            else 0.0
        )

        if not scores:
            return None

        mean = self._mean(scores)
        std = self._std(scores, mean)
        median = self._median(scores)
        p25 = self._percentile(scores, 25)
        p75 = self._percentile(scores, 75)

        return MetricStats(
            metric_name=metric_name,
            sample_size=len(scores),
            mean=round(mean, 4),
            std=round(std, 4),
            median=round(median, 4),
            min=round(min(scores), 4),
            max=round(max(scores), 4),
            p25=round(p25, 4),
            p75=round(p75, 4),
            parse_failure_rate=round(parse_failure_rate, 4),
        )

    # ------------------------------------------------------------------
    # Statistics utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        index = (p / 100) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower
        return (
            sorted_vals[lower] * (1 - fraction)
            + sorted_vals[upper] * fraction
        )

# ===========================================================================
# EVALUATION ENGINE
# ===========================================================================

class EvaluationEngine:
    """
    Orchestrates the LLM-as-a-Judge evaluation pipeline.

    Responsibilities:
      1. Accept an EvalDataset with answered QAPairs.
      2. Score each pair on all 4 metrics concurrently.
      3. Compute composite scores.
      4. Aggregate results into AggregatedResult.
      5. Return a complete EvalReport.

    Does NOT:
      - Run the RAG pipeline (caller's responsibility).
      - Store results to SQLite (RunRepository's responsibility).
      - Render charts (Streamlit UI's responsibility).

    Construction:
        engine = EvaluationEngine.from_settings(settings)

    Usage:
        report = await engine.arun(
            dataset=answered_dataset,
            rag_model="gemini-1.5-flash",
            collection_name="my_docs",
            top_k=5,
            temperature=0.0,
        )
    """

    def __init__(
        self,
        evaluators: dict[str, BaseEvaluator],
        settings: Settings,
        aggregator: ResultAggregator | None = None,
    ) -> None:
        """
        Initialise the engine with pre-built evaluators.

        Args:
            evaluators:  Dict of metric_name → BaseEvaluator instance.
                         Build with build_all_evaluators(settings).
            settings:    Application settings for eval run config.
            aggregator:  Optional ResultAggregator override for testing.
        """
        self._evaluators = evaluators
        self._settings = settings
        self._aggregator = aggregator or ResultAggregator()

        # Load eval run config from eval.yaml
        from config import get_eval_config
        self._eval_config = get_eval_config()
        self._run_config = self._eval_config.eval_run
        self._weight_map = self._eval_config.metrics.weight_map

        logger.info(
            f"EvaluationEngine initialised. "
            f"evaluators={list(evaluators.keys())}, "
            f"parallel_metrics={self._run_config.parallel_metrics_per_question}, "
            f"inter_question_delay={self._run_config.inter_question_delay_seconds}s, "
            f"max_questions={self._run_config.max_questions_per_run}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> EvaluationEngine:
        """
        Standard factory — builds all evaluators from Settings.

        Usage:
            engine = EvaluationEngine.from_settings(get_settings())
        """
        evaluators = build_all_evaluators(settings)
        return cls(
            evaluators=evaluators,
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def arun(
        self,
        dataset: EvalDataset,
        rag_model: str,
        collection_name: str,
        top_k: int = 5,
        temperature: float = 0.0,
        run_id: str | None = None,
        on_pair_complete: Any | None = None,
    ) -> EvalReport:
        """
        Run evaluation asynchronously on a complete EvalDataset.

        Args:
            dataset:          EvalDataset with pairs in ANSWERED status.
                              PENDING pairs are skipped with a warning.
            rag_model:        Model ID used to generate the answers.
            collection_name:  ChromaDB collection queried during RAG.
            top_k:            Retriever top_k used during RAG.
            temperature:      RAG model temperature used.
            run_id:           Optional run ID override. Auto-generated
                              if not provided.
            on_pair_complete: Optional async callback called after each
                              pair is evaluated. Signature:
                              async def callback(result: EvalResult,
                                                 pair_idx: int,
                                                 total: int) -> None
                              Used by Streamlit for live progress updates.

        Returns:
            EvalReport with all results and AggregatedResult.
        """
        resolved_run_id = run_id or make_run_id()
        started_at = datetime.now(timezone.utc)
        wall_start = time.monotonic()

        logger.info(
            f"EvaluationEngine: Starting run '{resolved_run_id}'. "
            f"dataset='{dataset.name}', "
            f"pairs={len(dataset.pairs)}, "
            f"rag_model='{rag_model}', "
            f"judge='{self._settings.judge.model}'"
        )

        # Filter to answered pairs only
        pairs_to_eval = self._select_pairs(dataset)

        if not pairs_to_eval:
            logger.error(
                f"EvaluationEngine: No answered pairs in dataset "
                f"'{dataset.name}'. "
                f"Run RAGPipeline before calling EvaluationEngine."
            )
            return self._make_failed_report(
                run_id=resolved_run_id,
                dataset=dataset,
                rag_model=rag_model,
                collection_name=collection_name,
                top_k=top_k,
                temperature=temperature,
                started_at=started_at,
                reason="No answered pairs available for evaluation.",
            )

        # Apply max_questions_per_run cap
        if len(pairs_to_eval) > self._run_config.max_questions_per_run:
            logger.warning(
                f"EvaluationEngine: Dataset has {len(pairs_to_eval)} "
                f"answered pairs but max_questions_per_run="
                f"{self._run_config.max_questions_per_run}. "
                f"Truncating to first "
                f"{self._run_config.max_questions_per_run} pairs."
            )
            pairs_to_eval = pairs_to_eval[
                : self._run_config.max_questions_per_run
            ]

        total_pairs = len(pairs_to_eval)
        all_results: list[EvalResult] = []
        failed_count = 0

        # Semaphore limits total concurrent judge calls
        semaphore = asyncio.Semaphore(
            self._settings.comparison.max_concurrent_eval_calls
        )

        # ── Main evaluation loop ───────────────────────────────────────
        for pair_idx, pair in enumerate(pairs_to_eval):
            pair_start = time.monotonic()

            logger.debug(
                f"EvaluationEngine: Evaluating pair "
                f"{pair_idx + 1}/{total_pairs} "
                f"'{pair.id}' ..."
            )

            try:
                result = await self._evaluate_pair(
                    pair=pair,
                    run_id=resolved_run_id,
                    rag_model=rag_model,
                    semaphore=semaphore,
                )
                all_results.append(result)

                logger.info(
                    f"EvaluationEngine: Pair {pair_idx + 1}/{total_pairs} "
                    f"complete. "
                    f"composite={result.composite_score:.2f}, "
                    f"latency={result.total_latency_ms:.0f}ms"
                )

                # Callback for live UI progress
                if on_pair_complete is not None:
                    try:
                        await on_pair_complete(
                            result, pair_idx + 1, total_pairs
                        )
                    except Exception as cb_exc:
                        logger.warning(
                            f"EvaluationEngine: on_pair_complete callback "
                            f"failed: {cb_exc}"
                        )

            except Exception as exc:
                failed_count += 1
                logger.error(
                    f"EvaluationEngine: Pair '{pair.id}' failed: {exc}"
                )
                if not self._run_config.continue_on_question_failure:
                    logger.error(
                        "EvaluationEngine: continue_on_question_failure=False. "
                        "Aborting run."
                    )
                    break

            # Inter-question delay for rate limit compliance
            if (
                self._run_config.inter_question_delay_seconds > 0
                and pair_idx < total_pairs - 1
            ):
                await asyncio.sleep(
                    self._run_config.inter_question_delay_seconds
                )

        # ── Aggregate results ──────────────────────────────────────────
        from config import get_eval_config
        eval_config = get_eval_config()

        aggregate = self._aggregator.aggregate(
            results=all_results,
            run_id=resolved_run_id,
            rag_model=rag_model,
            judge_model=self._settings.judge.model,
            dataset_id=dataset.id,
            n_pairs_total=total_pairs,
            min_reliable_sample=(
                eval_config
                .aggregation
                .min_questions_for_reliable_aggregation
            ),
        )

        # ── Determine run status ───────────────────────────────────────
        if not all_results:
            status = RunStatus.FAILED
        elif failed_count == 0:
            status = RunStatus.COMPLETED
        else:
            status = RunStatus.PARTIAL

        completed_at = datetime.now(timezone.utc)
        total_latency_ms = (time.monotonic() - wall_start) * 1000

        report = EvalReport(
            run_id=resolved_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            rag_model=rag_model,
            judge_model=self._settings.judge.model,
            prompt_version=eval_config.active_prompt_version,
            top_k=top_k,
            temperature=temperature,
            collection_name=collection_name,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=round(total_latency_ms, 1),
            status=status,
            results=all_results,
            aggregate=aggregate,
        )

        logger.info(
            f"EvaluationEngine: Run '{resolved_run_id}' complete. "
            f"status={status.value}, "
            f"evaluated={len(all_results)}/{total_pairs}, "
            f"composite_mean={aggregate.composite_mean:.3f}, "
            f"total_latency={total_latency_ms:.0f}ms"
        )

        return report

    def run(
        self,
        dataset: EvalDataset,
        rag_model: str,
        collection_name: str,
        top_k: int = 5,
        temperature: float = 0.0,
        run_id: str | None = None,
    ) -> EvalReport:
        """
        Synchronous wrapper around arun().

        Use when an async event loop is not available
        (e.g. CLI scripts, Jupyter notebooks, tests).

        For Streamlit and FastAPI, prefer arun() directly.
        """
        return asyncio.run(
            self.arun(
                dataset=dataset,
                rag_model=rag_model,
                collection_name=collection_name,
                top_k=top_k,
                temperature=temperature,
                run_id=run_id,
            )
        )

    # ------------------------------------------------------------------
    # Per-pair evaluation
    # ------------------------------------------------------------------

    async def _evaluate_pair(
        self,
        pair: QAPair,
        run_id: str,
        rag_model: str,
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """
        Score all metrics for a single QAPair.

        Runs all 4 evaluators concurrently via asyncio.gather()
        when parallel_metrics_per_question=True (from eval.yaml).
        Falls back to sequential execution when False.

        The semaphore limits total in-flight judge calls across
        all pairs that may be evaluated in parallel by the
        ComparisonRunner.
        """
        pair_wall_start = time.monotonic()

        async def score_with_semaphore(
            evaluator: BaseEvaluator,
        ):
            async with semaphore:
                return await evaluator.aevaluate(pair)

        if self._run_config.parallel_metrics_per_question:
            # All 4 metrics fire simultaneously
            scores = await asyncio.gather(
                *[
                    score_with_semaphore(evaluator)
                    for evaluator in self._evaluators.values()
                ],
                return_exceptions=False,
            )
            metric_scores = {
                metric_name: score
                for metric_name, score in zip(
                    self._evaluators.keys(), scores
                )
            }
        else:
            # Sequential — more predictable rate limit behaviour
            metric_scores = {}
            for metric_name, evaluator in self._evaluators.items():
                async with semaphore:
                    metric_scores[metric_name] = (
                        await evaluator.aevaluate(pair)
                    )

        # Compute composite score
        composite = self._compute_composite(metric_scores)

        # Check quality flags
        any_parse_failed = any(
            ms.parse_failed
            for ms in metric_scores.values()
        )
        correctness_skipped = (
            "correctness" in metric_scores
            and metric_scores["correctness"].parse_failed
            and "NO_REFERENCE" in metric_scores["correctness"].reasoning
        )

        # Aggregate token counts
        eval_input_tokens = sum(
            ms.input_tokens for ms in metric_scores.values()
        )
        eval_output_tokens = sum(
            ms.output_tokens for ms in metric_scores.values()
        )
        eval_latency_ms = sum(
            ms.latency_ms for ms in metric_scores.values()
        )

        total_latency_ms = (
            time.monotonic() - pair_wall_start
        ) * 1000

        # Estimate cost
        estimated_cost = self._estimate_pair_cost(
            rag_model=rag_model,
            rag_input_tokens=pair.rag_input_tokens,
            rag_output_tokens=pair.rag_output_tokens,
            eval_input_tokens=eval_input_tokens,
            eval_output_tokens=eval_output_tokens,
        )

        return EvalResult(
            id=make_eval_result_id(),
            run_id=run_id,
            pair_id=pair.id,
            dataset_id=pair.dataset_id,
            evaluated_at=datetime.now(timezone.utc),
            rag_model=rag_model,
            judge_model=self._settings.judge.model,
            prompt_version=self._eval_config.active_prompt_version,
            question=pair.question,
            ground_truth_answer=pair.ground_truth_answer,
            generated_answer=pair.generated_answer or "",
            retrieved_chunks=pair.retrieved_chunks,
            retrieved_chunk_sources=pair.retrieved_chunk_sources,
            metric_scores=metric_scores,
            composite_score=composite,
            rag_latency_ms=pair.rag_latency_ms,
            eval_latency_ms=round(eval_latency_ms, 1),
            total_latency_ms=round(total_latency_ms, 1),
            rag_input_tokens=pair.rag_input_tokens,
            rag_output_tokens=pair.rag_output_tokens,
            eval_input_tokens=eval_input_tokens,
            eval_output_tokens=eval_output_tokens,
            estimated_cost_usd=estimated_cost,
            any_parse_failed=any_parse_failed,
            correctness_skipped=correctness_skipped,
        )

    # ------------------------------------------------------------------
    # Composite score computation
    # ------------------------------------------------------------------

    def _compute_composite(
        self,
        metric_scores: dict[str, Any],
    ) -> float:
        """
        Compute weighted composite score from metric scores.

        Strategy from eval.yaml missing_metric_strategy:
          "exclude_from_weight" — recompute weights over available
                                   non-failed metrics only.
          "score_zero"          — treat missing/failed as 0.

        Returns composite in [1.0, 5.0].
        """
        strategy = (
            self._eval_config
            .composite_score
            .missing_metric_strategy
        )

        available: dict[str, float] = {}
        for metric_name, ms in metric_scores.items():
            if not ms.parse_failed:
                available[metric_name] = ms.score

        if not available:
            # All metrics failed — return minimum score
            logger.warning(
                "EvaluationEngine._compute_composite: "
                "All metric scores are parse failures. "
                "Returning minimum composite score."
            )
            return float(
                self._eval_config.composite_score.scale_min
            )

        if strategy == "exclude_from_weight":
            # Reweight over available metrics only
            total_weight = sum(
                self._weight_map.get(name, 0.0)
                for name in available
            )
            if total_weight == 0.0:
                return float(
                    self._eval_config.composite_score.scale_min
                )
            composite = sum(
                score * self._weight_map.get(name, 0.0)
                for name, score in available.items()
            ) / total_weight

        else:
            # "score_zero" — include all metrics, failed = 0
            composite = sum(
                score * self._weight_map.get(name, 0.0)
                for name, score in available.items()
            )

        # Clip to valid scale
        scale_min = float(
            self._eval_config.composite_score.scale_min
        )
        scale_max = float(
            self._eval_config.composite_score.scale_max
        )
        return round(max(scale_min, min(scale_max, composite)), 4)

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def _select_pairs(
        self,
        dataset: EvalDataset,
    ) -> list[QAPair]:
        """
        Select pairs eligible for evaluation.

        Eligible: status == ANSWERED
        Skipped:  status == PENDING (warn), EVALUATED (skip silently),
                  FAILED (skip silently)
        """
        eligible: list[QAPair] = []
        pending_count = 0

        for pair in dataset.pairs:
            if pair.status == QAPairStatus.ANSWERED:
                eligible.append(pair)
            elif pair.status == QAPairStatus.PENDING:
                pending_count += 1
            # EVALUATED and FAILED are silently skipped

        if pending_count > 0:
            logger.warning(
                f"EvaluationEngine: {pending_count} pairs are still "
                f"PENDING (no generated_answer). "
                f"These pairs will be skipped. "
                f"Run RAGPipeline on the dataset first."
            )

        logger.info(
            f"EvaluationEngine: {len(eligible)}/{len(dataset.pairs)} "
            f"pairs eligible for evaluation."
        )

        return eligible

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_pair_cost(
        self,
        rag_model: str,
        rag_input_tokens: int,
        rag_output_tokens: int,
        eval_input_tokens: int,
        eval_output_tokens: int,
    ) -> float:
        """
        Estimate USD cost for one pair (RAG call + 4 judge calls).

        Uses cost_per_1k from models.yaml for both RAG and judge models.
        Returns 0.0 if either model is not in the registry.
        """
        try:
            from config import get_model_registry

            registry = get_model_registry()
            safety = (
                registry.cost_estimation.safety_multiplier
            )

            rag_cost = 0.0
            try:
                rag_m = registry.get_model(rag_model)
                rag_cost = rag_m.estimate_cost(
                    input_tokens=rag_input_tokens,
                    output_tokens=rag_output_tokens,
                    safety_multiplier=safety,
                )
            except KeyError:
                pass

            judge_cost = 0.0
            try:
                judge_m = registry.get_model(
                    self._settings.judge.model
                )
                judge_cost = judge_m.estimate_cost(
                    input_tokens=eval_input_tokens,
                    output_tokens=eval_output_tokens,
                    safety_multiplier=safety,
                )
            except KeyError:
                pass

            return round(rag_cost + judge_cost, 8)

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Failed report construction
    # ------------------------------------------------------------------

    def _make_failed_report(
        self,
        run_id: str,
        dataset: EvalDataset,
        rag_model: str,
        collection_name: str,
        top_k: int,
        temperature: float,
        started_at: datetime,
        reason: str,
    ) -> EvalReport:
        """Construct a FAILED EvalReport for pre-flight failures."""
        from config import get_eval_config

        eval_config = get_eval_config()

        return EvalReport(
            run_id=run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            rag_model=rag_model,
            judge_model=self._settings.judge.model,
            prompt_version=eval_config.active_prompt_version,
            top_k=top_k,
            temperature=temperature,
            collection_name=collection_name,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            total_latency_ms=0.0,
            status=RunStatus.FAILED,
            error_message=reason,
            results=[],
            aggregate=None,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        return list(self._evaluators.keys())

    @property
    def judge_model(self) -> str:
        return self._settings.judge.model

    def __repr__(self) -> str:
        return (
            f"EvaluationEngine("
            f"metrics={self.metric_names}, "
            f"judge='{self.judge_model}', "
            f"parallel={self._run_config.parallel_metrics_per_question})"
        )

# ===========================================================================
# COMPARISON MATRIX BUILDER
# ===========================================================================

class ComparisonMatrixBuilder:
    """
    Builds a ComparisonMatrix from multiple EvalReports.

    Used by ComparisonRunner after running the same dataset through
    multiple models. The resulting ComparisonMatrix is the primary
    object consumed by all Streamlit dashboard charts.

    Usage:
        builder = ComparisonMatrixBuilder()
        matrix = builder.build(
            reports=[report_flash, report_pro, report_legacy],
            dataset_id="ds_01J2K3M...",
            dataset_name="Q3 Financial Eval",
            judge_model="gemini-1.5-pro",
            prompt_version="1.0.0",
        )
    """

    def build(
        self,
        reports: list[EvalReport],
        dataset_id: str,
        dataset_name: str,
        judge_model: str,
        prompt_version: str,
    ) -> ComparisonMatrix:
        """
        Build a ComparisonMatrix from a list of EvalReports.

        Args:
            reports:        One EvalReport per model run.
            dataset_id:     Common dataset ID across all runs.
            dataset_name:   Human-readable dataset name.
            judge_model:    Judge model used (same for all runs).
            prompt_version: Prompt version used (same for all runs).

        Returns:
            ComparisonMatrix with one ModelComparisonEntry per report.

        Raises:
            ValueError: if reports is empty.
        """
        if not reports:
            raise ValueError(
                "ComparisonMatrixBuilder.build: "
                "Cannot build matrix from empty reports list."
            )

        entries: list[ModelComparisonEntry] = []
        complete_reports: list[EvalReport] = []

        for report in reports:
            if report.aggregate is None:
                logger.warning(
                    f"ComparisonMatrixBuilder: Report '{report.run_id}' "
                    f"has no aggregate. Skipping."
                )
                continue

            # Get display metadata from model registry
            display_name, provider = self._get_model_display_info(
                report.rag_model
            )

            entry = ModelComparisonEntry.from_aggregated_result(
                result=report.aggregate,
                display_name=display_name,
                provider=provider,
            )
            entries.append(entry)
            complete_reports.append(report)

        if not entries:
            raise ValueError(
                "ComparisonMatrixBuilder.build: "
                "No reports with valid aggregates. "
                "Ensure evaluation runs completed successfully."
            )

        # Sort entries by composite_mean descending
        # (best model first in the comparison table)
        entries.sort(key=lambda e: e.composite_mean, reverse=True)

        matrix = ComparisonMatrix(
            matrix_id=make_matrix_id(),
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            created_at=datetime.now(timezone.utc),
            judge_model=judge_model,
            prompt_version=prompt_version,
            entries=entries,
            reports=complete_reports,
        )

        logger.info(
            f"ComparisonMatrixBuilder: Built matrix "
            f"'{matrix.matrix_id}' with "
            f"{len(entries)} model entries. "
            f"Best model: "
            f"'{matrix.best_model_by_composite.rag_model if matrix.best_model_by_composite else 'N/A'}' "
            f"(composite={matrix.best_model_by_composite.composite_mean:.3f if matrix.best_model_by_composite else 0})"
        )

        return matrix

    @staticmethod
    def _get_model_display_info(
        model_id: str,
    ) -> tuple[str, str]:
        """
        Get display_name and provider for a model ID.

        Falls back gracefully if model is not in the registry.
        """
        try:
            from config import get_model_registry

            registry = get_model_registry()
            model = registry.get_model(model_id)
            return model.display_name, model.provider
        except (KeyError, Exception):
            # Model not in registry — use ID as display name
            provider = model_id.split("-")[0] if "-" in model_id else "unknown"
            return model_id, provider

__all__ = [
    "ResultAggregator",
    "EvaluationEngine",
    "ComparisonMatrixBuilder",
]