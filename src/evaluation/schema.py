"""
src/evaluation/schema.py

Pydantic schemas for the evaluation engine layer.

These schemas represent the output of the LLM-as-a-Judge pipeline:
  - MetricScore      — score + reasoning for one metric on one QAPair
                       (defined in src/dataset/schema.py, re-exported here
                       for convenience within the evaluation layer)
  - EvalResult       — all 4 metric scores for one QAPair in one run
  - EvalReport       — all EvalResults for one model run on one dataset
  - AggregatedResult — per-metric statistics collapsed across all pairs
  - ComparisonMatrix — EvalReports from multiple model runs, side-by-side

Data flow:
    EvaluationEngine.evaluate_pair()
        → EvalResult (one QAPair × one model)

    EvaluationEngine.run()
        → EvalReport (all pairs × one model)

    ResultAggregator.aggregate()
        → AggregatedResult (statistics per metric per model)

    ComparisonRunner.run()
        → ComparisonMatrix (all models × all metrics)

    Dashboard reads ComparisonMatrix
        → radar chart, bar chart, scatter, drilldown table

Design rules:
  - All schemas are frozen after construction.
  - EvalReport is the unit of persistence in SQLite (RunRepository).
  - ComparisonMatrix is the unit of visualisation in Streamlit.
  - No schema in this file imports from src/evaluation/engine.py or
    any evaluator — schemas are pure data, zero behaviour.
  - All float scores are in [1.0, 5.0] — the same scale as MetricScore.
    Aggregated statistics (mean, std) inherit this scale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Re-import MetricScore from dataset layer — it is the atomic score unit
# shared by both dataset and evaluation layers.
from src.dataset.schema import MetricScore, QAPair

# ===========================================================================
# ENUMS
# ===========================================================================

class RunStatus(str, Enum):
    """
    Lifecycle status of an evaluation run.

    Transitions:
        pending → running → completed
        pending → running → partial   (some pairs failed)
        pending → failed              (catastrophic failure before any eval)

    Terminal states: completed, partial, failed.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"       # Some pairs evaluated, some failed
    FAILED = "failed"         # Catastrophic failure — no pairs evaluated

class AggregationStrategy(str, Enum):
    """
    How metric scores are aggregated across pairs.

    MEAN    — arithmetic mean (default, most interpretable)
    MEDIAN  — robust to outliers
    WEIGHTED_MEAN — weighted by pair confidence (future extension)
    """

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"

# ===========================================================================
# EVAL RESULT — one QAPair × one model
# ===========================================================================

class EvalResult(BaseModel):
    """
    Evaluation result for a single QAPair in a single run.

    Contains:
      - The QAPair that was evaluated (with generated_answer populated)
      - All metric scores from the judge LLM
      - Composite score (weighted average)
      - Run-level metadata (run_id, model, prompt_version)
      - Performance metadata (total latency, total tokens, total cost)

    EvalResult is the atomic unit of the evaluation pipeline.
    One EvalResult is produced per QAPair per run.
    An EvalReport contains N EvalResults (one per pair).

    Frozen — results are immutable after the judge produces scores.
    Mutating a score after evaluation would compromise reproducibility.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    id: Annotated[
        str,
        Field(description="Unique EvalResult ID. Format: 'evr_{ulid}'."),
    ]

    run_id: Annotated[
        str,
        Field(description="ID of the EvalReport this result belongs to."),
    ]

    pair_id: Annotated[
        str,
        Field(description="ID of the QAPair that was evaluated."),
    ]

    dataset_id: Annotated[
        str,
        Field(description="ID of the EvalDataset containing this pair."),
    ]

    evaluated_at: Annotated[
        datetime,
        Field(description="UTC timestamp when evaluation completed."),
    ]

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    rag_model: Annotated[
        str,
        Field(description="Model ID used for RAG answer generation."),
    ]

    judge_model: Annotated[
        str,
        Field(description="Model ID used as the evaluation judge."),
    ]

    prompt_version: Annotated[
        str,
        Field(
            description=(
                "Version of judge prompts used. "
                "Stored for reproducibility — results from different "
                "prompt versions should not be directly compared."
            )
        ),
    ]

    # ------------------------------------------------------------------
    # Q&A content (snapshot)
    # ------------------------------------------------------------------

    question: str
    ground_truth_answer: str
    generated_answer: str

    retrieved_chunks: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Context chunks retrieved for this question.",
        ),
    ]

    retrieved_chunk_sources: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Source filenames for retrieved chunks.",
        ),
    ]

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    metric_scores: Annotated[
        dict[str, MetricScore],
        Field(
            description=(
                "Map of metric_name → MetricScore. "
                "Keys: faithfulness, answer_relevance, "
                "context_precision, correctness (if reference available)."
            )
        ),
    ]

    composite_score: Annotated[
        float,
        Field(
            ge=1.0,
            le=5.0,
            description=(
                "Weighted average across all available metrics. "
                "Range [1.0, 5.0]."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Performance metadata
    # ------------------------------------------------------------------

    rag_latency_ms: float = Field(default=0.0, ge=0.0)
    eval_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time for all 4 judge calls.",
    )
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    rag_input_tokens: int = Field(default=0, ge=0)
    rag_output_tokens: int = Field(default=0, ge=0)
    eval_input_tokens: int = Field(default=0, ge=0)
    eval_output_tokens: int = Field(default=0, ge=0)

    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated USD cost for this result (RAG + eval).",
    )

    # ------------------------------------------------------------------
    # Quality flags
    # ------------------------------------------------------------------

    any_parse_failed: bool = Field(
        default=False,
        description=(
            "True if any metric score had parse_failed=True. "
            "Results with parse failures are less reliable."
        ),
    )

    correctness_skipped: bool = Field(
        default=False,
        description=(
            "True if Correctness metric was skipped "
            "(no ground truth reference available)."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def sync_parse_failed_flag(self) -> EvalResult:
        """
        Sync any_parse_failed with individual metric scores.

        model_config is frozen so we cannot set this after init —
        it must be set correctly at construction time.
        This validator verifies it was set correctly.
        """
        has_failure = any(
            ms.parse_failed
            for ms in self.metric_scores.values()
        )
        if has_failure and not self.any_parse_failed:
            raise ValueError(
                "EvalResult: one or more MetricScores have "
                "parse_failed=True but any_parse_failed=False. "
                "Set any_parse_failed=True when constructing EvalResult."
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def faithfulness_score(self) -> float | None:
        ms = self.metric_scores.get("faithfulness")
        return ms.score if ms else None

    @property
    def answer_relevance_score(self) -> float | None:
        ms = self.metric_scores.get("answer_relevance")
        return ms.score if ms else None

    @property
    def context_precision_score(self) -> float | None:
        ms = self.metric_scores.get("context_precision")
        return ms.score if ms else None

    @property
    def correctness_score(self) -> float | None:
        ms = self.metric_scores.get("correctness")
        return ms.score if ms else None

    @property
    def total_tokens(self) -> int:
        return (
            self.rag_input_tokens
            + self.rag_output_tokens
            + self.eval_input_tokens
            + self.eval_output_tokens
        )

    @property
    def context_as_numbered_list(self) -> str:
        if not self.retrieved_chunks:
            return "No context retrieved."
        return "\n\n".join(
            f"{i + 1}. {chunk}"
            for i, chunk in enumerate(self.retrieved_chunks)
        )

    @property
    def is_reliable(self) -> bool:
        """
        True if no quality flags are raised.

        Results with parse failures or extremely low composite scores
        should be treated with lower confidence in aggregate statistics.
        """
        return not self.any_parse_failed

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Serialise to a flat dict for SQLite storage and CSV export.

        Metric scores are flattened:
          faithfulness_score, faithfulness_reasoning, etc.
        """
        row: dict[str, Any] = {
            "id":                      self.id,
            "run_id":                  self.run_id,
            "pair_id":                 self.pair_id,
            "dataset_id":              self.dataset_id,
            "evaluated_at":            self.evaluated_at.isoformat(),
            "rag_model":               self.rag_model,
            "judge_model":             self.judge_model,
            "prompt_version":          self.prompt_version,
            "question":                self.question,
            "ground_truth_answer":     self.ground_truth_answer,
            "generated_answer":        self.generated_answer,
            "retrieved_chunks_count":  len(self.retrieved_chunks),
            "composite_score":         self.composite_score,
            "rag_latency_ms":          round(self.rag_latency_ms, 1),
            "eval_latency_ms":         round(self.eval_latency_ms, 1),
            "total_latency_ms":        round(self.total_latency_ms, 1),
            "rag_input_tokens":        self.rag_input_tokens,
            "rag_output_tokens":       self.rag_output_tokens,
            "eval_input_tokens":       self.eval_input_tokens,
            "eval_output_tokens":      self.eval_output_tokens,
            "total_tokens":            self.total_tokens,
            "estimated_cost_usd":      self.estimated_cost_usd,
            "any_parse_failed":        self.any_parse_failed,
            "correctness_skipped":     self.correctness_skipped,
        }

        for metric_name in [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "correctness",
        ]:
            ms = self.metric_scores.get(metric_name)
            row[f"{metric_name}_score"] = (
                ms.score if ms else None
            )
            row[f"{metric_name}_reasoning"] = (
                ms.reasoning if ms else None
            )
            row[f"{metric_name}_latency_ms"] = (
                round(ms.latency_ms, 1) if ms else None
            )
            row[f"{metric_name}_parse_failed"] = (
                ms.parse_failed if ms else None
            )
            row[f"{metric_name}_low_confidence"] = (
                ms.low_confidence if ms else None
            )

        return row

    def __repr__(self) -> str:
        return (
            f"EvalResult("
            f"pair='{self.pair_id}', "
            f"composite={self.composite_score:.2f}, "
            f"rag_model='{self.rag_model}')"
        )

# ===========================================================================
# AGGREGATED RESULT — statistics per metric per model
# ===========================================================================

class MetricStats(BaseModel):
    """
    Descriptive statistics for one metric across all evaluated pairs.

    Produced by ResultAggregator. Used in ComparisonMatrix rows.
    All float fields are in [1.0, 5.0] except std which is in [0.0, 4.0].
    """

    model_config = ConfigDict(frozen=True)

    metric_name: str
    sample_size: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Number of pairs that contributed to these statistics. "
                "May be less than total pairs if some were skipped "
                "(e.g. correctness with no reference) or parse-failed."
            ),
        ),
    ]
    mean: float = Field(ge=0.0, le=5.0)
    std: float = Field(ge=0.0, le=5.0)
    median: float = Field(ge=0.0, le=5.0)
    min: float = Field(ge=0.0, le=5.0)
    max: float = Field(ge=0.0, le=5.0)
    p25: float = Field(ge=0.0, le=5.0)
    p75: float = Field(ge=0.0, le=5.0)
    parse_failure_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of scores where parse_failed=True.",
    )

    @property
    def is_reliable(self) -> bool:
        """True if parse failure rate is below 10%."""
        return self.parse_failure_rate < 0.10

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """
        Approximate 95% CI: mean ± 1.96 * (std / sqrt(n)).
        Assumes normal distribution — reasonable for n >= 30.
        """
        import math
        if self.sample_size < 2:
            return (self.mean, self.mean)
        margin = 1.96 * (self.std / math.sqrt(self.sample_size))
        return (
            max(1.0, round(self.mean - margin, 4)),
            min(5.0, round(self.mean + margin, 4)),
        )

    def __repr__(self) -> str:
        return (
            f"MetricStats("
            f"metric={self.metric_name}, "
            f"mean={self.mean:.2f}±{self.std:.2f}, "
            f"n={self.sample_size})"
        )

class AggregatedResult(BaseModel):
    """
    Aggregated statistics for one model run across all evaluated pairs.

    Contains MetricStats for each metric plus performance aggregates
    (latency, cost, token counts) and quality signal aggregates
    (parse failure rate, low confidence rate).

    Produced by ResultAggregator from a list of EvalResults.
    Stored on EvalReport and in ComparisonMatrix.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    rag_model: str
    judge_model: str
    dataset_id: str
    n_pairs_total: int = Field(ge=0)
    n_pairs_evaluated: int = Field(ge=0)
    n_pairs_failed: int = Field(ge=0)
    aggregation_strategy: AggregationStrategy = (
        AggregationStrategy.MEAN
    )

    # Per-metric statistics
    faithfulness: MetricStats | None = None
    answer_relevance: MetricStats | None = None
    context_precision: MetricStats | None = None
    correctness: MetricStats | None = None

    # Composite score statistics
    composite_mean: float = Field(ge=0.0, le=5.0, default=0.0)
    composite_std: float = Field(ge=0.0, le=5.0, default=0.0)
    composite_median: float = Field(ge=0.0, le=5.0, default=0.0)

    # Performance aggregates
    avg_rag_latency_ms: float = Field(ge=0.0, default=0.0)
    avg_eval_latency_ms: float = Field(ge=0.0, default=0.0)
    avg_total_latency_ms: float = Field(ge=0.0, default=0.0)
    total_input_tokens: int = Field(ge=0, default=0)
    total_output_tokens: int = Field(ge=0, default=0)
    total_cost_usd: float = Field(ge=0.0, default=0.0)

    # Quality signals
    overall_parse_failure_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    correctness_skip_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    low_sample_warning: bool = Field(
        default=False,
        description=(
            "True if n_pairs_evaluated < min_questions_for_reliable_aggregation "
            "from eval.yaml. Aggregate statistics may not be reliable."
        ),
    )

    @property
    def evaluation_rate(self) -> float:
        """Fraction of total pairs that were successfully evaluated."""
        if self.n_pairs_total == 0:
            return 0.0
        return self.n_pairs_evaluated / self.n_pairs_total

    @property
    def metric_stats_map(self) -> dict[str, MetricStats]:
        """
        Return all non-None MetricStats as a dict.

        Used by ComparisonMatrix and dashboard charts for uniform
        iteration without None-checking at every call site.
        """
        result: dict[str, MetricStats] = {}
        for name in [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "correctness",
        ]:
            stats = getattr(self, name, None)
            if stats is not None:
                result[name] = stats
        return result

    @property
    def radar_values(self) -> dict[str, float]:
        """
        Return mean scores for radar chart rendering.

        Keys match eval.yaml ui.radar_axes.
        Missing metrics return 0.0 (plotted at centre of radar).
        """
        return {
            "faithfulness":      (
                self.faithfulness.mean if self.faithfulness else 0.0
            ),
            "answer_relevance":  (
                self.answer_relevance.mean
                if self.answer_relevance else 0.0
            ),
            "context_precision": (
                self.context_precision.mean
                if self.context_precision else 0.0
            ),
            "correctness":       (
                self.correctness.mean if self.correctness else 0.0
            ),
        }

    def to_summary_row(self) -> dict[str, Any]:
        """
        Serialise to a flat dict for the comparison summary CSV.

        Matches summary_csv_columns from eval.yaml.
        """
        return {
            "run_id":                  self.run_id,
            "rag_model":               self.rag_model,
            "judge_model":             self.judge_model,
            "dataset_id":              self.dataset_id,
            "n_pairs_total":           self.n_pairs_total,
            "n_pairs_evaluated":       self.n_pairs_evaluated,
            "n_pairs_failed":          self.n_pairs_failed,
            "evaluation_rate":         round(self.evaluation_rate, 4),
            "faithfulness_mean":       (
                self.faithfulness.mean if self.faithfulness else None
            ),
            "faithfulness_std":        (
                self.faithfulness.std if self.faithfulness else None
            ),
            "answer_relevance_mean":   (
                self.answer_relevance.mean
                if self.answer_relevance else None
            ),
            "answer_relevance_std":    (
                self.answer_relevance.std
                if self.answer_relevance else None
            ),
            "context_precision_mean":  (
                self.context_precision.mean
                if self.context_precision else None
            ),
            "context_precision_std":   (
                self.context_precision.std
                if self.context_precision else None
            ),
            "correctness_mean":        (
                self.correctness.mean if self.correctness else None
            ),
            "correctness_std":         (
                self.correctness.std if self.correctness else None
            ),
            "composite_mean":          self.composite_mean,
            "composite_std":           self.composite_std,
            "composite_median":        self.composite_median,
            "avg_rag_latency_ms":      round(self.avg_rag_latency_ms, 1),
            "avg_eval_latency_ms":     round(self.avg_eval_latency_ms, 1),
            "avg_total_latency_ms":    round(self.avg_total_latency_ms, 1),
            "total_input_tokens":      self.total_input_tokens,
            "total_output_tokens":     self.total_output_tokens,
            "total_cost_usd":          round(self.total_cost_usd, 6),
            "parse_failure_rate":      round(
                self.overall_parse_failure_rate, 4
            ),
            "low_sample_warning":      self.low_sample_warning,
        }

    def __repr__(self) -> str:
        return (
            f"AggregatedResult("
            f"model='{self.rag_model}', "
            f"composite={self.composite_mean:.2f}±{self.composite_std:.2f}, "
            f"n={self.n_pairs_evaluated})"
        )

# ===========================================================================
# EVAL REPORT — all results for one model on one dataset
# ===========================================================================

class EvalReport(BaseModel):
    """
    Complete evaluation report for one model run on one dataset.

    The unit of persistence in SQLite (one row in RunRecord table,
    full results stored as JSON blob or in EvalResultRecord rows).

    Contains:
      - Run metadata (IDs, models, timestamps, status)
      - All EvalResults (one per QAPair)
      - AggregatedResult (statistics across all pairs)

    Frozen — a completed run's results are immutable.
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    run_id: Annotated[
        str,
        Field(description="Unique run ID. Format: 'run_{ulid}'."),
    ]

    dataset_id: str
    dataset_name: str

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    rag_model: str
    judge_model: str
    prompt_version: str
    top_k: int = Field(ge=1, description="Retriever top_k used.")
    temperature: float = Field(ge=0.0, le=2.0)
    collection_name: str = Field(
        description="ChromaDB collection queried."
    )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    started_at: datetime
    completed_at: datetime | None = None
    total_latency_ms: float = Field(ge=0.0, default=0.0)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    status: RunStatus = Field(default=RunStatus.PENDING)
    error_message: str | None = None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    results: Annotated[
        list[EvalResult],
        Field(
            default_factory=list,
            description="One EvalResult per evaluated QAPair.",
        ),
    ]

    aggregate: AggregatedResult | None = Field(
        default=None,
        description=(
            "Populated by ResultAggregator after all pairs are evaluated."
        ),
    )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def is_complete(self) -> bool:
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
        )

    @property
    def reliable_results(self) -> list[EvalResult]:
        """EvalResults where no parse failures occurred."""
        return [r for r in self.results if r.is_reliable]

    @property
    def composite_scores(self) -> list[float]:
        """All composite scores for distribution plotting."""
        return [r.composite_score for r in self.results]

    @property
    def avg_composite_score(self) -> float:
        scores = self.composite_scores
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    def get_result_for_pair(
        self,
        pair_id: str,
    ) -> EvalResult | None:
        """Find EvalResult by pair ID. Returns None if not found."""
        for result in self.results:
            if result.pair_id == pair_id:
                return result
        return None

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> EvalReport:
        return cls.model_validate_json(json_str)

    def __repr__(self) -> str:
        return (
            f"EvalReport("
            f"run_id='{self.run_id}', "
            f"model='{self.rag_model}', "
            f"dataset='{self.dataset_name}', "
            f"n_results={self.n_results}, "
            f"status={self.status.value})"
        )

# ===========================================================================
# COMPARISON MATRIX — multiple models side-by-side
# ===========================================================================

class ModelComparisonEntry(BaseModel):
    """
    One row in the ComparisonMatrix — one model's aggregated results.

    Combines AggregatedResult with display metadata for the dashboard.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    rag_model: str
    display_name: str
    provider: str

    # Radar chart values — mean score per metric
    faithfulness_mean: float = Field(ge=0.0, le=5.0, default=0.0)
    answer_relevance_mean: float = Field(ge=0.0, le=5.0, default=0.0)
    context_precision_mean: float = Field(ge=0.0, le=5.0, default=0.0)
    correctness_mean: float = Field(ge=0.0, le=5.0, default=0.0)
    composite_mean: float = Field(ge=0.0, le=5.0, default=0.0)

    # Std for error bars in bar chart
    faithfulness_std: float = Field(ge=0.0, default=0.0)
    answer_relevance_std: float = Field(ge=0.0, default=0.0)
    context_precision_std: float = Field(ge=0.0, default=0.0)
    correctness_std: float = Field(ge=0.0, default=0.0)
    composite_std: float = Field(ge=0.0, default=0.0)

    # Performance
    avg_latency_ms: float = Field(ge=0.0, default=0.0)
    total_cost_usd: float = Field(ge=0.0, default=0.0)
    n_evaluated: int = Field(ge=0, default=0)
    parse_failure_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    low_sample_warning: bool = False

    @classmethod
    def from_aggregated_result(
        cls,
        result: AggregatedResult,
        display_name: str,
        provider: str,
    ) -> ModelComparisonEntry:
        """Construct from an AggregatedResult + display metadata."""
        radar = result.radar_values
        return cls(
            run_id=result.run_id,
            rag_model=result.rag_model,
            display_name=display_name,
            provider=provider,
            faithfulness_mean=radar.get("faithfulness", 0.0),
            answer_relevance_mean=radar.get("answer_relevance", 0.0),
            context_precision_mean=radar.get("context_precision", 0.0),
            correctness_mean=radar.get("correctness", 0.0),
            composite_mean=result.composite_mean,
            faithfulness_std=(
                result.faithfulness.std if result.faithfulness else 0.0
            ),
            answer_relevance_std=(
                result.answer_relevance.std
                if result.answer_relevance else 0.0
            ),
            context_precision_std=(
                result.context_precision.std
                if result.context_precision else 0.0
            ),
            correctness_std=(
                result.correctness.std if result.correctness else 0.0
            ),
            composite_std=result.composite_std,
            avg_latency_ms=result.avg_total_latency_ms,
            total_cost_usd=result.total_cost_usd,
            n_evaluated=result.n_pairs_evaluated,
            parse_failure_rate=result.overall_parse_failure_rate,
            low_sample_warning=result.low_sample_warning,
        )

    @property
    def radar_values(self) -> dict[str, float]:
        return {
            "faithfulness":      self.faithfulness_mean,
            "answer_relevance":  self.answer_relevance_mean,
            "context_precision": self.context_precision_mean,
            "correctness":       self.correctness_mean,
        }

    def to_table_row(self) -> dict[str, Any]:
        """Flat dict for the Streamlit comparison table."""
        return {
            "Model":              self.display_name,
            "Provider":           self.provider,
            "Composite ↑":        round(self.composite_mean, 3),
            "Faithfulness ↑":     round(self.faithfulness_mean, 3),
            "Ans. Relevance ↑":   round(self.answer_relevance_mean, 3),
            "Ctx. Precision ↑":   round(self.context_precision_mean, 3),
            "Correctness ↑":      round(self.correctness_mean, 3),
            "Latency (ms) ↓":     round(self.avg_latency_ms, 1),
            "Cost (USD) ↓":       round(self.total_cost_usd, 5),
            "N Evaluated":        self.n_evaluated,
            "Parse Fail %":       round(
                self.parse_failure_rate * 100, 1
            ),
            "⚠ Low Sample":       "⚠" if self.low_sample_warning else "",
        }

    def __repr__(self) -> str:
        return (
            f"ModelComparisonEntry("
            f"model='{self.display_name}', "
            f"composite={self.composite_mean:.2f}, "
            f"latency={self.avg_latency_ms:.0f}ms)"
        )

class ComparisonMatrix(BaseModel):
    """
    Side-by-side comparison of multiple model runs on the same dataset.

    The root object consumed by the Streamlit dashboard for all
    comparison visualisations:
      - Radar chart (ui/components/charts.py → radar_chart())
      - Bar chart (metric_bar_chart())
      - Latency vs quality scatter (latency_quality_scatter())
      - Score distribution (score_distribution())
      - Comparison table (tables.py → comparison_table())

    Also persisted to SQLite for run history and CSV export.
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    matrix_id: Annotated[
        str,
        Field(description="Unique matrix ID. Format: 'cmp_{ulid}'."),
    ]

    dataset_id: str
    dataset_name: str
    created_at: datetime
    judge_model: str
    prompt_version: str

    # ------------------------------------------------------------------
    # Entries — one per model run
    # ------------------------------------------------------------------

    entries: Annotated[
        list[ModelComparisonEntry],
        Field(
            min_length=1,
            description=(
                "One entry per model run. "
                "Minimum 1 entry (single-model eval is valid)."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Raw reports — for drilldown access
    # ------------------------------------------------------------------

    reports: Annotated[
        list[EvalReport],
        Field(
            default_factory=list,
            description=(
                "Full EvalReports for drilldown. "
                "Parallel list to entries — entries[i] summarises reports[i]."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def unique_models_in_entries(self) -> ComparisonMatrix:
        """
        Each model should appear at most once in entries.

        Duplicate models produce misleading radar charts.
        If the same model was run multiple times, use the most recent.
        """
        models_seen: set[str] = set()
        duplicates = []
        for entry in self.entries:
            if entry.rag_model in models_seen:
                duplicates.append(entry.rag_model)
            models_seen.add(entry.rag_model)

        if duplicates:
            import warnings
            warnings.warn(
                f"ComparisonMatrix: Duplicate models in entries: "
                f"{duplicates}. "
                f"Dashboard charts may show duplicate series.",
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def model_ids(self) -> list[str]:
        return [e.rag_model for e in self.entries]

    @property
    def best_model_by_composite(self) -> ModelComparisonEntry | None:
        """Return the entry with the highest composite_mean."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.composite_mean)

    @property
    def fastest_model(self) -> ModelComparisonEntry | None:
        """Return the entry with the lowest avg_latency_ms."""
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e.avg_latency_ms)

    @property
    def cheapest_model(self) -> ModelComparisonEntry | None:
        """Return the entry with the lowest total_cost_usd."""
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e.total_cost_usd)

    @property
    def has_multiple_models(self) -> bool:
        return len(self.entries) > 1

    def get_entry_for_model(
        self,
        rag_model: str,
    ) -> ModelComparisonEntry | None:
        for entry in self.entries:
            if entry.rag_model == rag_model:
                return entry
        return None

    def get_report_for_model(
        self,
        rag_model: str,
    ) -> EvalReport | None:
        for report in self.reports:
            if report.rag_model == rag_model:
                return report
        return None

    def to_comparison_table_rows(self) -> list[dict[str, Any]]:
        """All entries as table rows for the Streamlit data table."""
        return [e.to_table_row() for e in self.entries]

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ComparisonMatrix:
        return cls.model_validate_json(json_str)

    def __repr__(self) -> str:
        return (
            f"ComparisonMatrix("
            f"matrix_id='{self.matrix_id}', "
            f"dataset='{self.dataset_name}', "
            f"models={self.model_ids}, "
            f"entries={len(self.entries)})"
        )

# ===========================================================================
# FACTORY HELPERS
# ===========================================================================

def make_run_id() -> str:
    """Generate a ULID-based run ID. Format: 'run_{ulid}'."""
    return f"run_{_generate_ulid()}"

def make_eval_result_id() -> str:
    """Generate a ULID-based EvalResult ID. Format: 'evr_{ulid}'."""
    return f"evr_{_generate_ulid()}"

def make_matrix_id() -> str:
    """Generate a ULID-based matrix ID. Format: 'cmp_{ulid}'."""
    return f"cmp_{_generate_ulid()}"

def _generate_ulid() -> str:
    """ULID generation with UUID4 fallback."""
    try:
        import ulid
        return str(ulid.new())
    except ImportError:
        import uuid
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"{ts:013x}{uuid.uuid4().hex[:13]}"

__all__ = [
    # Enums
    "RunStatus",
    "AggregationStrategy",
    # Core schemas
    "EvalResult",
    "MetricStats",
    "AggregatedResult",
    "EvalReport",
    "ModelComparisonEntry",
    "ComparisonMatrix",
    # Re-exports for convenience
    "MetricScore",
    "QAPair",
    # Factories
    "make_run_id",
    "make_eval_result_id",
    "make_matrix_id",
]