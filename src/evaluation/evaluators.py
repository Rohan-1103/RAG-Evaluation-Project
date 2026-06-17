"""
src/evaluation/evaluators.py

Concrete LLM-as-a-Judge evaluators for all 4 RAG evaluation metrics.

Each evaluator:
  1. Inherits BaseEvaluator — gets retry, parsing, scoring for free.
  2. Implements metric_name — declares which metric it scores.
  3. Implements _build_prompt() — the only method that differs.

That is literally all each class does. The template method pattern
in BaseEvaluator means 95% of the logic lives once in the base class.

Prompt construction strategy:
  - Prompts are stored in eval.yaml (not hardcoded here).
  - Each evaluator calls eval_config.get_metric_prompt(metric_name, **vars)
    which injects variables and appends output_format_instruction.
  - Changing a prompt requires editing eval.yaml only — zero code changes.

Variable injection per metric:
  FaithfulnessEvaluator:
    {question} {answer} {context_chunks} {output_format_instruction}

  AnswerRelevanceEvaluator:
    {question} {answer} {output_format_instruction}

  ContextPrecisionEvaluator:
    {question} {answer} {context_chunks} {output_format_instruction}

  CorrectnessEvaluator:
    {question} {answer} {reference_answer} {output_format_instruction}

  Note: CorrectnessEvaluator does NOT receive context_chunks —
  correctness is measured against ground truth, not retrieved context.
  Injecting context would let the judge "cheat" by comparing context
  to reference rather than generated answer to reference.

Skip logic per metric:
  CorrectnessEvaluator overrides _should_skip() to skip pairs with
  no ground truth reference. All other evaluators use the base
  class skip logic (skip PENDING pairs and empty answers only).
"""

from __future__ import annotations

from loguru import logger

from src.dataset.schema import MetricScore, QAPair
from src.evaluation.base import BaseEvaluator

# ===========================================================================
# 1. FAITHFULNESS EVALUATOR
# ===========================================================================

class FaithfulnessEvaluator(BaseEvaluator):
    """
    Scores whether every claim in the answer is grounded in context.

    Primary hallucination detection metric.

    High score (4-5): Every factual claim is explicitly supported
                      by at least one retrieved chunk.
    Low score (1-2):  Answer contains facts not present in context,
                      contradicts context, or is largely fabricated.

    Prompt variables injected:
      {question}        — original user question
      {answer}          — generated answer to evaluate
      {context_chunks}  — numbered list of retrieved chunks
    """

    @property
    def metric_name(self) -> str:
        return "faithfulness"

    def _build_prompt(self, pair: QAPair) -> str:
        """
        Build the faithfulness judge prompt.

        Context chunks are formatted as a numbered list so the judge
        can reference specific chunks by number in its reasoning:
        "Claim X is not supported by any of chunks 1-5."
        """
        if not pair.retrieved_chunks:
            logger.warning(
                f"FaithfulnessEvaluator: QAPair '{pair.id}' has no "
                f"retrieved_chunks. Faithfulness score will be low — "
                f"an answer with no context is inherently ungrounded."
            )

        return self._eval_config.get_metric_prompt(
            "faithfulness",
            question=pair.question,
            answer=pair.generated_answer or "",
            context_chunks=pair.context_as_numbered_list,
            reference_answer=pair.ground_truth_answer,
        )

# ===========================================================================
# 2. ANSWER RELEVANCE EVALUATOR
# ===========================================================================

class AnswerRelevanceEvaluator(BaseEvaluator):
    """
    Scores whether the answer directly addresses the question.

    Independent of factual correctness and grounding.
    A wrong answer that stays on-topic scores higher than a correct
    answer to a different question.

    High score (4-5): Answer directly and completely addresses
                      all parts of the question.
    Low score (1-2):  Answer drifts off-topic, addresses a different
                      question, or gives context without answering.

    Prompt variables injected:
      {question} — original user question
      {answer}   — generated answer to evaluate

    Note: context_chunks are intentionally excluded.
    Answer relevance is purely about question ↔ answer alignment.
    Injecting context might cause the judge to score retrieval
    quality instead of answer relevance.
    """

    @property
    def metric_name(self) -> str:
        return "answer_relevance"

    def _build_prompt(self, pair: QAPair) -> str:
        """
        Build the answer relevance judge prompt.

        Only question and answer are injected — no context.
        This forces the judge to evaluate relevance purely based on
        whether the answer addresses the question, not whether it
        matches the retrieved documents.
        """
        return self._eval_config.get_metric_prompt(
            "answer_relevance",
            question=pair.question,
            answer=pair.generated_answer or "",
            context_chunks=pair.context_as_numbered_list,
            reference_answer=pair.ground_truth_answer,
        )

# ===========================================================================
# 3. CONTEXT PRECISION EVALUATOR
# ===========================================================================

class ContextPrecisionEvaluator(BaseEvaluator):
    """
    Scores retrieval signal-to-noise ratio.

    Measures what fraction of retrieved chunks were actually useful
    for answering the question. Evaluates retriever quality, not
    answer quality.

    High score (4-5): Most or all retrieved chunks are directly
                      relevant to the question.
    Low score (1-2):  Most chunks are off-topic noise. The retriever
                      failed to find relevant content.

    Prompt variables injected:
      {question}       — original user question
      {answer}         — generated answer (for reference only)
      {context_chunks} — numbered list of retrieved chunks

    Why include the generated answer in context precision:
      The judge needs to understand what the answer used to evaluate
      which chunks were helpful vs noise. Without the answer,
      the judge might score a chunk as irrelevant when it was actually
      the primary source for the generated response.
    """

    @property
    def metric_name(self) -> str:
        return "context_precision"

    def _build_prompt(self, pair: QAPair) -> str:
        """
        Build the context precision judge prompt.

        Both context and answer are injected. The judge evaluates
        each chunk independently for relevance to the question,
        using the generated answer as a reference for what was used.
        """
        if not pair.retrieved_chunks:
            logger.warning(
                f"ContextPrecisionEvaluator: QAPair '{pair.id}' has no "
                f"retrieved_chunks. Cannot evaluate precision with "
                f"empty context — will score 1 (no relevant chunks)."
            )

        return self._eval_config.get_metric_prompt(
            "context_precision",
            question=pair.question,
            answer=pair.generated_answer or "",
            context_chunks=pair.context_as_numbered_list,
            reference_answer=pair.ground_truth_answer,
        )

# ===========================================================================
# 4. CORRECTNESS EVALUATOR
# ===========================================================================

class CorrectnessEvaluator(BaseEvaluator):
    """
    Scores semantic agreement with the ground truth reference answer.

    The only metric that requires a ground truth dataset.
    Measures factual accuracy independently of retrieval or grounding.

    High score (4-5): Generated answer agrees with reference on all
                      key facts. Extra correct detail is acceptable.
    Low score (1-2):  Generated answer contradicts the reference or
                      omits critical facts.

    Skip behaviour:
      If pair.ground_truth_answer is empty, this evaluator skips
      the pair and returns a MetricScore with parse_failed=True
      and flag NO_REFERENCE. This is tracked in:
        - EvalResult.correctness_skipped = True
        - AggregatedResult.correctness_skip_rate
      The composite score is recomputed excluding this metric
      when it is skipped (exclude_from_weight strategy in eval.yaml).

    Prompt variables injected:
      {question}         — original user question
      {answer}           — generated answer to evaluate
      {reference_answer} — ground truth from the dataset

    Note: context_chunks are excluded from this prompt.
    Correctness measures generated answer vs reference answer.
    Injecting context would let the judge compare context to
    reference (a retrieval quality signal) rather than answer
    accuracy (what this metric measures).
    """

    @property
    def metric_name(self) -> str:
        return "correctness"

    def _should_skip(self, pair: QAPair) -> MetricScore | None:
        """
        Skip if no ground truth reference is available.

        Calls the base class check first (PENDING, empty answer),
        then adds the correctness-specific check (no reference).

        The NO_REFERENCE flag in the reasoning string is used by
        the EvaluationEngine to set EvalResult.correctness_skipped=True.
        """
        # Base class checks (PENDING, empty answer)
        base_skip = super()._should_skip(pair)
        if base_skip is not None:
            return base_skip

        # Correctness-specific: skip if no ground truth
        if not pair.has_reference:
            logger.debug(
                f"CorrectnessEvaluator: QAPair '{pair.id}' has no "
                f"ground_truth_answer. Skipping correctness evaluation. "
                f"Set ground_truth_answer to enable this metric."
            )
            return self._make_no_reference_score()

        return None

    def _make_no_reference_score(self) -> MetricScore:
        """
        Construct a MetricScore for the no-reference case.

        Uses the no_reference_behaviour from eval.yaml:
          - "skip": score=None is not possible in MetricScore (float field)
            so we use the fallback score and flag it.
          - "score_zero": uses 1.0 (minimum) as the score.

        The NO_REFERENCE flag in reasoning is parsed by the engine
        to set correctness_skipped=True on the EvalResult.
        """
        no_ref_config = self._metric_config

        # Determine score based on no_reference_behaviour
        behaviour = getattr(
            no_ref_config, "no_reference_behaviour", "skip"
        )
        flag = getattr(
            no_ref_config,
            "no_reference_flag",
            "NO_REFERENCE",
        )

        if behaviour == "score_zero":
            score = 1.0    # 1.0 = minimum on the 1-5 scale
        else:
            # "skip" behaviour — use fallback score
            score = self._eval_config.judge.parse_failure_fallback_score

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=f"{flag}: No ground truth reference available.",
            prompt_version=self._eval_config.active_prompt_version,
            judge_model=self._judge_config.model,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            parse_failed=True,
            low_confidence=True,
        )

    def _build_prompt(self, pair: QAPair) -> str:
        """
        Build the correctness judge prompt.

        Injects question, generated answer, and reference answer.
        Context chunks are deliberately excluded — see class docstring.

        This method is only called when pair.has_reference is True
        (guaranteed by _should_skip() returning None).
        """
        return self._eval_config.get_metric_prompt(
            "correctness",
            question=pair.question,
            answer=pair.generated_answer or "",
            context_chunks=pair.context_as_numbered_list,
            reference_answer=pair.ground_truth_answer,
        )

# ===========================================================================
# EVALUATOR REGISTRY
# ===========================================================================

# Maps metric_name → evaluator class
# Used by EvaluationEngine to instantiate all evaluators without
# importing concrete classes directly.
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "faithfulness":      FaithfulnessEvaluator,
    "answer_relevance":  AnswerRelevanceEvaluator,
    "context_precision": ContextPrecisionEvaluator,
    "correctness":       CorrectnessEvaluator,
}

def build_all_evaluators(
    settings: "Settings",  # noqa: F821 — avoid circular import
) -> dict[str, BaseEvaluator]:
    """
    Instantiate all 4 evaluators from Settings.

    Called once by EvaluationEngine at construction time.
    Returns a dict keyed by metric_name for O(1) lookup during eval.

    Usage:
        evaluators = build_all_evaluators(settings)
        score = evaluators["faithfulness"].evaluate(pair)

    Args:
        settings: Application Settings instance.

    Returns:
        Dict mapping metric_name → evaluator instance.
        All 4 evaluators are always constructed — the engine decides
        which to call per pair based on skip logic.
    """
    from config import get_eval_config
    from config.settings import get_settings

    eval_config = get_eval_config()

    evaluators: dict[str, BaseEvaluator] = {}

    for metric_name, evaluator_cls in EVALUATOR_REGISTRY.items():
        try:
            evaluator = evaluator_cls(
                judge_config=settings.judge,
                eval_config=eval_config,
                gemini_api_key=settings.gemini.api_key,
            )
            evaluators[metric_name] = evaluator
            logger.debug(
                f"build_all_evaluators: Instantiated "
                f"{evaluator_cls.__name__}."
            )
        except Exception as exc:
            logger.error(
                f"build_all_evaluators: Failed to instantiate "
                f"{evaluator_cls.__name__}: {exc}"
            )
            raise RuntimeError(
                f"Failed to build evaluator for metric '{metric_name}': "
                f"{exc}"
            ) from exc

    logger.info(
        f"build_all_evaluators: All {len(evaluators)} evaluators ready. "
        f"metrics={list(evaluators.keys())}"
    )

    return evaluators

__all__ = [
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextPrecisionEvaluator",
    "CorrectnessEvaluator",
    "EVALUATOR_REGISTRY",
    "build_all_evaluators",
]