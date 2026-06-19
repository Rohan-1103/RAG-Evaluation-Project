"""
src/evaluation/base.py

Abstract base class for all LLM-as-a-Judge metric evaluators.

Design contract:
  - Every evaluator receives a QAPair and returns a MetricScore.
  - No evaluator touches the RAG pipeline, vector store, or dataset store.
    Single responsibility: score one metric on one QAPair.
  - The ABC enforces the full interface. Four concrete evaluators
    (Faithfulness, AnswerRelevance, ContextPrecision, Correctness)
    implement this interface — the EvaluationEngine depends only on
    BaseEvaluator, never on any concrete class.
  - Every evaluator is stateless after construction. The same instance
    can score thousands of pairs concurrently without locking.
  - Retry logic, JSON parsing, and score validation are implemented
    once here in the base class. Concrete evaluators implement only
    _build_prompt() — the single method that differs per metric.

Why abstract the evaluator per metric instead of one big evaluate() call:
  - Each metric has an independent prompt, rubric, and parsing logic.
  - The EvaluationEngine runs all 4 concurrently via asyncio.gather().
    Independent classes make this natural — each is a separate coroutine.
  - Unit testing: FaithfulnessEvaluator can be tested with a mock LLM
    without touching AnswerRelevanceEvaluator at all.
  - Adding a 5th metric (e.g. "Coherence") requires one new class and
    one registration in EvaluationEngine — zero changes elsewhere.
  - Individual metric evaluators can be disabled per run
    (e.g. skip Correctness when no ground truth is available)
    without any if/else logic in the engine.

Score validation contract:
  Every evaluate() call returns a MetricScore with:
    - score in [1.0, 5.0] — never outside this range
    - reasoning with min_length characters — never empty
    - parse_failed=True if JSON parsing failed (score is fallback value)
    - low_confidence=True if reasoning is suspiciously short
  The engine NEVER receives a raw LLM string — only typed MetricScore.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import JudgeConfig, Settings
from src.dataset.schema import MetricScore, QAPair
from src.evaluation.schema import make_eval_result_id

# ===========================================================================
# JUDGE RESPONSE — intermediate parsed representation
# ===========================================================================

class JudgeResponse:
    """
    Parsed intermediate representation of a judge LLM response.

    The evaluator parses raw LLM text into JudgeResponse first,
    then promotes it to MetricScore. This two-step process:
      1. Isolates JSON parsing from score validation.
      2. Makes the parsing logic testable in isolation.
      3. Allows the base class to handle all parsing consistently.

    Not a Pydantic model — it is a transient object that never
    leaves the evaluator. Only MetricScore crosses module boundaries.
    """

    __slots__ = (
        "raw_text",
        "score",
        "reasoning",
        "parse_failed",
        "parse_error",
    )

    def __init__(
        self,
        raw_text: str,
        score: float | None,
        reasoning: str,
        parse_failed: bool,
        parse_error: str | None = None,
    ) -> None:
        self.raw_text = raw_text
        self.score = score
        self.reasoning = reasoning
        self.parse_failed = parse_failed
        self.parse_error = parse_error

    @property
    def is_valid(self) -> bool:
        return (
            not self.parse_failed
            and self.score is not None
            and 1.0 <= self.score <= 5.0
            and bool(self.reasoning.strip())
        )

    def __repr__(self) -> str:
        return (
            f"JudgeResponse("
            f"score={self.score}, "
            f"parse_failed={self.parse_failed}, "
            f"reasoning='{self.reasoning[:40]}...')"
        )

# ===========================================================================
# ABSTRACT BASE CLASS
# ===========================================================================

class BaseEvaluator(ABC):
    """
    Abstract base for all metric evaluators.

    Subclasses implement:
      metric_name   — the snake_case metric identifier
      _build_prompt — constructs the judge prompt for this metric

    Subclasses must NOT implement:
      evaluate()       — public sync interface (template method)
      aevaluate()      — public async interface (template method)
      _call_judge()    — Gemini API call with retry
      _parse_response()— JSON parsing and validation
      _to_metric_score()— JudgeResponse → MetricScore promotion

    The template method pattern ensures:
      - All evaluators share identical retry, parsing, and validation.
      - Adding a new metric never requires reimplementing these.
      - Unit tests mock _build_prompt() alone — the rest is tested once.

    Thread and async safety:
      All instance state is set at construction and never mutated.
      evaluate() and aevaluate() are safe to call concurrently.
    """

    def __init__(
        self,
        judge_config: JudgeConfig,
        eval_config: Any,          # EvalConfig from config/__init__.py
        gemini_api_key: str,
    ) -> None:
        """
        Initialise the evaluator.

        Args:
            judge_config:   JudgeConfig from Settings — controls model,
                            temperature, retry behaviour, score thresholds.
            eval_config:    EvalConfig loaded from eval.yaml — provides
                            the prompt template and rubric for this metric.
            gemini_api_key: Gemini API key for judge LLM calls.
        """
        self._judge_config = judge_config
        self._eval_config = eval_config
        self._api_key = gemini_api_key
        self._model: Any = None       # google.generativeai.GenerativeModel
        self._metric_config: Any = None   # MetricConfig from eval.yaml

        self._initialise_model()
        self._load_metric_config()

        logger.debug(
            f"{self.__class__.__name__} initialised. "
            f"metric={self.metric_name}, "
            f"judge_model={judge_config.model}, "
            f"temperature={judge_config.temperature}"
        )

    # ------------------------------------------------------------------
    # Abstract interface — implement in subclasses
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        Snake_case metric identifier.

        Must match a key in eval.yaml metrics section:
          "faithfulness" | "answer_relevance" |
          "context_precision" | "correctness"

        Used to:
          - Load the correct MetricConfig from eval.yaml
          - Key the MetricScore in EvalResult.metric_scores dict
          - Label the score in logs and dashboard
        """
        ...

    @abstractmethod
    def _build_prompt(self, pair: QAPair) -> str:
        """
        Build the complete judge prompt for this metric and QAPair.

        Called by evaluate() after all pre-checks pass.

        Args:
            pair: QAPair with generated_answer and retrieved_chunks
                  already populated. Callers must ensure pair.status
                  is ANSWERED before calling evaluate().

        Returns:
            Complete prompt string ready to send to the judge LLM.
            Must include all required variables:
              - question
              - generated answer
              - context chunks (as numbered list)
              - reference answer (for Correctness only)
              - output_format_instruction (from eval.yaml)

        Contract:
          - Never raises — if prompt building fails, return a minimal
            fallback prompt. The judge call will likely fail too but
            this is handled at the calling level.
          - Never makes API calls.
          - Never accesses the filesystem.
        """
        ...

    # ------------------------------------------------------------------
    # Public interface — template methods, do not override
    # ------------------------------------------------------------------

    def evaluate(self, pair: QAPair) -> MetricScore:
        """
        Score this metric for a single QAPair. Synchronous.

        This is the primary public interface for the EvaluationEngine
        when running metrics sequentially.

        Args:
            pair: QAPair with status=ANSWERED (generated_answer populated).
                  Pairs with status=PENDING are rejected immediately.

        Returns:
            MetricScore with score, reasoning, and quality flags.
            NEVER raises — all failures return a MetricScore with
            parse_failed=True and the fallback score.

        Do not override. Implement _build_prompt() instead.
        """
        start_ms = time.monotonic() * 1000

        # Pre-check
        skip_result = self._should_skip(pair)
        if skip_result is not None:
            elapsed = time.monotonic() * 1000 - start_ms
            return skip_result

        prompt = self._build_prompt(pair)

        # Call judge with retry
        raw_text, input_tokens, output_tokens = (
            self._call_judge_with_fallback(prompt)
        )

        # Parse response
        judge_response = self._parse_response(raw_text)

        # Promote to MetricScore
        elapsed_ms = time.monotonic() * 1000 - start_ms
        score = self._to_metric_score(
            judge_response=judge_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

        self._log_score(pair, score)
        return score

    async def aevaluate(self, pair: QAPair) -> MetricScore:
        """
        Score this metric for a single QAPair. Asynchronous.

        Used by EvaluationEngine when running metrics concurrently
        via asyncio.gather(). Wraps the synchronous evaluate() in
        an executor to avoid blocking the event loop.

        For a true async implementation (e.g. using httpx for the
        Gemini API), override this method in the concrete evaluator
        and call the async SDK directly.

        Do not override the executor wrapping unless the subclass
        has a native async SDK available.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, pair)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    # def _initialise_model(self) -> None:
    #     """
    #     Configure Gemini client and create the judge model instance.

    #     Called once at construction. Judge temperature is always 0.0
    #     for deterministic, reproducible scoring — the JudgeConfig
    #     validator enforces this but we log a warning if it's not 0.
    #     """
    #     try:
    #         import google.generativeai as genai

    #         genai.configure(api_key=self._api_key)

    #         if self._judge_config.temperature != 0.0:
    #             logger.warning(
    #                 f"{self.__class__.__name__}: "
    #                 f"judge temperature={self._judge_config.temperature} "
    #                 f"(non-zero). Judge evaluations should use "
    #                 f"temperature=0.0 for reproducibility."
    #             )

    #         generation_config = genai.GenerationConfig(
    #             temperature=self._judge_config.temperature,
    #             max_output_tokens=self._eval_config.judge.max_response_tokens,
    #             response_mime_type="text/plain",
    #         )

    #         self._model = genai.GenerativeModel(
    #             model_name=self._judge_config.model,
    #             generation_config=generation_config,
    #         )

    #     except ImportError as exc:
    #         raise ImportError(
    #             "google-generativeai is not installed. "
    #             "Run: poetry add google-generativeai"
    #         ) from exc
    #     except Exception as exc:
    #         raise RuntimeError(
    #             f"{self.__class__.__name__}: Failed to initialise "
    #             f"judge model '{self._judge_config.model}': {exc}"
    #         ) from exc
    
    def _initialise_model(self) -> None:
        """
            Configure Gemini client and create the judge model instance.

            Called once at construction. Judge temperature is always 0.0
            for deterministic, reproducible scoring — the JudgeConfig
            validator enforces this but we log a warning if it's not 0.
            """
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self._api_key)
            self._model = client
            self._gen_config = types.GenerateContentConfig(
                temperature=self._judge_config.temperature,
                max_output_tokens=self._eval_config.judge.max_response_tokens,
            )
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{self.__class__.__name__}: Failed to initialise "
                f"judge model '{self._judge_config.model}': {exc}"
            ) from exc

    def _load_metric_config(self) -> None:
        """
        Load the MetricConfig for this evaluator from eval.yaml.

        Called after _initialise_model(). Validates that the metric_name
        property matches a metric defined in eval.yaml.
        """
        try:
            self._metric_config = self._eval_config.metrics.get_metric(
                self.metric_name
            )
        except KeyError as exc:
            raise ValueError(
                f"{self.__class__.__name__}: metric_name='{self.metric_name}' "
                f"not found in eval.yaml metrics section. "
                f"Valid names: faithfulness, answer_relevance, "
                f"context_precision, correctness."
            ) from exc

    # ------------------------------------------------------------------
    # Skip logic
    # ------------------------------------------------------------------

    def _should_skip(self, pair: QAPair) -> MetricScore | None:
        """
        Check whether this pair should be skipped for this metric.

        Returns a MetricScore with parse_failed=True if the pair
        should be skipped, or None if evaluation should proceed.

        Base class checks:
          - pair.generated_answer must be non-empty
          - pair.status must not be PENDING

        Subclasses override to add metric-specific skip logic:
          - CorrectnessEvaluator skips if no ground_truth_answer
        """
        from src.dataset.schema import QAPairStatus

        if pair.status == QAPairStatus.PENDING:
            logger.warning(
                f"{self.__class__.__name__}: QAPair '{pair.id}' "
                f"is PENDING (no generated_answer). "
                f"Run RAGPipeline before evaluation."
            )
            return self._make_skip_score(
                reason="QAPair is PENDING — no generated answer.",
            )

        if not pair.generated_answer or not pair.generated_answer.strip():
            return self._make_skip_score(
                reason="generated_answer is empty.",
            )

        return None

    def _make_skip_score(self, reason: str) -> MetricScore:
        """
        Construct a MetricScore for a skipped evaluation.

        Uses the fallback score from eval.yaml (default 1.0).
        Marks parse_failed=True so the aggregator excludes it
        from reliable statistics.
        """
        fallback = self._eval_config.judge.parse_failure_fallback_score
        return MetricScore(
            metric_name=self.metric_name,
            score=fallback,
            reasoning=f"SKIPPED: {reason}",
            prompt_version=self._eval_config.active_prompt_version,
            judge_model=self._judge_config.model,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            parse_failed=True,
            low_confidence=True,
        )

    # ------------------------------------------------------------------
    # Judge API call
    # ------------------------------------------------------------------

    def _call_judge_with_fallback(
        self,
        prompt: str,
    ) -> tuple[str, int, int]:
        """
        Call the judge LLM with retry. Returns (text, in_tokens, out_tokens).

        If all retries fail, returns ("", 0, 0) — the caller handles
        empty response as a parse failure. Never raises.
        """
        try:
            return self._call_judge_api(prompt)
        except Exception as exc:
            logger.error(
                f"{self.__class__.__name__}: Judge API call failed "
                f"after all retries: {exc}"
            )
            return "", 0, 0

    # @retry(
    #     retry=retry_if_exception_type(Exception),
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=2, min=2, max=30),
    #     reraise=True,
    # )
    # def _call_judge_api(
    #     self,
    #     prompt: str,
    # ) -> tuple[str, int, int]:
    #     """
    #     Single Gemini API call with tenacity exponential backoff.

    #     Retries on any exception:
    #       - 429 rate limit (most common on free tier)
    #       - 503 service unavailable
    #       - Network timeout

    #     Returns (response_text, input_tokens, output_tokens).
    #     Raises on non-retriable errors (reraise=True) after 3 attempts.
    #     """
    #     if self._model is None:
    #         raise RuntimeError(
    #             f"{self.__class__.__name__}: Model not initialised."
    #         )

    #     response = self._model.generate_content(prompt)

    #     # Extract token counts
    #     input_tokens = 0
    #     output_tokens = 0
    #     if hasattr(response, "usage_metadata") and response.usage_metadata:
    #         input_tokens = getattr(
    #             response.usage_metadata, "prompt_token_count", 0
    #         ) or 0
    #         output_tokens = getattr(
    #             response.usage_metadata, "candidates_token_count", 0
    #         ) or 0

    #     # Extract response text
    #     text = ""
    #     if response.candidates:
    #         candidate = response.candidates[0]
    #         if candidate.content and candidate.content.parts:
    #             text = "".join(
    #                 part.text
    #                 for part in candidate.content.parts
    #                 if hasattr(part, "text")
    #             )

    #     if not text.strip():
    #         raise EvaluatorError(
    #             evaluator=self.__class__.__name__,
    #             metric=self.metric_name,
    #             reason=(
    #                 "Judge returned empty response. "
    #                 "May be a safety filter block."
    #             ),
    #         )

    #     return text, input_tokens, output_tokens
    
    @retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
    def _call_judge_api(self, prompt: str) -> tuple[str, int, int]:
        response = self._model.models.generate_content(
            model=self._judge_config.model,
            contents=prompt,
            config=self._gen_config,
        )
        input_tokens = 0
        output_tokens = 0
        """
        Single Gemini API call with tenacity exponential backoff.

        Retries on any exception:
          - 429 rate limit (most common on free tier)
          - 503 service unavailable
          - Network timeout

        Returns (response_text, input_tokens, output_tokens).
        Raises on non-retriable errors (reraise=True) after 3 attempts.
        """
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
        text = response.text or ""
        if not text.strip():
            raise EvaluatorError(
                evaluator=self.__class__.__name__,
                metric=self.metric_name,
                reason="Judge returned empty response.",
            )
        return text, input_tokens, output_tokens
    

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw_text: str) -> JudgeResponse:
        """
        Parse the judge's raw text response into a JudgeResponse.

        Expected format (from eval.yaml output_format_instruction):
            {
              "reasoning": "<step-by-step analysis>",
              "score": <integer 1-5>
            }

        Handles:
          1. Markdown code fences (```json ... ```)
          2. Score as string instead of int ("3" instead of 3)
          3. Score as float (3.0 instead of 3)
          4. Extra fields (ignored)
          5. Reasoning and score keys with different capitalisation
          6. Empty or whitespace-only response
          7. Response with text before/after JSON
        """
        if not raw_text or not raw_text.strip():
            return JudgeResponse(
                raw_text=raw_text,
                score=None,
                reasoning="",
                parse_failed=True,
                parse_error="Empty response from judge.",
            )

        # Step 1: Extract JSON from response
        json_text = self._extract_json_from_text(raw_text)
        if not json_text:
            return JudgeResponse(
                raw_text=raw_text,
                score=None,
                reasoning="",
                parse_failed=True,
                parse_error=(
                    f"Could not extract JSON from response: "
                    f"'{raw_text[:100]}'"
                ),
            )

        # Step 2: Fix common JSON issues
        json_text = self._fix_json(json_text)

        # Step 3: Parse JSON
        try:
            parsed: dict[str, Any] = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return JudgeResponse(
                raw_text=raw_text,
                score=None,
                reasoning="",
                parse_failed=True,
                parse_error=f"JSON parse error: {exc}",
            )

        if not isinstance(parsed, dict):
            return JudgeResponse(
                raw_text=raw_text,
                score=None,
                reasoning="",
                parse_failed=True,
                parse_error=(
                    f"Expected JSON object, got {type(parsed).__name__}."
                ),
            )

        # Step 4: Extract reasoning (try multiple key variants)
        reasoning = self._extract_string_field(
            parsed,
            keys=["reasoning", "Reasoning", "reason", "explanation"],
        )

        # Step 5: Extract score (try multiple key variants)
        raw_score = self._extract_field(
            parsed,
            keys=["score", "Score", "rating", "Rating"],
        )

        # Step 6: Coerce score to float
        score, score_error = self._coerce_score(raw_score)

        if score_error:
            return JudgeResponse(
                raw_text=raw_text,
                score=None,
                reasoning=reasoning or "",
                parse_failed=True,
                parse_error=score_error,
            )

        return JudgeResponse(
            raw_text=raw_text,
            score=score,
            reasoning=reasoning or "",
            parse_failed=False,
        )

    def _extract_json_from_text(self, text: str) -> str | None:
        """
        Extract JSON object from raw LLM response.

        Handles:
          - Markdown fences: ```json {...} ```
          - Clean JSON starting with {
          - JSON embedded within explanation text
        """
        text = text.strip()

        # Pattern 1: Markdown code fence
        fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(\{.*?\})\n?```",
            re.DOTALL,
        )
        match = fence_pattern.search(text)
        if match:
            return match.group(1).strip()

        # Pattern 2: Clean JSON object
        if text.startswith("{"):
            return text

        # Pattern 3: Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start: end + 1]

        return None

    def _fix_json(self, json_text: str) -> str:
        """Apply heuristic fixes to malformed judge JSON."""
        # Trailing commas before }
        json_text = re.sub(r",\s*}", "}", json_text)
        # Trailing commas before ]
        json_text = re.sub(r",\s*]", "]", json_text)
        return json_text

    @staticmethod
    def _extract_string_field(
        data: dict[str, Any],
        keys: list[str],
    ) -> str | None:
        """Try multiple key names and return first non-empty string."""
        for key in keys:
            val = data.get(key)
            if val is not None:
                return str(val).strip() or None
        return None

    @staticmethod
    def _extract_field(
        data: dict[str, Any],
        keys: list[str],
    ) -> Any:
        """Try multiple key names and return first non-None value."""
        for key in keys:
            val = data.get(key)
            if val is not None:
                return val
        return None

    def _coerce_score(
        self,
        raw_score: Any,
    ) -> tuple[float | None, str | None]:
        """
        Coerce a raw score value to float in [1.0, 5.0].

        Returns (score, error_message).
        error_message is None on success.

        Handles:
          - int: 3 → 3.0
          - float: 3.5 → 3.5
          - str: "3" → 3.0, "3.5" → 3.5
          - out of range: clipped with warning
          - None: error
          - non-numeric string: error
        """
        score_min = self._eval_config.judge.score_min
        score_max = self._eval_config.judge.score_max

        if raw_score is None:
            return None, "Score field not found in judge response."

        # Coerce to float
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return None, (
                f"Cannot convert score '{raw_score}' "
                f"({type(raw_score).__name__}) to float."
            )

        # Clip to valid range with warning
        if score < score_min or score > score_max:
            logger.warning(
                f"{self.__class__.__name__}: Judge score {score} is "
                f"outside valid range [{score_min}, {score_max}]. "
                f"Clipping."
            )
            score = max(float(score_min), min(float(score_max), score))

        return score, None

    # ------------------------------------------------------------------
    # MetricScore construction
    # ------------------------------------------------------------------

    def _to_metric_score(
        self,
        judge_response: JudgeResponse,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> MetricScore:
        """
        Promote a JudgeResponse to a MetricScore.

        Applies:
          - Fallback score if parse_failed
          - low_confidence flag if reasoning is too short
          - prompt_version from eval.yaml active version
        """
        fallback = self._eval_config.judge.parse_failure_fallback_score
        min_reasoning_len = (
            self._eval_config.judge.min_reasoning_length
        )

        if judge_response.parse_failed or judge_response.score is None:
            score = fallback
            reasoning = (
                judge_response.reasoning
                or f"PARSE_FAILED: {judge_response.parse_error}"
            )
            parse_failed = True
        else:
            score = judge_response.score
            reasoning = judge_response.reasoning
            parse_failed = False

        low_confidence = len(reasoning.strip()) < min_reasoning_len

        if parse_failed:
            logger.warning(
                f"{self.__class__.__name__}: Parse failed. "
                f"Using fallback score={fallback}. "
                f"Error: {judge_response.parse_error}"
            )

        if low_confidence and not parse_failed:
            logger.debug(
                f"{self.__class__.__name__}: Low confidence — "
                f"reasoning length={len(reasoning.strip())} chars "
                f"(min={min_reasoning_len})."
            )

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=reasoning,
            prompt_version=self._eval_config.active_prompt_version,
            judge_model=self._judge_config.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            parse_failed=parse_failed,
            low_confidence=low_confidence,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_score(self, pair: QAPair, score: MetricScore) -> None:
        """Structured log entry per evaluation call."""
        level = "debug"
        if score.parse_failed:
            level = "warning"
        elif score.low_confidence:
            level = "debug"

        getattr(logger, level)(
            f"{self.__class__.__name__}: "
            f"pair='{pair.id}' "
            f"metric={self.metric_name} "
            f"score={score.score:.1f} "
            f"parse_failed={score.parse_failed} "
            f"low_confidence={score.low_confidence} "
            f"latency={score.latency_ms:.0f}ms "
            f"tokens={score.input_tokens}+{score.output_tokens}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> BaseEvaluator:
        """
        Standard factory — construct from application Settings.

        Loads EvalConfig from eval.yaml automatically.
        Subclasses inherit this factory.

        Usage:
            evaluator = FaithfulnessEvaluator.from_settings(settings)
        """
        from config import get_eval_config

        return cls(
            judge_config=settings.judge,
            eval_config=get_eval_config(),
            gemini_api_key=settings.gemini.api_key,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def judge_model(self) -> str:
        return self._judge_config.model

    @property
    def prompt_version(self) -> str:
        return self._eval_config.active_prompt_version

    @property
    def weight(self) -> float:
        """This metric's weight in the composite score."""
        return self._metric_config.weight if self._metric_config else 0.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"metric='{self.metric_name}', "
            f"judge='{self.judge_model}', "
            f"weight={self.weight})"
        )

# ===========================================================================
# CUSTOM EXCEPTIONS
# ===========================================================================

class EvaluatorError(Exception):
    """
    Raised when an evaluator cannot complete a metric score.

    Distinct from parse failures (which return MetricScore with
    parse_failed=True) — EvaluatorError means the API call itself
    failed, not the response parsing.

    The EvaluationEngine catches this at the pair level and marks
    the pair as failed without aborting the full run.
    """

    def __init__(
        self,
        evaluator: str,
        metric: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.metric = metric
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[{evaluator}/{metric}] {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )

class ScoreOutOfRangeError(EvaluatorError):
    """
    Raised when a judge returns a score outside [score_min, score_max]
    that cannot be clipped without losing meaning.

    In practice, clipping is always applied — this error is reserved
    for cases where the score is so far outside range (e.g. 99, -5)
    that it clearly indicates a prompt or model misconfiguration.
    """

    def __init__(
        self,
        evaluator: str,
        metric: str,
        score: float,
        score_min: int,
        score_max: int,
    ) -> None:
        self.score = score
        self.score_min = score_min
        self.score_max = score_max
        super().__init__(
            evaluator=evaluator,
            metric=metric,
            reason=(
                f"Score {score} is far outside valid range "
                f"[{score_min}, {score_max}]. "
                f"Check judge prompt and model configuration."
            ),
        )

__all__ = [
    "JudgeResponse",
    "BaseEvaluator",
    "EvaluatorError",
    "ScoreOutOfRangeError",
]