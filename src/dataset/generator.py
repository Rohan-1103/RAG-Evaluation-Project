"""
src/dataset/generator.py

GeminiDatasetGenerator — Gemini-powered synthetic Q&A dataset generator.

Implements BaseDatasetGenerator._generate_from_chunk() using the
Google Gemini API to produce factual question-answer pairs from
document chunks.

Generation strategy:
  1. Build a structured prompt instructing Gemini to output a JSON
     array of {question, answer} objects.
  2. Call Gemini with temperature from GeneratorConfig.
  3. Parse the JSON response — handle partial JSON, truncated arrays,
     and common LLM formatting mistakes (markdown fences, trailing commas).
  4. Return RawQAPairs for the base class to filter and promote.

JSON parsing robustness:
  Gemini occasionally:
    - Wraps JSON in markdown code fences (```json ... ```)
    - Adds trailing commas in arrays
    - Truncates output mid-JSON when hitting max_output_tokens
    - Returns a single object instead of an array when n_pairs=1

  All of these are handled in _parse_llm_response() without raising.
  Malformed output is logged and returns empty list — the base class
  records it as pairs_rejected, not as a chunk failure.

Retry logic:
  Uses tenacity for exponential backoff on:
    - 429 rate limit errors (Gemini free tier: 15 RPM)
    - 503 service unavailable
    - Network timeouts

  Retry is applied at the individual API call level, not at the
  chunk level. The base class handles chunk-level retries via
  max_retries_per_chunk in GeneratorConfig.

Cost tracking:
  Every API call records input_tokens and output_tokens via
  Gemini's usage_metadata. These flow into ChunkGenerationResult
  and aggregate in GenerationStats for dashboard display.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import DatasetGenConfig, Settings
from src.dataset.base import (
    BaseDatasetGenerator,
    ChunkGenerationError,
    ChunkGenerationResult,
    GeneratorConfig,
    RawQAPair,
)
from src.dataset.schema import GenerationMethod
from src.ingestion.base import Document

# ===========================================================================
# PROMPT TEMPLATES
# ===========================================================================

_GENERATION_PROMPT_TEMPLATE = """You are an expert at creating evaluation datasets for Retrieval-Augmented Generation (RAG) systems.

Your task is to generate exactly {n_pairs} high-quality question-answer pairs from the document chunk provided below.

DOCUMENT CHUNK:
\"\"\"
{chunk_text}
\"\"\"

REQUIREMENTS:
1. Each question must be answerable SOLELY from the content in the chunk above.
2. Each answer must be derived ONLY from the chunk — do not use outside knowledge.
3. Questions must be specific and factual — not vague or opinion-based.
4. Questions must be complete sentences ending with a question mark.
5. Answers must be concise but complete — 1 to 4 sentences maximum.
6. Do NOT generate questions about formatting, structure, or meta-content
   (e.g. "What is the title of this section?" is invalid).
7. Do NOT repeat or rephrase the same question.
8. Each question-answer pair must test a different fact from the chunk.

OUTPUT FORMAT:
Respond with ONLY a valid JSON array. No markdown, no code fences, no explanation.
Return exactly {n_pairs} objects (or fewer if the chunk lacks sufficient content).

Required format:
[
  {{
    "question": "Your specific factual question here?",
    "answer": "The concise answer derived only from the chunk."
  }}
]

If the chunk does not contain enough factual content for even one good question,
return an empty array: []"""


_SINGLE_PAIR_PROMPT_TEMPLATE = """You are an expert at creating evaluation datasets for RAG systems.

Generate exactly 1 high-quality question-answer pair from the document chunk below.

DOCUMENT CHUNK:
\"\"\"
{chunk_text}
\"\"\"

REQUIREMENTS:
- Question must be answerable solely from the chunk above.
- Answer must be derived only from the chunk.
- Question must be specific and factual.
- Answer must be 1-3 sentences.

Respond with ONLY a valid JSON object. No markdown, no code fences.

Required format:
{{
  "question": "Your specific factual question here?",
  "answer": "The concise answer derived only from the chunk."
}}

If the chunk lacks sufficient content, return: {{}}"""

# ===========================================================================
# GEMINI DATASET GENERATOR
# ===========================================================================

class GeminiDatasetGenerator(BaseDatasetGenerator):
    """
    Generates synthetic Q&A evaluation datasets using Google Gemini.

    Inherits the full generation orchestration from BaseDatasetGenerator.
    This class implements only _generate_from_chunk() — the single
    abstract method that defines how one chunk becomes Q&A pairs.

    Constructor:
        generator = GeminiDatasetGenerator.from_settings(settings)

    Or manually:
        generator = GeminiDatasetGenerator(
            config=GeneratorConfig(n_pairs_per_chunk=3, temperature=0.4),
            dataset_gen_config=settings.dataset_gen,
            gemini_api_key=settings.gemini.api_key,
        )
    """

    def __init__(
        self,
        config: GeneratorConfig,
        dataset_gen_config: DatasetGenConfig,
        gemini_api_key: str,
    ) -> None:
        super().__init__(config)
        self._dataset_gen_config = dataset_gen_config
        self._api_key = gemini_api_key
        self._model: Any = None     # google.generativeai.GenerativeModel

        self._initialise_model()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        config: GeneratorConfig | None = None,
    ) -> GeminiDatasetGenerator:
        """
        Standard factory — construct from application Settings.

        Args:
            settings: Application Settings instance from get_settings().
            config:   Optional GeneratorConfig override.
                      Defaults to GeneratorConfig built from
                      settings.dataset_gen if not provided.
        """
        generator_config = config or GeneratorConfig(
            n_pairs_per_chunk=settings.dataset_gen.max_pairs_per_chunk,
            temperature=settings.dataset_gen.temperature,
        )
        return cls(
            config=generator_config,
            dataset_gen_config=settings.dataset_gen,
            gemini_api_key=settings.gemini.api_key,
        )

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def generation_method(self) -> GenerationMethod:
        return GenerationMethod.SYNTHETIC

    @property
    def model_name(self) -> str:
        return self._dataset_gen_config.model

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _initialise_model(self) -> None:
        """
        Configure the Gemini client and instantiate the model.

        Called once at construction time. Fails loudly if the API key
        is invalid or the model name is unrecognised — better to fail
        at startup than mid-generation on chunk 47.
        """
        try:
            # import google.generativeai as genai

            # genai.configure(api_key=self._api_key)

            # generation_config = genai.GenerationConfig(
            #     temperature=self._config.temperature,
            #     max_output_tokens=2048,
            #     response_mime_type="text/plain",
            # )

            # self._model = genai.GenerativeModel(
            #     model_name=self._dataset_gen_config.model,
            #     generation_config=generation_config,
            # )
            # NEW
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self._api_key)
            self._model = client
            self._gen_config = types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=2048,
            )

            logger.info(
                f"GeminiDatasetGenerator initialised. "
                f"model={self._dataset_gen_config.model}, "
                f"temperature={self._config.temperature}"
            )

        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: poetry add google-generativeai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"GeminiDatasetGenerator: Failed to initialise Gemini model "
                f"'{self._dataset_gen_config.model}': {exc}. "
                f"Check GEMINI_API_KEY in .env."
            ) from exc

    # ------------------------------------------------------------------
    # Core abstract method implementation
    # ------------------------------------------------------------------

    def _generate_from_chunk(
        self,
        chunk: Document,
        n_pairs: int,
        dataset_id: str,
    ) -> tuple[list[RawQAPair], ChunkGenerationResult]:
        """
        Generate n_pairs Q&A pairs from a single document chunk.

        Implements the BaseDatasetGenerator contract:
          - Never raises — all exceptions caught and returned as failed
          - Records latency_ms, input_tokens, output_tokens
          - Returns (raw_pairs, chunk_result)

        Retry logic is applied at the _call_gemini_api() level via
        tenacity. If all retries are exhausted, the exception is caught
        here and returned as chunk_result.status = "failed".
        """
        chunk_result = ChunkGenerationResult(
            chunk_index=chunk.chunk_index or 0,
            source_file=chunk.metadata.source_file,
            source_page=chunk.metadata.page_number,
            status="failed",
        )

        start_ms = time.monotonic() * 1000

        # Check minimum chunk length
        chunk_text = chunk.page_content.strip()
        if len(chunk_text) < self._config.min_chunk_length:
            chunk_result.status = "skipped"
            chunk_result.error_message = (
                f"Chunk too short: {len(chunk_text)} chars "
                f"(min {self._config.min_chunk_length})"
            )
            chunk_result.latency_ms = time.monotonic() * 1000 - start_ms
            return [], chunk_result

        # Build prompt
        prompt = self._build_prompt(
            chunk_text=chunk_text,
            n_pairs=n_pairs,
        )

        # Call Gemini with retries
        raw_response: str | None = None
        input_tokens = 0
        output_tokens = 0
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries_per_chunk + 1):
            try:
                raw_response, input_tokens, output_tokens = (
                    self._call_gemini_api(prompt)
                )
                chunk_result.retry_count = attempt
                break

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"GeminiDatasetGenerator: Chunk "
                    f"'{chunk.metadata.source_file}' "
                    f"attempt {attempt + 1}/"
                    f"{self._config.max_retries_per_chunk + 1} "
                    f"failed: {exc}"
                )
                if attempt < self._config.max_retries_per_chunk:
                    # Brief pause between manual retries
                    time.sleep(2.0 ** attempt)
                    continue
                break

        chunk_result.input_tokens = input_tokens
        chunk_result.output_tokens = output_tokens
        chunk_result.latency_ms = time.monotonic() * 1000 - start_ms

        if raw_response is None:
            chunk_result.status = "failed"
            chunk_result.error_message = (
                f"All {self._config.max_retries_per_chunk + 1} attempts "
                f"failed. Last error: {last_error}"
            )
            logger.error(
                f"GeminiDatasetGenerator: Failed to generate from chunk "
                f"'{chunk.metadata.source_file}' "
                f"p={chunk.metadata.page_number} "
                f"after all retries: {last_error}"
            )
            return [], chunk_result

        # Parse response
        raw_pairs = self._parse_llm_response(
            raw_response=raw_response,
            n_pairs_requested=n_pairs,
        )

        pairs_generated = len(raw_pairs)
        chunk_result.pairs_generated = pairs_generated
        chunk_result.status = "success"

        logger.debug(
            f"GeminiDatasetGenerator: chunk "
            f"'{chunk.metadata.source_file}' "
            f"p={chunk.metadata.page_number} → "
            f"{pairs_generated} raw pairs, "
            f"tokens={input_tokens}+{output_tokens}, "
            f"latency={chunk_result.latency_ms:.0f}ms"
        )

        return raw_pairs, chunk_result

    # ------------------------------------------------------------------
    # Gemini API call
    # ------------------------------------------------------------------

    # @retry(
    #     retry=retry_if_exception_type((Exception,)),
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=2, min=2, max=30),
    #     reraise=True,
    # )
    # def _call_gemini_api(
    #     self,
    #     prompt: str,
    # ) -> tuple[str, int, int]:
    #     """
    #     Make a single Gemini API call with tenacity retry.

    #     Returns (response_text, input_tokens, output_tokens).

    #     The @retry decorator handles:
    #       - 429 rate limit errors (waits exponentially: 2s, 4s, 8s...)
    #       - 503 service unavailable
    #       - Network timeouts

    #     Raises on all other errors (reraise=True).
    #     """
    #     if self._model is None:
    #         raise RuntimeError(
    #             "Gemini model not initialised. "
    #             "Call _initialise_model() first."
    #         )

    #     response = self._model.generate_content(prompt)

    #     # Extract token counts from usage metadata
    #     input_tokens = 0
    #     output_tokens = 0
    #     if hasattr(response, "usage_metadata") and response.usage_metadata:
    #         input_tokens = getattr(
    #             response.usage_metadata, "prompt_token_count", 0
    #         ) or 0
    #         output_tokens = getattr(
    #             response.usage_metadata, "candidates_token_count", 0
    #         ) or 0

    #     # Extract text content
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
    #         raise ChunkGenerationError(
    #             chunk_index=0,
    #             source_file="unknown",
    #             reason=(
    #                 "Gemini returned empty response. "
    #                 "This may indicate a safety filter block or "
    #                 "a content policy rejection."
    #             ),
    #         )

    #     return text, input_tokens, output_tokens
    
    @retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
    def _call_gemini_api(self, prompt: str) -> tuple[str, int, int]:
        """
        Make a single Gemini API call with tenacity retry.

        Returns (response_text, input_tokens, output_tokens).
        The @retry decorator handles:
          - 429 rate limit errors (waits exponentially: 2s, 4s, 8s...)
          - 503 service unavailable
          - Network timeouts
        Raises on all other errors (reraise=True).
        """
        response = self._model.models.generate_content(
            model=self._dataset_gen_config.model,
            contents=prompt,
            config=self._gen_config,
        )
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
        text = response.text or ""
        if not text.strip():
            raise ChunkGenerationError(
                chunk_index=0,
                source_file="unknown",
                reason="Gemini returned empty response.",
            )
        return text, input_tokens, output_tokens
    

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        chunk_text: str,
        n_pairs: int,
    ) -> str:
        """
        Build the generation prompt for a given chunk and pair count.

        Uses a single-object prompt for n_pairs=1 to reduce the chance
        of Gemini wrapping the result in an unnecessary array — single
        objects are more reliably parsed from short outputs.
        """
        # Truncate very long chunks to avoid exceeding prompt limits
        # Gemini 1.5 Flash: 1M token context, but prompts over ~20K
        # chars slow generation and rarely improve pair quality.
        max_chunk_chars = 8000
        if len(chunk_text) > max_chunk_chars:
            chunk_text = chunk_text[:max_chunk_chars] + "\n[... truncated]"
            logger.debug(
                f"GeminiDatasetGenerator: Chunk truncated to "
                f"{max_chunk_chars} chars for prompt."
            )

        if n_pairs == 1:
            return _SINGLE_PAIR_PROMPT_TEMPLATE.format(
                chunk_text=chunk_text,
            )

        return _GENERATION_PROMPT_TEMPLATE.format(
            n_pairs=n_pairs,
            chunk_text=chunk_text,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        raw_response: str,
        n_pairs_requested: int,
    ) -> list[RawQAPair]:
        """
        Parse Gemini's text response into a list of RawQAPairs.

        Handles all known Gemini formatting quirks:
          1. Markdown code fences: ```json [...] ```
          2. Single object instead of array (when n_pairs=1)
          3. Trailing commas in JSON arrays/objects
          4. Truncated JSON (mid-array when hitting token limits)
          5. Extra text before/after the JSON
          6. Empty response or response with only whitespace
        """
        if not raw_response or not raw_response.strip():
            logger.warning(
                "GeminiDatasetGenerator: Empty response from Gemini."
            )
            return []

        # Step 1: Extract JSON from response
        json_text = self._extract_json(raw_response)
        if not json_text:
            logger.warning(
                f"GeminiDatasetGenerator: Could not extract JSON from "
                f"response: '{raw_response[:200]}'"
            )
            return []

        # Step 2: Fix common JSON issues
        json_text = self._fix_json(json_text)

        # Step 3: Parse
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"GeminiDatasetGenerator: JSON parse failed after fixes: "
                f"{exc}. "
                f"Attempting partial recovery..."
            )
            parsed = self._attempt_partial_recovery(json_text)
            if parsed is None:
                logger.warning(
                    f"GeminiDatasetGenerator: Partial recovery failed. "
                    f"Discarding response."
                )
                return []

        # Step 4: Normalise to list
        if isinstance(parsed, dict):
            # Single object response — wrap in list
            if parsed:   # Non-empty dict
                parsed = [parsed]
            else:
                return []   # Empty dict = no content

        if not isinstance(parsed, list):
            logger.warning(
                f"GeminiDatasetGenerator: Unexpected JSON type: "
                f"{type(parsed).__name__}. Expected list or dict."
            )
            return []

        # Step 5: Convert to RawQAPairs
        raw_pairs: list[RawQAPair] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.debug(
                    f"GeminiDatasetGenerator: Item {i} is not a dict "
                    f"({type(item).__name__}). Skipping."
                )
                continue

            # Handle common key variations from LLM output
            question = (
                item.get("question")
                or item.get("Question")
                or item.get("q")
                or item.get("Q")
            )
            answer = (
                item.get("answer")
                or item.get("Answer")
                or item.get("a")
                or item.get("A")
            )

            raw_pair = RawQAPair(
                question=str(question).strip() if question else None,
                answer=str(answer).strip() if answer else None,
            )
            raw_pairs.append(raw_pair)

        # Cap at requested count — sometimes Gemini returns more
        if len(raw_pairs) > n_pairs_requested:
            logger.debug(
                f"GeminiDatasetGenerator: Gemini returned "
                f"{len(raw_pairs)} pairs but {n_pairs_requested} "
                f"were requested. Truncating."
            )
            raw_pairs = raw_pairs[:n_pairs_requested]

        return raw_pairs

    def _extract_json(self, text: str) -> str | None:
        """
        Extract JSON content from raw LLM response text.

        Handles:
          - Clean JSON (no wrapping)
          - Markdown fences: ```json ... ``` or ``` ... ```
          - JSON preceded/followed by explanation text

        Returns the extracted JSON string, or None if not found.
        """
        text = text.strip()

        # Pattern 1: Markdown code fence with json tag
        fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?```",
            re.DOTALL,
        )
        match = fence_pattern.search(text)
        if match:
            return match.group(1).strip()

        # Pattern 2: Starts with [ or { (clean JSON)
        if text.startswith("[") or text.startswith("{"):
            return text

        # Pattern 3: Find first [ or { in the text
        array_start = text.find("[")
        object_start = text.find("{")

        if array_start == -1 and object_start == -1:
            return None

        if array_start == -1:
            start = object_start
        elif object_start == -1:
            start = array_start
        else:
            start = min(array_start, object_start)

        # Find matching closing bracket
        opening = text[start]
        closing = "]" if opening == "[" else "}"

        # Find the last occurrence of the closing bracket
        end = text.rfind(closing)
        if end <= start:
            return None

        return text[start: end + 1]

    def _fix_json(self, json_text: str) -> str:
        """
        Apply heuristic fixes to malformed JSON from LLM output.

        Fixes applied:
          1. Trailing commas before ] or } (common LLM mistake)
          2. Single quotes used instead of double quotes (less common)
          3. Missing closing brackets (truncated output)
        """
        # Fix 1: Trailing commas — ,] or ,}
        json_text = re.sub(r",\s*]", "]", json_text)
        json_text = re.sub(r",\s*}", "}", json_text)

        # Fix 2: Attempt to close truncated arrays
        # If text ends with partial JSON, try to close it
        stripped = json_text.strip()
        if stripped.startswith("[") and not stripped.endswith("]"):
            # Count open/close braces to see if we need to close objects
            open_braces = stripped.count("{")
            close_braces = stripped.count("}")
            unclosed = open_braces - close_braces

            # Close any unclosed objects (truncated last item)
            if unclosed > 0:
                # Remove the partial last item (likely truncated)
                last_complete = stripped.rfind("},")
                if last_complete > 0:
                    stripped = stripped[: last_complete + 1] + "]"
                else:
                    # No complete items — return empty array
                    stripped = "[]"
            else:
                stripped = stripped + "]"

            json_text = stripped

        return json_text

    def _attempt_partial_recovery(
        self,
        json_text: str,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        """
        Last-resort recovery: extract whatever complete objects we can.

        If the full JSON string fails to parse, try to extract
        complete {"question": ..., "answer": ...} objects one by one
        using regex.

        Returns parsed objects as list, single dict, or None.
        """
        # Try to find complete question-answer objects
        object_pattern = re.compile(
            r'\{\s*"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
            r'\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
            re.DOTALL,
        )
        matches = object_pattern.findall(json_text)

        if matches:
            recovered = [
                {"question": q, "answer": a}
                for q, a in matches
            ]
            logger.info(
                f"GeminiDatasetGenerator: Partial recovery succeeded. "
                f"Recovered {len(recovered)} objects."
            )
            return recovered

        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def estimate_generation_cost(
        self,
        n_chunks: int,
        avg_chunk_chars: int = 1000,
    ) -> dict[str, float]:
        """
        Estimate the cost of generating a dataset before running.

        Used by the Streamlit UI to show a cost preview before
        the user clicks "Generate Dataset".

        Returns dict with keys:
          estimated_input_tokens  — total input tokens
          estimated_output_tokens — total output tokens
          estimated_cost_usd      — total estimated cost
          estimated_api_calls     — total Gemini API calls
        """
        from config import get_model_registry

        # Approximate: prompt template + chunk text
        avg_prompt_chars = len(_GENERATION_PROMPT_TEMPLATE) + avg_chunk_chars
        avg_prompt_tokens = int(avg_prompt_chars / 4)

        # Each pair generates ~50 tokens of JSON output
        avg_output_tokens = self._config.n_pairs_per_chunk * 50

        total_api_calls = n_chunks
        total_input_tokens = avg_prompt_tokens * total_api_calls
        total_output_tokens = avg_output_tokens * total_api_calls

        try:
            registry = get_model_registry()
            model = registry.get_model(self._dataset_gen_config.model)
            cost = model.estimate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )
        except KeyError:
            cost = 0.0

        return {
            "estimated_input_tokens":  total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_cost_usd":      cost,
            "estimated_api_calls":     total_api_calls,
        }

    def __repr__(self) -> str:
        return (
            f"GeminiDatasetGenerator("
            f"model='{self.model_name}', "
            f"n_pairs_per_chunk={self._config.n_pairs_per_chunk}, "
            f"temperature={self._config.temperature}, "
            f"max_pairs_total={self._config.max_pairs_total})"
        )

__all__ = ["GeminiDatasetGenerator"]