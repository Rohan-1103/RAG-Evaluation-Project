"""
src/rag/clients.py

Multi-provider LLM abstraction layer — the single chokepoint every
text-generation call in this codebase flows through.

Before this file existed, three separate call sites (RAGPipeline,
BaseEvaluator, GeminiDatasetGenerator) each constructed a
google.genai.Client directly and embedded Gemini-specific retry and
parsing logic inline. This file collapses all three into one contract:

    result = get_llm_client(model_id).generate(model_id, prompt, ...)

Adding a new provider (e.g. Anthropic, Mistral) requires:
  1. One new BaseLLMClient subclass (~50 lines)
  2. One new provider block in config/models.yaml
  3. One elif branch in get_llm_client()
  Zero changes to RAGPipeline, BaseEvaluator, or GeminiDatasetGenerator.

Architecture:
  BaseLLMClient       — abstract interface, one method: generate()
  GeminiClient        — Google Gemini via google-genai SDK
  GroqClient          — Groq via openai SDK (OpenAI-compatible endpoint)
  OpenRouterClient    — OpenRouter via openai SDK (OpenAI-compatible)
  CircuitBreaker      — per-provider failure detector, fast-fails on outage
  get_llm_client()    — factory: model_id → cached provider client
  LLMGenerationResult — uniform output shape across all providers
  LLMClientError      — uniform exception type across all providers

Design decisions:
  - Clients are PROVIDER-scoped, not MODEL-scoped. One GeminiClient
    handles every Gemini model variant. model_id travels with each
    call, not with client construction.
  - Retry logic (tenacity) lives INSIDE each client's _generate_internal()
    since Gemini's and Groq's transient-failure exception shapes differ.
    Call sites never see a retry decorator.
  - Circuit breakers are MODULE-LEVEL (one per provider) so all client
    instances share state — a Groq outage detected by RAGPipeline's
    call is immediately visible to BaseEvaluator's next judge call,
    not just the next RAGPipeline call.
  - model_name (from registry) vs model_id (our internal key) are
    resolved at call time inside generate() so callers always use our
    stable internal IDs even when a provider's API model string changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from threading import Lock

from loguru import logger
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ===========================================================================
# RESULT TYPE
# ===========================================================================


class LLMGenerationResult(BaseModel):
    """
    Uniform output shape returned by every BaseLLMClient.generate() call,
    regardless of provider.

    Callers (RAGPipeline, BaseEvaluator, GeminiDatasetGenerator) depend
    only on this type — they never branch on provider to extract text,
    token counts, or model metadata. This is what makes a comparison grid
    mixing Gemini and Groq in one request work transparently.

    Frozen — generation results are immutable after the API responds.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    """Cleaned response text with leading/trailing whitespace stripped."""

    input_tokens: int
    """Prompt token count reported by the provider. 0 if unavailable."""

    output_tokens: int
    """Completion token count reported by the provider. 0 if unavailable."""

    model_id: str
    """Our internal registry model ID (e.g. 'gpt-oss-120b'), not the API string."""

    provider: str
    """Provider name: 'google', 'groq', or 'openrouter'."""


# ===========================================================================
# CIRCUIT BREAKER
# ===========================================================================


class CircuitState(str, Enum):
    """
    Three-state machine for a per-provider circuit breaker.

    Transitions:
        CLOSED    → OPEN:      after FAILURE_THRESHOLD consecutive failures
        OPEN      → HALF_OPEN: after RECOVERY_TIMEOUT_SECONDS elapses
        HALF_OPEN → CLOSED:    on first successful generate() call
        HALF_OPEN → OPEN:      on first failed generate() call (still down)

    Terminal states: none — the breaker always eventually retries.
    """

    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Provider down — fast-fail immediately
    HALF_OPEN = "half_open" # Probing recovery — one request let through


class CircuitBreaker:
    """
    Per-provider circuit breaker that prevents repeated slow-failing
    retries against a known-down LLM provider.

    Without this: a Groq outage during a 10-pair evaluation causes
    10 × 3 tenacity retries × 30s max backoff = up to 15 minutes of
    stalled execution before the run fails with an error that was
    knowable after the first 5 failures.

    With this: after FAILURE_THRESHOLD consecutive failures the breaker
    opens and every subsequent call fast-fails with a clear error message
    until RECOVERY_TIMEOUT_SECONDS have elapsed and a probe succeeds.

    Thread-safe via threading.Lock — safe for FastAPI's ThreadPoolExecutor-
    wrapped synchronous generate() calls and for multiple uvicorn workers
    sharing the module-level _circuit_breakers dict via the same process
    (though per-process state means each worker has its own breaker state
    in a multi-worker deployment).

    Usage:
        cb = _get_circuit_breaker("groq")
        if cb.is_open():
            raise LLMClientError(...)
        try:
            result = self._generate_internal(...)
            cb.record_success()
            return result
        except LLMClientError:
            cb.record_failure()
            raise
    """

    FAILURE_THRESHOLD: int = 5
    """Consecutive failures before opening the circuit."""

    RECOVERY_TIMEOUT_SECONDS: int = 60
    """Seconds the circuit stays OPEN before probing for recovery."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """
        Current circuit state, with automatic OPEN → HALF_OPEN
        transition when the recovery timeout has elapsed.

        Thread-safe — acquires the lock before reading or mutating state.
        """
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time
                > self.RECOVERY_TIMEOUT_SECONDS
            ):
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    f"CircuitBreaker [{self.provider}]: "
                    f"OPEN → HALF_OPEN (probing recovery after "
                    f"{self.RECOVERY_TIMEOUT_SECONDS}s)"
                )
            return self._state

    def is_open(self) -> bool:
        """True if the circuit should fast-fail the next request."""
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        """
        Record a successful generate() call.

        Resets failure count and transitions HALF_OPEN → CLOSED.
        No-op when already CLOSED (normal operation).
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"CircuitBreaker [{self.provider}]: "
                    f"HALF_OPEN → CLOSED (provider recovered ✓)"
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed generate() call.

        Increments failure count and transitions CLOSED → OPEN once
        FAILURE_THRESHOLD consecutive failures are recorded.
        In HALF_OPEN state, a single failure immediately re-opens.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"CircuitBreaker [{self.provider}]: "
                    f"HALF_OPEN → OPEN (probe failed, provider still down)"
                )
                self._state = CircuitState.OPEN

            elif self._failure_count >= self.FAILURE_THRESHOLD:
                if self._state != CircuitState.OPEN:
                    logger.error(
                        f"CircuitBreaker [{self.provider}]: "
                        f"CLOSED → OPEN after {self._failure_count} "
                        f"consecutive failures. Fast-failing all requests "
                        f"for {self.RECOVERY_TIMEOUT_SECONDS}s. "
                        f"Provider may be experiencing an outage."
                    )
                self._state = CircuitState.OPEN

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(provider='{self.provider}', "
            f"state={self._state.value}, "
            f"failures={self._failure_count})"
        )


# Module-level registry — one CircuitBreaker per provider, shared
# across all client instances for that provider.
_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_circuit_breaker(provider: str) -> CircuitBreaker:
    """Return (creating if necessary) the circuit breaker for a provider."""
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker(provider)
    return _circuit_breakers[provider]


def get_circuit_breaker_states() -> dict[str, str]:
    """
    Return the current state of every active circuit breaker.

    Used by the /health endpoint to expose provider availability:
        {"google": "closed", "groq": "open", "openrouter": "closed"}
    "closed" = normal. "open" = provider down. "half_open" = probing.
    """
    return {
        provider: cb.state.value
        for provider, cb in _circuit_breakers.items()
    }


# ===========================================================================
# ABSTRACT BASE
# ===========================================================================


class BaseLLMClient(ABC):
    """
    Abstract interface that every LLM provider client must implement.

    The contract: one method, generate(), returns LLMGenerationResult.
    Everything provider-specific (SDK imports, request shape, response
    parsing, retry policy, circuit breaker integration) lives inside
    the concrete subclass. Callers depend only on this interface.

    Subclassing checklist:
      1. Implement generate() and _generate_internal()
      2. Implement provider_name property
      3. Register in get_llm_client() factory
      4. Add provider block to config/models.yaml
      5. Add XxxConfig to config/settings.py
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Short provider identifier: 'google', 'groq', 'openrouter'.
        Must match the provider key in config/models.yaml's providers block.
        """
        ...

    @abstractmethod
    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate text from a single prompt.

        Args:
            model_id:           Our internal registry model ID. The
                                concrete implementation resolves this to
                                the provider's actual API model string via
                                get_model_registry().get_model(model_id).
            prompt:             User-facing prompt text. For judge calls
                                this is the full structured rubric prompt;
                                for RAG calls this is the context + question.
            temperature:        Generation temperature. 0.0 = deterministic
                                (used for evaluation judge calls). Higher
                                values for dataset generation (0.4 default).
            max_output_tokens:  Hard ceiling on completion length. Critical
                                for reasoning models (e.g. gpt-oss-120b)
                                which consume reasoning tokens before content
                                tokens — too low a value yields empty content.
            system_instruction: Optional system prompt prepended before the
                                user message. Gemini sends this via the SDK's
                                system_instruction field; Groq/OpenRouter
                                prepend a {"role": "system"} message.

        Returns:
            LLMGenerationResult with text, token counts, model_id, provider.

        Raises:
            LLMClientError: on provider API failure, after all retries
                            are exhausted. Never raises the provider's own
                            SDK exception types — callers only ever see
                            LLMClientError regardless of which provider
                            is underneath.

        Contract:
            - Never raises after exhausting retries — wraps SDK exceptions
              in LLMClientError before propagating.
            - Circuit breaker integration: fast-fails on OPEN state.
            - Retry: handled inside _generate_internal() via tenacity.
        """
        ...


# ===========================================================================
# GEMINI CLIENT
# ===========================================================================


class GeminiClient(BaseLLMClient):
    """
    Google Gemini client via the google-genai SDK.

    Extracted from the three call sites that previously duplicated this
    logic (RAGPipeline._call_gemini, BaseEvaluator._call_judge_api,
    GeminiDatasetGenerator._call_gemini_api). Now implemented exactly
    once, behind the BaseLLMClient interface.

    Rate limits (free tier):
        15 RPM, 1M tokens/day (as of July 2026 — check AI Studio for
        current limits). The evaluation engine's
        inter_question_delay_seconds=4.0 and max_concurrent_eval_calls=2
        in eval.yaml are tuned to stay within this ceiling.

    Reasoning models:
        Some Gemini models produce internal reasoning tokens before
        content tokens. max_output_tokens must be high enough to cover
        both reasoning and visible output — 1024 minimum recommended.

    system_instruction:
        Mapped to GenerateContentConfig(system_instruction=...) — a
        model-level field in the Gemini SDK, distinct from the message
        content, which improves adherence for judge prompts.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialise the Gemini client.

        Args:
            api_key: Google Gemini API key from .env GEMINI_API_KEY.
                     Get a free key at https://aistudio.google.com/app/apikey
                     with no billing project attached for free-tier access.

        Raises:
            ImportError: if google-genai is not installed.
            LLMClientError: if the client cannot be constructed (bad key
                            format, SDK initialisation failure).
        """
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        logger.debug("GeminiClient initialised.")

    @property
    def provider_name(self) -> str:
        return "google"

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate text via Gemini with circuit breaker protection.

        Circuit breaker opens after 5 consecutive failures and fast-fails
        for 60s before probing recovery — prevents stalling an entire
        evaluation run against a known-down Gemini endpoint.
        """
        cb = _get_circuit_breaker("google")

        if cb.is_open():
            raise LLMClientError(
                provider="google",
                model_id=model_id,
                reason=(
                    f"Circuit breaker OPEN — Google Gemini has failed "
                    f"{CircuitBreaker.FAILURE_THRESHOLD}+ consecutive times. "
                    f"Retrying automatically in up to "
                    f"{CircuitBreaker.RECOVERY_TIMEOUT_SECONDS}s. "
                    f"Check https://status.google.com for outage info."
                ),
            )

        try:
            result = self._generate_internal(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
            cb.record_success()
            return result
        except LLMClientError:
            cb.record_failure()
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _generate_internal(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> LLMGenerationResult:
        """
        Single Gemini API call with tenacity retry.

        Retries up to 3× with exponential backoff (2s, 4s, 8s) on any
        exception — handles 429 rate limits, 503 unavailable, network
        timeouts. reraise=True means the final failure propagates as the
        original exception, caught by generate() and converted to
        LLMClientError + circuit breaker failure recording.

        model_name resolution:
            model_id is our internal key (e.g. 'gemini-3.1-flash-lite').
            The registry's model_name field holds the exact API string.
            For Gemini these currently match, but the resolution step
            future-proofs against Gemini renaming their model strings
            (which they have done at least twice during this project's
            development).
        """
        from google.genai import types

        # Resolve API model string from registry
        from config import get_model_registry
        try:
            registry_entry = get_model_registry().get_model(model_id)
            api_model_name = (
                getattr(registry_entry, "model_name", model_id) or model_id
            )
        except KeyError:
            api_model_name = model_id

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        try:
            response = self._client.models.generate_content(
                model=api_model_name,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise LLMClientError(
                provider="google",
                model_id=model_id,
                reason=str(exc),
                original_exception=exc,
            ) from exc

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = (
                response.usage_metadata.prompt_token_count or 0
            )
            output_tokens = (
                response.usage_metadata.candidates_token_count or 0
            )

        text = response.text or ""
        if not text.strip():
            raise LLMClientError(
                provider="google",
                model_id=model_id,
                reason=(
                    "Empty response from Gemini — possible safety filter "
                    "block or max_output_tokens too low for a reasoning model."
                ),
            )

        return LLMGenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            provider="google",
        )


# ===========================================================================
# GROQ CLIENT
# ===========================================================================


class GroqClient(BaseLLMClient):
    """
    Groq client via the openai SDK pointed at Groq's OpenAI-compatible
    chat completions endpoint.

    Groq's free tier (30 RPM, 1K RPD) offers double Gemini's request
    rate, making it a better fit for judge evaluation where 4 concurrent
    metric calls per question can saturate Gemini's 15 RPM ceiling.

    system_instruction:
        Mapped to a prepended {"role": "system"} message since Groq's
        chat completions endpoint has no model-level system field.

    Reasoning models (e.g. openai/gpt-oss-120b):
        These models spend internal reasoning tokens before producing
        visible content. With max_tokens=10 the budget is exhausted during
        reasoning, leaving content empty. Always use max_output_tokens≥200
        for reasoning models — production defaults (1024 for judge calls,
        2048 for generation) are safe.

    model_name resolution:
        Groq uses namespaced API strings (e.g. "openai/gpt-oss-120b")
        that differ from our internal registry IDs ("gpt-oss-120b").
        The model_name field in models.yaml carries the full API string;
        this client resolves it at call time.
    """

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str) -> None:
        """
        Initialise the Groq client.

        Args:
            api_key: Groq API key from .env GROQ_API_KEY.
                     Get a free key at https://console.groq.com/keys
                     (no credit card, 30 RPM free tier).

        Raises:
            ImportError: if openai SDK is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=self._BASE_URL)
        logger.debug("GroqClient initialised.")

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate text via Groq with circuit breaker protection.

        Identical circuit breaker pattern to GeminiClient — opens after
        5 consecutive failures, fast-fails for 60s, then probes recovery.
        Independent from GeminiClient's breaker: a Groq outage does not
        affect the Gemini breaker, allowing mixed-provider comparisons
        to continue with the available provider while the other recovers.
        """
        cb = _get_circuit_breaker("groq")

        if cb.is_open():
            raise LLMClientError(
                provider="groq",
                model_id=model_id,
                reason=(
                    f"Circuit breaker OPEN — Groq has failed "
                    f"{CircuitBreaker.FAILURE_THRESHOLD}+ consecutive times. "
                    f"Retrying automatically in up to "
                    f"{CircuitBreaker.RECOVERY_TIMEOUT_SECONDS}s. "
                    f"Check https://groqstatus.com for outage info."
                ),
            )

        try:
            result = self._generate_internal(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
            cb.record_success()
            return result
        except LLMClientError:
            cb.record_failure()
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _generate_internal(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> LLMGenerationResult:
        """
        Single Groq API call with tenacity retry.

        Constructs the messages list with an optional system message
        prepended, then calls chat.completions.create. Token counts come
        from response.usage.prompt_tokens / completion_tokens (OpenAI
        standard fields, reliably populated by Groq).
        """
        from config import get_model_registry
        try:
            registry_entry = get_model_registry().get_model(model_id)
            api_model_name = (
                getattr(registry_entry, "model_name", model_id) or model_id
            )
        except KeyError:
            api_model_name = model_id

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=api_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise LLMClientError(
                provider="groq",
                model_id=model_id,
                reason=str(exc),
                original_exception=exc,
            ) from exc

        choice = response.choices[0] if response.choices else None
        text = (
            choice.message.content
            if choice and choice.message
            else ""
        )

        if not text or not text.strip():
            # Diagnostic hint for reasoning models consuming all tokens
            usage = response.usage
            hint = ""
            if (
                usage
                and hasattr(usage, "completion_tokens_details")
                and usage.completion_tokens_details
                and getattr(
                    usage.completion_tokens_details,
                    "reasoning_tokens", 0
                )
            ):
                reasoning_tokens = (
                    usage.completion_tokens_details.reasoning_tokens
                )
                hint = (
                    f" Model used {reasoning_tokens} reasoning tokens — "
                    f"increase max_output_tokens to leave room for "
                    f"visible content after internal reasoning."
                )
            raise LLMClientError(
                provider="groq",
                model_id=model_id,
                reason=f"Empty response from Groq.{hint}",
            )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMGenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            provider="groq",
        )


# ===========================================================================
# OPENROUTER CLIENT
# ===========================================================================


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter client via the openai SDK pointed at OpenRouter's
    OpenAI-compatible endpoint.

    OpenRouter provides access to 300+ models from 60+ providers through
    one API key and one integration. Free models (`:free` suffix in
    model_name) are permanently free — no credits consumed.

    Two required headers distinguish OpenRouter from Groq despite sharing
    the openai SDK:
        HTTP-Referer: your app's URL (for OpenRouter dashboard tracking)
        X-Title:      your app's name (shown in OpenRouter usage logs)
    These are required by OpenRouter's API terms and also help with
    debugging in the OpenRouter dashboard when investigating rate limit
    or routing issues.

    Free model latency:
        OpenRouter free models run on shared infrastructure and are
        subject to higher latency during peak hours (typically 2-15s
        per call vs Groq's 0.5-2s). This is expected behaviour, not
        a code issue — free capacity is lower priority than paid.
        For evaluation workloads, set inter_question_delay_seconds≥4
        in eval.yaml to avoid queuing too many slow calls simultaneously.

    Daily limits:
        Without purchasing credits: 50 requests/day.
        After purchasing ≥$10 credits: 1,000 requests/day.
        Check current limits at https://openrouter.ai/docs#limits.

    Reasoning models:
        Same max_output_tokens consideration as Groq — reasoning models
        consume internal tokens before producing visible content.
    """

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        site_url: str = "http://localhost:8501",
        app_name: str = "RAG Eval Bench",
    ) -> None:
        """
        Initialise the OpenRouter client.

        Args:
            api_key:  OpenRouter API key from .env OPENROUTER_API_KEY.
                      Get a free key at https://openrouter.ai/settings/keys
                      (no credit card required for free-tier models).
            site_url: Your app's URL, sent as HTTP-Referer header.
                      Use your Streamlit Cloud URL in production.
            app_name: Your app's display name, sent as X-Title header.

        Raises:
            ImportError: if openai SDK is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )
        logger.debug(
            f"OpenRouterClient initialised. "
            f"site_url='{site_url}', app_name='{app_name}'"
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate text via OpenRouter with circuit breaker protection.

        OpenRouter's circuit breaker is independent from Gemini's and
        Groq's — an OpenRouter outage does not affect the other two
        providers, allowing a comparison run to complete with partial
        results from available providers.
        """
        cb = _get_circuit_breaker("openrouter")

        if cb.is_open():
            raise LLMClientError(
                provider="openrouter",
                model_id=model_id,
                reason=(
                    f"Circuit breaker OPEN — OpenRouter has failed "
                    f"{CircuitBreaker.FAILURE_THRESHOLD}+ consecutive times. "
                    f"Retrying automatically in up to "
                    f"{CircuitBreaker.RECOVERY_TIMEOUT_SECONDS}s. "
                    f"Check https://status.openrouter.ai for outage info."
                ),
            )

        try:
            result = self._generate_internal(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
            cb.record_success()
            return result
        except LLMClientError:
            cb.record_failure()
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _generate_internal(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> LLMGenerationResult:
        """
        Single OpenRouter API call with tenacity retry.

        Structurally identical to GroqClient._generate_internal() since
        both use the openai SDK against an OpenAI-compatible endpoint.
        The key difference is the :free suffix in api_model_name — free
        OpenRouter models must have this suffix or the request is routed
        to a paid variant and credits are deducted.
        """
        from config import get_model_registry
        try:
            registry_entry = get_model_registry().get_model(model_id)
            api_model_name = (
                getattr(registry_entry, "model_name", model_id) or model_id
            )
        except KeyError:
            api_model_name = model_id

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=api_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise LLMClientError(
                provider="openrouter",
                model_id=model_id,
                reason=str(exc),
                original_exception=exc,
            ) from exc

        choice = response.choices[0] if response.choices else None
        text = (
            choice.message.content
            if choice and choice.message
            else ""
        )

        if not text or not text.strip():
            usage = response.usage
            hint = ""
            if (
                usage
                and hasattr(usage, "completion_tokens_details")
                and usage.completion_tokens_details
                and getattr(
                    usage.completion_tokens_details,
                    "reasoning_tokens", 0
                )
            ):
                reasoning_tokens = (
                    usage.completion_tokens_details.reasoning_tokens
                )
                hint = (
                    f" Model used {reasoning_tokens} reasoning tokens — "
                    f"increase max_output_tokens."
                )
            raise LLMClientError(
                provider="openrouter",
                model_id=model_id,
                reason=f"Empty response from OpenRouter.{hint}",
            )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMGenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            provider="openrouter",
        )


# ===========================================================================
# CLIENT FACTORY
# ===========================================================================

# Process-level cache — one client instance per provider.
# Constructed lazily on first get_llm_client() call for that provider,
# then reused for the lifetime of the process. Not reconstructed per model
# (model_id travels with each generate() call), not per request (would
# re-validate API credentials on every single API call).
_client_cache: dict[str, BaseLLMClient] = {}


def get_llm_client(model_id: str) -> BaseLLMClient:
    """
    Resolve model_id → provider → cached BaseLLMClient instance.

    This is the ONLY entry point callers should use. The three refactored
    call sites (RAGPipeline._call_llm, BaseEvaluator._call_judge_api,
    GeminiDatasetGenerator._call_gemini_api) all call this function and
    depend only on the returned BaseLLMClient, never on a concrete class.

    Resolution steps:
        1. Look up model_id in config/models.yaml registry
        2. Read .provider field ('google', 'groq', 'openrouter')
        3. Return cached client for that provider, or construct one
           using the matching API key from Settings

    Cache semantics:
        One entry per provider string, not per model_id. 100 calls to
        get_llm_client() with 5 different Groq model IDs hit the cache
        after the first Groq call — they all use the same GroqClient.

    Args:
        model_id: Internal registry key (e.g. 'gpt-oss-120b', not
                  'openai/gpt-oss-120b'). Must be enabled in models.yaml.

    Returns:
        BaseLLMClient instance for the model's provider.

    Raises:
        KeyError:       if model_id is not in the registry or is disabled.
        LLMClientError: if the provider's API key is not configured.
    """
    from config import get_model_registry
    from config.settings import get_settings

    registry = get_model_registry()
    model = registry.get_model(model_id)
    provider = model.provider

    if provider in _client_cache:
        return _client_cache[provider]

    settings = get_settings()

    if provider == "google":
        client: BaseLLMClient = GeminiClient(
            api_key=settings.gemini.api_key
        )
    elif provider == "groq":
        if not settings.groq.api_key:
            raise LLMClientError(
                provider="groq",
                model_id=model_id,
                reason=(
                    "GROQ_API_KEY is not set in .env. "
                    "Get a free key at https://console.groq.com/keys"
                ),
            )
        client = GroqClient(api_key=settings.groq.api_key)
    elif provider == "openrouter":
        if not settings.openrouter.api_key:
            raise LLMClientError(
                provider="openrouter",
                model_id=model_id,
                reason=(
                    "OPENROUTER_API_KEY is not set in .env. "
                    "Get a free key at https://openrouter.ai/settings/keys"
                ),
            )
        client = OpenRouterClient(
            api_key=settings.openrouter.api_key,
            site_url=settings.openrouter.site_url,
            app_name=settings.openrouter.app_name,
        )
    else:
        raise LLMClientError(
            provider=provider,
            model_id=model_id,
            reason=(
                f"No BaseLLMClient implementation registered for "
                f"provider '{provider}'. "
                f"Add a concrete subclass to src/rag/clients.py and "
                f"register it in get_llm_client()."
            ),
        )

    _client_cache[provider] = client
    logger.info(
        f"get_llm_client: Constructed and cached {provider} client "
        f"for model '{model_id}'."
    )
    return client


def reset_client_cache() -> None:
    """
    Clear the provider client cache.

    Intended for:
        - Tests: each test gets a fresh client with controlled credentials
        - Settings hot-reload: forces re-construction with new API keys
        - Debugging: force re-initialisation after a suspected bad state

    Not needed in normal operation — the cache is valid for the full
    process lifetime.
    """
    _client_cache.clear()
    logger.debug("get_llm_client: Provider client cache cleared.")


# ===========================================================================
# EXCEPTION
# ===========================================================================


class LLMClientError(Exception):
    """
    Single provider-agnostic exception type for all LLM client failures.

    Every concrete BaseLLMClient subclass catches its own SDK-specific
    exceptions (google.api_core.exceptions.*, openai.APIError, etc.) and
    re-raises them as LLMClientError. Callers (RAGPipeline, BaseEvaluator,
    GeminiDatasetGenerator) catch exactly this one type regardless of
    which provider failed.

    Carries:
        provider: which provider failed ('google', 'groq', 'openrouter')
        model_id: which model was being called
        reason:   human-readable failure description, including circuit
                  breaker state, provider status URL, and debugging hints
                  for common failure modes (empty response, reasoning
                  model token budget, API key not set)

    Also registered as a FastAPI exception handler in src/api/app.py:
        LLMClientError → 502 Bad Gateway with provider field in body,
        so the Streamlit UI can display provider-specific guidance rather
        than a generic "server error" message.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[{provider}/{model_id}] {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )


__all__ = [
    # Result type
    "LLMGenerationResult",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "get_circuit_breaker_states",
    # Clients
    "BaseLLMClient",
    "GeminiClient",
    "GroqClient",
    "OpenRouterClient",
    # Factory
    "get_llm_client",
    "reset_client_cache",
    # Exception
    "LLMClientError",
]