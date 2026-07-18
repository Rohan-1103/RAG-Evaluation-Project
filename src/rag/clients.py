"""
src/rag/clients.py

BaseLLMClient — the single abstraction every text-generation call in
this codebase goes through. Before this file existed, three separate
call sites (RAGPipeline, BaseEvaluator, GeminiDatasetGenerator) each
constructed a google.genai.Client directly and embedded Gemini-specific
retry/parsing logic inline. This collapses all three into one contract:
generate(model_id, prompt, ...) -> LLMGenerationResult, with exactly
one concrete implementation per provider.

Design notes:
  - Clients are PROVIDER-scoped, not MODEL-scoped. One GeminiClient
    instance serves every Gemini model variant; model_id travels with
    each call, not with client construction. This is simpler than
    RAGPipeline's old per-(model,temperature,tokens) cache and is
    possible specifically because neither SDK requires a model-bound
    object — both accept model_id as a per-call parameter.
  - Retry lives INSIDE each client's generate(), one policy per
    provider, since Gemini's and Groq's transient-failure shapes
    differ (different exception types, different rate-limit headers).
    Call sites never see a retry decorator again.
  - get_llm_client(model_id) is the single chokepoint every caller
    uses. It resolves provider from config/models.yaml's registry,
    constructs (or returns a cached) client, and is what makes a
    comparison grid mixing "gemini-2.0-flash" and
    "llama-3.1-70b-versatile" in one request work transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
    """Uniform output shape across every provider — callers never branch on provider."""

    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    provider: str


# ===========================================================================
# ABSTRACT BASE
# ===========================================================================


class BaseLLMClient(ABC):
    """
    One method, implemented once per provider. Every retry, every
    SDK-specific request/response shape, lives inside the concrete
    subclass — callers (RAGPipeline, BaseEvaluator, dataset generator)
    never import google.genai or openai directly again.
    """

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

        Never raises after exhausting retries — implementations catch
        their own SDK exceptions internally and raise LLMClientError,
        a single provider-agnostic type callers can catch uniformly.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ===========================================================================
# GEMINI CLIENT
# ===========================================================================


class GeminiClient(BaseLLMClient):
    """Wraps google.genai. Extracted unchanged from the 3 call sites that
    previously duplicated this logic — RAGPipeline, BaseEvaluator,
    GeminiDatasetGenerator."""

    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "google"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        try:
            # With:
            from config import get_model_registry
            try:
                registry_entry = get_model_registry().get_model(model_id)
                api_model_name = getattr(registry_entry, "model_name", model_id) or model_id
            except KeyError:
                api_model_name = model_id
            
            response = self._client.models.generate_content(
                model=api_model_name, contents=prompt, config=config,
            )
        except Exception as exc:
            raise LLMClientError(
                provider="google", model_id=model_id, reason=str(exc),
                original_exception=exc,
            ) from exc

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        text = response.text or ""
        if not text.strip():
            raise LLMClientError(
                provider="google", model_id=model_id,
                reason="Empty response — possible safety filter block.",
            )

        return LLMGenerationResult(
            text=text.strip(), input_tokens=input_tokens,
            output_tokens=output_tokens, model_id=model_id, provider="google",
        )


# ===========================================================================
# GROQ CLIENT
# ===========================================================================


class GroqClient(BaseLLMClient):
    """
    Wraps Groq's OpenAI-compatible chat completions endpoint via the
    `openai` SDK pointed at Groq's base URL. system_instruction maps
    to a prepended {"role": "system"} message since Groq's endpoint
    has no model-level system field the way Gemini does.
    """

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=self._BASE_URL)

    @property
    def provider_name(self) -> str:
        return "groq"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def generate(
    self,
    model_id: str,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        # Resolve the actual API model string from the registry.
        # model_id is our internal registry key (e.g. "gpt-oss-120b");
        # model_name is what Groq's API actually accepts (e.g. "openai/gpt-oss-120b").
        # These differ whenever a provider uses a namespaced API string.
        from config import get_model_registry
        try:
            registry_entry = get_model_registry().get_model(model_id)
            api_model_name = getattr(registry_entry, "model_name", model_id) or model_id
        except KeyError:
            api_model_name = model_id

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=api_model_name,   # <-- use the resolved API name, not model_id
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise LLMClientError(
                provider="groq", model_id=model_id, reason=str(exc),
                original_exception=exc,
            ) from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else ""

        # NEW
        if not text or not text.strip():
            # Reasoning models (e.g. gpt-oss-120b) spend tokens on internal
            # reasoning before producing content. If content is empty, the most
            # common cause is max_tokens being too low to leave room for output
            # after reasoning tokens are consumed. Check usage for reasoning_tokens.
            reasoning = getattr(
                response.choices[0].message if response.choices else None,
                "reasoning", None
            )
            hint = (
                f" Model produced {usage.completion_tokens_details.reasoning_tokens} "
                f"reasoning tokens — increase max_output_tokens."
                if (usage and hasattr(usage, "completion_tokens_details")
                    and usage.completion_tokens_details
                    and usage.completion_tokens_details.reasoning_tokens)
                else " Increase max_output_tokens if using a reasoning model."
            )
            raise LLMClientError(
                provider="groq", model_id=model_id,
                reason=f"Empty response from Groq.{hint}",
            )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMGenerationResult(
            text=text.strip(), input_tokens=input_tokens,
            output_tokens=output_tokens, model_id=model_id, provider="groq",
        )

class OpenRouterClient(BaseLLMClient):
    """
    Wraps OpenRouter's OpenAI-compatible endpoint.

    OpenRouter gives access to 200+ models (NVIDIA, Qwen, Llama,
    Mistral, Claude, GPT-4o) through one API key, including 20+
    permanently free models (marked with :free suffix in model ID).

    Two required headers distinguish OpenRouter from Groq despite
    sharing the same OpenAI SDK:
      HTTP-Referer: your app's URL (for OpenRouter dashboard tracking)
      X-Title:      your app's name (shown in OpenRouter logs)

    See: https://openrouter.ai/docs#headers
    """

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        site_url: str = "http://localhost:8501",
        app_name: str = "RAG Eval Bench",
    ) -> None:
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

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_instruction: str | None = None,
    ) -> LLMGenerationResult:
        # Resolve actual API model string from registry
        from config import get_model_registry
        try:
            registry_entry = get_model_registry().get_model(model_id)
            api_model_name = getattr(
                registry_entry, "model_name", model_id
            ) or model_id
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
        text = choice.message.content if choice and choice.message else ""

        if not text or not text.strip():
            # Same reasoning-model guard as GroqClient
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
                hint = (
                    f" Model used reasoning tokens — "
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
# FACTORY — single chokepoint every call site uses
# ===========================================================================


_client_cache: dict[str, BaseLLMClient] = {}


def get_llm_client(model_id: str) -> BaseLLMClient:
    """
    Resolve model_id -> provider via the registry, return a cached
    client for that provider (constructed lazily, once per provider
    per process — not once per model).
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
        client: BaseLLMClient = GeminiClient(api_key=settings.gemini.api_key)
    elif provider == "groq":
        if not settings.groq.api_key:
            raise LLMClientError(
                provider="groq", model_id=model_id,
                reason=(
                    "GROQ_API_KEY is not set in .env. Get a free key at "
                    "https://console.groq.com/keys"
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
            provider=provider, model_id=model_id,
            reason=f"No BaseLLMClient implementation registered for provider '{provider}'.",
        )

    _client_cache[provider] = client
    logger.info(f"get_llm_client: Constructed and cached {provider} client.")
    return client


def reset_client_cache() -> None:
    """Test-only — clears cached provider clients."""
    _client_cache.clear()


# ===========================================================================
# EXCEPTION
# ===========================================================================


class LLMClientError(Exception):
    """Single provider-agnostic exception type. Callers (RAGPipeline,
    BaseEvaluator, dataset generator) catch this one type regardless
    of which provider actually failed."""

    def __init__(
        self, provider: str, model_id: str, reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(f"[{provider}/{model_id}] {reason}")


__all__ = [
    "LLMGenerationResult", "BaseLLMClient", "GeminiClient", "GroqClient",
    "get_llm_client", "reset_client_cache", "LLMClientError",
]