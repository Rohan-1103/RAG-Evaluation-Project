"""
src/api/routes/models.py

Read-only model catalogue route — exposes the enabled models from
config/models.yaml over HTTP so the UI can populate a real dropdown
instead of relying on free-text model ID input.

Includes a 60-second in-memory TTL cache so repeated calls from
Streamlit reruns (which hit this endpoint on every page load for
the dropdown) don't re-parse models.yaml on every single rerun.
The cache is per-process and invalidated automatically after TTL —
no explicit invalidation needed since models.yaml is effectively
read-only at runtime.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from config import get_model_registry

router = APIRouter()

# ---------------------------------------------------------------------------
# TTL cache — single slot, keyed on (role, provider) filter combination.
# The UI only ever calls with role="rag_pipeline" or no filter, so a
# single-slot cache covers 99% of calls without complexity.
# ---------------------------------------------------------------------------
_cache: dict[tuple[str | None, str | None], tuple[Any, float]] = {}
_CACHE_TTL_SECONDS: int = 60


def _get_cached(
    role: str | None,
    provider: str | None,
) -> Any | None:
    """Return cached result if still within TTL, else None."""
    key = (role, provider)
    if key in _cache:
        result, ts = _cache[key]
        if time.monotonic() - ts < _CACHE_TTL_SECONDS:
            return result
    return None


def _set_cached(
    role: str | None,
    provider: str | None,
    result: Any,
) -> None:
    """Store result in cache with current timestamp."""
    _cache[(role, provider)] = (result, time.monotonic())


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ModelOut(BaseModel):
    """API-facing view of one entry in the model registry."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    provider: str
    context_window: int
    recommended_for: list[str]
    tags: list[str]


class ModelListOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    models: list[ModelOut]
    total: int
    cached: bool = False
    """True if this response was served from the in-memory cache."""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ModelListOut,
    summary="List enabled models, optionally filtered by role or provider",
)
async def list_models(
    role: str | None = Query(
        default=None,
        description=(
            "Filter by recommended role: "
            "rag_pipeline, judge, dataset_generation."
        ),
    ),
    provider: str | None = Query(
        default=None,
        description=(
            "Filter by provider: google, groq, openrouter."
        ),
    ),
) -> ModelListOut:
    """
    Return every enabled model in the registry, optionally narrowed
    by role or provider. Backs the model dropdown on the Evaluate and
    Compare pages.

    Response is cached for 60 seconds — repeated calls from Streamlit
    reruns are served from memory without re-parsing models.yaml.
    """
    # Cache hit
    cached = _get_cached(role, provider)
    if cached is not None:
        return ModelListOut(
            models=cached,
            total=len(cached),
            cached=True,
        )

    # Cache miss — query registry
    registry = get_model_registry()

    if role is not None:
        models = registry.get_models_for_role(role)
    elif provider is not None:
        models = registry.get_models_by_provider(provider)
    else:
        models = registry.get_enabled_models()

    result = [
        ModelOut(
            id=m.id,
            display_name=m.display_name,
            provider=m.provider,
            context_window=m.context_window,
            recommended_for=m.recommended_for,
            tags=m.tags,
        )
        for m in models
    ]

    _set_cached(role, provider, result)

    return ModelListOut(
        models=result,
        total=len(result),
        cached=False,
    )


__all__ = ["router"]