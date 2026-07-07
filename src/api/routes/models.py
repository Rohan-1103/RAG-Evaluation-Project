"""
src/api/routes/models.py

Read-only model catalogue route — exposes the enabled models from
config/models.yaml over HTTP so the UI can populate a real dropdown
instead of relying on free-text model ID input.

This route exists specifically to close the gap documented in
ui/pages/04_compare.py's own docstring: a model dropdown was
deliberately NOT built earlier because no endpoint exposed the live
registry, and a UI-side hardcoded copy would have silently drifted
from models.yaml. This route is that missing endpoint — nothing here
duplicates registry logic, it only serializes what
config.get_model_registry() already computes.

Endpoint:
    GET /api/v1/models                List all enabled models
    GET /api/v1/models?role=judge      Filter by recommended role
    GET /api/v1/models?provider=groq    Filter by provider
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from config import get_model_registry

router = APIRouter()


class ModelOut(BaseModel):
    """API-facing view of one entry in the model registry."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    provider: str
    context_window: int
    recommended_for: list[str]
    is_free_tier: bool
    tags: list[str]


class ModelListOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    models: list[ModelOut]
    total: int


@router.get(
    "",
    response_model=ModelListOut,
    summary="List enabled models, optionally filtered by role or provider",
)
async def list_models(
    role: str | None = Query(
        default=None,
        description="Filter by recommended role: rag_pipeline, judge, dataset_generation.",
    ),
    provider: str | None = Query(
        default=None,
        description="Filter by provider: google, groq, openai, anthropic.",
    ),
) -> ModelListOut:
    """
    Return every enabled model in the registry, optionally narrowed by
    role or provider. Backs the model dropdown on the Evaluate and
    Compare pages.
    """
    registry = get_model_registry()

    if role is not None:
        models = registry.get_models_for_role(role)
    elif provider is not None:
        models = registry.get_models_by_provider(provider)
    else:
        models = registry.get_enabled_models()

    return ModelListOut(
        models=[
            ModelOut(
                id=m.id,
                display_name=m.display_name,
                provider=m.provider,
                context_window=m.context_window,
                recommended_for=m.recommended_for,
                is_free_tier=m.is_free_tier,
                tags=m.tags,
            )
            for m in models
        ],
        total=len(models),
    )


__all__ = ["router"]