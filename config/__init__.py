"""
config/__init__.py

Configuration package entry point.

Responsibilities:
  1. Define Pydantic models for models.yaml and eval.yaml schemas.
  2. Load, validate, and cache both YAML files as immutable typed objects.
  3. Re-export all public accessors so callers use a single import surface:

        from config import get_settings, get_model_registry, get_eval_config

  4. Provide domain-level query helpers on the registry so no downstream
     module ever iterates raw YAML dicts.

Design rules enforced here:
  - YAML is loaded exactly once per process (lru_cache).
  - Every YAML field is validated by Pydantic before any module sees it.
  - Path resolution is anchored to this file's directory, not CWD.
    A module running from any working directory always finds the configs.
  - Logging is not yet initialised when this module loads — stderr only.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import BaseModel, ConfigDict

from config.settings import Settings, get_settings  # re-export

# ---------------------------------------------------------------------------
# Package-level constants
# ---------------------------------------------------------------------------
_CONFIG_DIR: Path = Path(__file__).parent
_MODELS_YAML: Path = _CONFIG_DIR / "models.yaml"
_EVAL_YAML: Path = _CONFIG_DIR / "eval.yaml"


# ===========================================================================
# PYDANTIC MODELS FOR models.yaml
# ===========================================================================


class ProviderConfig(BaseModel):
    """One entry under providers: in models.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: str
    base_url: str
    env_key: str
    client_class: str


class ScoreBand(BaseModel):
    """Colour-coded score range for dashboard traffic-light colouring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: Annotated[float, Field(ge=1.0, le=5.0)]
    max: Annotated[float, Field(ge=1.0, le=5.0)]
    color: Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]

    @model_validator(mode="after")
    def min_less_than_max(self) -> ScoreBand:
        if self.min >= self.max:
            raise ValueError(
                f"ScoreBand min ({self.min}) must be strictly less than max ({self.max})."
            )
        return self


class UIConfig(BaseModel):
    """Dashboard display configuration block from models.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filter_tags: list[str] = Field(default_factory=list)
    radar_axes: list[str] = Field(min_length=3, max_length=8)
    provider_colors: dict[str, Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]]
    score_bands: dict[str, ScoreBand]

    def color_for_provider(self, provider: str) -> str:
        """Return hex colour for a provider, falling back to 'unknown'."""
        return self.provider_colors.get(
            provider,
            self.provider_colors.get("unknown", "#6B7280"),
        )

    def band_for_score(self, score: float) -> tuple[str, ScoreBand] | None:
        """
        Return (band_name, ScoreBand) for a given score value.
        Returns None if score falls outside all defined bands.
        """
        for name, band in self.score_bands.items():
            if band.min <= score <= band.max:
                return name, band
        return None


class ModelDefaults(BaseModel):
    """Default model assignments per role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rag_model: str
    judge_model: str
    dataset_gen_model: str
    default_comparison_models: list[str] = Field(min_length=1)


class ComparisonGrid(BaseModel):
    """Parameter grid used by the multi-model comparison engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperatures: Annotated[list[float], Field(min_length=1)]
    top_k_values: Annotated[list[int], Field(min_length=1)]
    restrict_grid_to_default_models: bool = True
    max_total_runs: Annotated[PositiveInt, Field(ge=1, le=100)]

    @field_validator("temperatures")
    @classmethod
    def valid_temperatures(cls, v: list[float]) -> list[float]:
        for t in v:
            if not (0.0 <= t <= 2.0):
                raise ValueError(
                    f"Temperature {t} is outside valid range [0.0, 2.0]."
                )
        return v

    @field_validator("top_k_values")
    @classmethod
    def valid_top_k(cls, v: list[int]) -> list[int]:
        for k in v:
            if k < 1 or k > 50:
                raise ValueError(
                    f"top_k value {k} is outside valid range [1, 50]."
                )
        return v

    @property
    def total_combinations(self) -> int:
        """Total parameter combinations without model axis."""
        return len(self.temperatures) * len(self.top_k_values)


class CostEstimationConfig(BaseModel):
    """Token assumptions used for pre-run cost estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    avg_tokens_per_rag_call: PositiveInt
    avg_tokens_per_eval_call: PositiveInt
    avg_tokens_per_gen_call: PositiveInt
    safety_multiplier: Annotated[PositiveFloat, Field(ge=1.0, le=3.0)]


class LLMModelConfig(BaseModel):
    """
    Full specification for one LLM in the registry.

    Named LLMModelConfig (not ModelConfig) to avoid shadowing Pydantic's
    own ModelConfig / ConfigDict in this module's namespace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    display_name: str
    provider: str
    model_name: str
    context_window: Annotated[PositiveInt, Field(ge=512)]
    max_output_tokens: Annotated[PositiveInt, Field(ge=64)]
    cost_input_per_1k: Annotated[float, Field(ge=0.0)]
    cost_output_per_1k: Annotated[float, Field(ge=0.0)]
    supports_system_prompt: bool
    supports_json_mode: bool
    recommended_for: list[str] = Field(default_factory=list)
    rate_limit_rpm: Annotated[int, Field(ge=0)] = 0
    rate_limit_tpm: Annotated[int, Field(ge=0)] = 0
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        safety_multiplier: float = 1.25,
    ) -> float:
        """
        Return estimated USD cost for given token counts.

        Applies safety_multiplier to account for retries and overhead.
        Returns 0.0 for free-tier models (cost_input_per_1k == 0.0).
        """
        raw = (
            (input_tokens / 1000) * self.cost_input_per_1k
            + (output_tokens / 1000) * self.cost_output_per_1k
        )
        return round(raw * safety_multiplier, 8)

    def is_recommended_for(self, role: str) -> bool:
        """Check if this model is recommended for a given role."""
        return role in self.recommended_for

    @property
    def is_free_tier(self) -> bool:
        """True if both input and output costs are zero."""
        return self.cost_input_per_1k == 0.0 and self.cost_output_per_1k == 0.0


class ModelRegistry(BaseModel):
    """
    Root model for the entire models.yaml file.

    Exposes query helpers so downstream code never iterates raw lists.
    All public methods are pure — no side effects, no I/O.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: dict[str, ProviderConfig]
    models: Annotated[list[LLMModelConfig], Field(min_length=1)]
    defaults: ModelDefaults
    comparison_grid: ComparisonGrid
    cost_estimation: CostEstimationConfig
    ui: UIConfig

    # ------------------------------------------------------------------
    # Structural validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def default_models_must_exist_and_be_enabled(self) -> ModelRegistry:
        """
        Every model referenced in defaults must exist in the registry
        and have enabled=true. A default pointing at a disabled model
        would cause silent runtime failures.
        """
        checks: dict[str, str] = {
            "defaults.rag_model": self.defaults.rag_model,
            "defaults.judge_model": self.defaults.judge_model,
            "defaults.dataset_gen_model": self.defaults.dataset_gen_model,
        }
        for field, model_id in checks.items():
            model = self._find_by_id(model_id)
            if model is None:
                raise ValueError(
                    f"{field}='{model_id}' does not exist in the models list."
                )
            if not model.enabled:
                raise ValueError(
                    f"{field}='{model_id}' exists but has enabled=false. "
                    "Default models must be enabled."
                )

        for model_id in self.defaults.default_comparison_models:
            model = self._find_by_id(model_id)
            if model is None:
                raise ValueError(
                    f"defaults.default_comparison_models contains '{model_id}' "
                    "which does not exist in the models list."
                )

        return self

    @model_validator(mode="after")
    def model_providers_must_be_declared(self) -> ModelRegistry:
        """Every model's provider must be declared in the providers block."""
        declared = set(self.providers.keys())
        for m in self.models:
            if m.provider not in declared:
                raise ValueError(
                    f"Model '{m.id}' references provider '{m.provider}' "
                    f"which is not declared in providers. "
                    f"Declared providers: {sorted(declared)}."
                )
        return self

    @model_validator(mode="after")
    def model_ids_must_be_unique(self) -> ModelRegistry:
        """Duplicate model IDs cause silent incorrect lookups."""
        ids = [m.id for m in self.models]
        seen: set[str] = set()
        for model_id in ids:
            if model_id in seen:
                raise ValueError(
                    f"Duplicate model id '{model_id}' found in models list. "
                    "Every model id must be unique."
                )
            seen.add(model_id)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_by_id(self, model_id: str) -> LLMModelConfig | None:
        """Linear scan — registry is small, this is never a bottleneck."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    # ------------------------------------------------------------------
    # Public query API
    # Used by all downstream modules — never iterate self.models directly.
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> LLMModelConfig:
        """
        Return model by id.

        Raises KeyError (not None) so callers fail loudly on bad lookups.
        A module that silently gets None and proceeds causes confusing errors
        deep in the call stack.
        """
        model = self._find_by_id(model_id)
        if model is None:
            enabled_ids = [m.id for m in self.models if m.enabled]
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                f"Enabled model ids: {enabled_ids}"
            )
        return model

    def get_enabled_models(self) -> list[LLMModelConfig]:
        """Return all models where enabled=true."""
        return [m for m in self.models if m.enabled]

    def get_models_for_role(self, role: str) -> list[LLMModelConfig]:
        """
        Return enabled models recommended for a given role.

        Roles defined in models.yaml: 'judge', 'rag_pipeline', 'dataset_generation'.
        """
        return [
            m for m in self.models
            if m.enabled and role in m.recommended_for
        ]

    def get_models_by_provider(self, provider: str) -> list[LLMModelConfig]:
        """Return all enabled models from a specific provider."""
        return [
            m for m in self.models
            if m.enabled and m.provider == provider
        ]

    def get_models_by_tag(self, tag: str) -> list[LLMModelConfig]:
        """Return all enabled models carrying a specific tag."""
        return [
            m for m in self.models
            if m.enabled and tag in m.tags
        ]

    def get_default_rag_model(self) -> LLMModelConfig:
        return self.get_model(self.defaults.rag_model)

    def get_default_judge_model(self) -> LLMModelConfig:
        return self.get_model(self.defaults.judge_model)

    def get_default_dataset_gen_model(self) -> LLMModelConfig:
        return self.get_model(self.defaults.dataset_gen_model)

    def get_default_comparison_models(self) -> list[LLMModelConfig]:
        """
        Return the default comparison set.

        Silently skips disabled models — the comparison grid may have been
        configured before a model was disabled.
        """
        result: list[LLMModelConfig] = []
        for model_id in self.defaults.default_comparison_models:
            model = self._find_by_id(model_id)
            if model is not None and model.enabled:
                result.append(model)
        return result

    def get_provider(self, provider_key: str) -> ProviderConfig:
        """Return provider config by key. Raises KeyError if not found."""
        if provider_key not in self.providers:
            raise KeyError(
                f"Provider '{provider_key}' not found. "
                f"Declared providers: {list(self.providers.keys())}"
            )
        return self.providers[provider_key]

    def estimate_comparison_cost(
        self,
        model_ids: list[str],
        n_questions: int,
    ) -> dict[str, float]:
        """
        Return estimated USD cost per model for a comparison run.

        Uses CostEstimationConfig averages. Suitable for dashboard
        pre-run cost display — not billing accuracy.

        Returns dict of {model_id: estimated_cost_usd}.
        """
        ce = self.cost_estimation
        costs: dict[str, float] = {}
        for model_id in model_ids:
            model = self.get_model(model_id)
            rag_cost = model.estimate_cost(
                input_tokens=int(ce.avg_tokens_per_rag_call * 0.8),
                output_tokens=int(ce.avg_tokens_per_rag_call * 0.2),
                safety_multiplier=ce.safety_multiplier,
            )
            eval_cost = model.estimate_cost(
                input_tokens=int(ce.avg_tokens_per_eval_call * 0.85),
                output_tokens=int(ce.avg_tokens_per_eval_call * 0.15),
                safety_multiplier=ce.safety_multiplier,
            )
            costs[model_id] = round(
                (rag_cost + eval_cost) * n_questions, 6
            )
        return costs


# ===========================================================================
# PYDANTIC MODELS FOR eval.yaml
# ===========================================================================


class JudgeGlobalConfig(BaseModel):
    """Global judge behaviour settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_min: Annotated[int, Field(ge=1, le=5)]
    score_max: Annotated[int, Field(ge=1, le=5)]
    reasoning_style: Literal["step_by_step", "chain_of_thought"]
    max_reasoning_tokens: Annotated[PositiveInt, Field(ge=50, le=2000)]
    max_response_tokens: Annotated[PositiveInt, Field(ge=100, le=4096)]
    parse_failure_fallback_score: Annotated[float, Field(ge=1.0, le=5.0)]
    parse_failure_flag: str
    min_reasoning_length: Annotated[PositiveInt, Field(ge=10)]
    output_format_instruction: str

    @model_validator(mode="after")
    def min_less_than_max(self) -> JudgeGlobalConfig:
        if self.score_min >= self.score_max:
            raise ValueError(
                f"score_min ({self.score_min}) must be less than "
                f"score_max ({self.score_max})."
            )
        return self


class PromptVersionEntry(BaseModel):
    """Single changelog entry for a prompt version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    changes: list[str]


class PromptVersioningConfig(BaseModel):
    """Prompt version registry — maps version string to changelog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    active_version: str
    changelog: dict[str, PromptVersionEntry]

    @model_validator(mode="after")
    def active_version_in_changelog(self) -> PromptVersioningConfig:
        if self.active_version not in self.changelog:
            raise ValueError(
                f"active_version='{self.active_version}' is not present "
                f"in changelog. Add an entry for this version."
            )
        return self


class MetricConfig(BaseModel):
    """
    Configuration for a single evaluation metric.

    The prompt field contains the raw template string with {variable}
    placeholders. Injecting variables is the responsibility of
    evaluation/prompts.py — this model only holds the template.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    display_name: str
    description: str
    weight: Annotated[float, Field(ge=0.0, le=1.0)]
    prompt_version: str
    rubric: dict[int, str]
    prompt: str

    # Correctness-only fields — optional on other metrics
    no_reference_behaviour: Literal["skip", "score_zero"] | None = None
    no_reference_flag: str | None = None
    no_reference_score: float | None = None

    @model_validator(mode="after")
    def rubric_covers_full_scale(self) -> MetricConfig:
        """Every integer from 1 to 5 must have a rubric entry."""
        required = {1, 2, 3, 4, 5}
        present = set(self.rubric.keys())
        missing = required - present
        if missing:
            raise ValueError(
                f"Metric '{self.name}' rubric is missing entries for "
                f"scores: {sorted(missing)}. All integers 1-5 are required."
            )
        return self

    def render_prompt(self, **kwargs: str) -> str:
        """
        Inject template variables into the prompt string.

        Raises KeyError if a required variable is missing so the caller
        gets an immediate, actionable error rather than a malformed prompt
        reaching the judge LLM.
        """
        try:
            return self.prompt.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Metric '{self.name}' prompt is missing required "
                f"variable: {exc}. "
                f"Prompt expects variables derived from: "
                f"{{question}}, {{answer}}, {{context_chunks}}, "
                f"{{reference_answer}}, {{output_format_instruction}}."
            ) from exc

    def rubric_for_score(self, score: int) -> str:
        """Return rubric text for a given score integer."""
        if score not in self.rubric:
            raise KeyError(
                f"Score {score} not in rubric for metric '{self.name}'."
            )
        return self.rubric[score]


class MetricsConfig(BaseModel):
    """Container for all 4 metric configurations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    faithfulness: MetricConfig
    answer_relevance: MetricConfig
    context_precision: MetricConfig
    correctness: MetricConfig

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> MetricsConfig:
        total = (
            self.faithfulness.weight
            + self.answer_relevance.weight
            + self.context_precision.weight
            + self.correctness.weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Metric weights in eval.yaml must sum to 1.0. "
                f"Current sum: {total:.4f}. "
                f"faithfulness={self.faithfulness.weight}, "
                f"answer_relevance={self.answer_relevance.weight}, "
                f"context_precision={self.context_precision.weight}, "
                f"correctness={self.correctness.weight}."
            )
        return self

    def as_dict(self) -> dict[str, MetricConfig]:
        """Return {metric_name: MetricConfig} for uniform iteration."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "correctness": self.correctness,
        }

    def get_metric(self, name: str) -> MetricConfig:
        """Return metric by snake_case name. Raises KeyError if not found."""
        metrics = self.as_dict()
        if name not in metrics:
            raise KeyError(
                f"Metric '{name}' not found. "
                f"Valid metric names: {list(metrics.keys())}"
            )
        return metrics[name]

    @property
    def weight_map(self) -> dict[str, float]:
        """Return {metric_name: weight} for aggregation calculations."""
        return {
            name: cfg.weight
            for name, cfg in self.as_dict().items()
        }


class CompositeScoreConfig(BaseModel):
    """Composite score computation and display settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    formula: Literal["weighted_sum"]
    scale_min: Annotated[float, Field(ge=0.0)]
    scale_max: Annotated[float, Field(le=10.0)]
    display_label: str
    missing_metric_strategy: Literal["exclude_from_weight", "score_zero"]


class AggregationConfig(BaseModel):
    """Statistical aggregation settings for per-model summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    statistics: Annotated[list[str], Field(min_length=1)]
    min_questions_for_reliable_aggregation: PositiveInt
    low_sample_warning_threshold: PositiveInt
    low_sample_warning_message: str


class EvalRunConfig(BaseModel):
    """Job-level evaluation run behaviour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_questions_per_run: Annotated[PositiveInt, Field(ge=1, le=10000)]
    continue_on_question_failure: bool
    parallel_metrics_per_question: bool
    inter_question_delay_seconds: Annotated[float, Field(ge=0.0, le=60.0)]
    checkpoint_after_each_question: bool


class ReportConfig(BaseModel):
    """Export report column and section configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    csv_columns: Annotated[list[str], Field(min_length=1)]
    summary_csv_columns: Annotated[list[str], Field(min_length=1)]
    pdf_sections: Annotated[list[str], Field(min_length=1)]


class EvalConfig(BaseModel):
    """
    Root model for the entire eval.yaml file.

    Exposes the resolved, output_format_instruction-injected prompt
    for each metric via get_rendered_system_prompt().
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge: JudgeGlobalConfig
    prompt_versioning: PromptVersioningConfig
    metrics: MetricsConfig
    composite_score: CompositeScoreConfig
    aggregation: AggregationConfig
    eval_run: EvalRunConfig
    report: ReportConfig

    @model_validator(mode="after")
    def metric_prompt_versions_match_active(self) -> EvalConfig:
        """
        Each metric's prompt_version must match the active version.
        A stale metric prompt is a reproducibility error —
        you cannot compare runs if prompts silently differ.
        """
        active = self.prompt_versioning.active_version
        for name, cfg in self.metrics.as_dict().items():
            if cfg.prompt_version != active:
                raise ValueError(
                    f"Metric '{name}' has prompt_version='{cfg.prompt_version}' "
                    f"but active_version='{active}'. "
                    f"Update the metric's prompt_version or bump active_version."
                )
        return self

    def get_metric_prompt(
        self,
        metric_name: str,
        **template_vars: str,
    ) -> str:
        """
        Return the fully rendered judge prompt for a metric.

        Automatically injects output_format_instruction from the global
        judge config so callers never need to pass it explicitly.

        Usage:
            prompt = eval_cfg.get_metric_prompt(
                "faithfulness",
                question=q,
                answer=a,
                context_chunks=ctx,
            )
        """
        metric = self.metrics.get_metric(metric_name)
        return metric.render_prompt(
            output_format_instruction=self.judge.output_format_instruction,
            **template_vars,
        )

    @property
    def active_prompt_version(self) -> str:
        return self.prompt_versioning.active_version


# ===========================================================================
# YAML LOADERS — with full validation and actionable error messages
# ===========================================================================


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return the raw dict.

    Raises FileNotFoundError with the absolute path so the error message
    is immediately actionable — no hunting for which file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required config file not found: {path.resolve()}\n"
            f"Ensure '{path.name}' exists in the config/ directory."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file '{path.resolve()}' must contain a YAML mapping "
            f"at the top level, got {type(data).__name__}."
        )
    return data


def _load_model_registry() -> ModelRegistry:
    """
    Load and validate models.yaml.

    Called once by get_model_registry() — not called directly.
    Separating load logic from the lru_cache accessor keeps the
    cache-clear test pattern clean:

        get_model_registry.cache_clear()
        registry = get_model_registry()
    """
    raw = _load_yaml(_MODELS_YAML)
    try:
        return ModelRegistry.model_validate(raw)
    except Exception as exc:
        print(
            f"[FATAL] models.yaml validation failed.\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_eval_config() -> EvalConfig:
    """Load and validate eval.yaml. Called once by get_eval_config()."""
    raw = _load_yaml(_EVAL_YAML)
    try:
        return EvalConfig.model_validate(raw)
    except Exception as exc:
        print(
            f"[FATAL] eval.yaml validation failed.\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# ===========================================================================
# CACHED SINGLETON ACCESSORS — public API
# ===========================================================================


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """
    Return the process-level ModelRegistry singleton.

    Pattern identical to get_settings() — constructed once,
    immutable after construction, cache_clear() in tests.

    Usage:
        from config import get_model_registry
        registry = get_model_registry()
        flash = registry.get_model("gemini-1.5-flash")
    """
    return _load_model_registry()


@lru_cache(maxsize=1)
def get_eval_config() -> EvalConfig:
    """
    Return the process-level EvalConfig singleton.

    Usage:
        from config import get_eval_config
        eval_cfg = get_eval_config()
        prompt = eval_cfg.get_metric_prompt(
            "faithfulness",
            question=q,
            answer=a,
            context_chunks=ctx,
        )
    """
    return _load_eval_config()


# ===========================================================================
# PUBLIC SURFACE — everything a module needs from the config package
# ===========================================================================

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Model registry
    "ModelRegistry",
    "LLMModelConfig",
    "ProviderConfig",
    "ComparisonGrid",
    "CostEstimationConfig",
    "UIConfig",
    "ScoreBand",
    "ModelDefaults",
    "get_model_registry",
    # Eval config
    "EvalConfig",
    "JudgeGlobalConfig",
    "MetricConfig",
    "MetricsConfig",
    "CompositeScoreConfig",
    "AggregationConfig",
    "EvalRunConfig",
    "ReportConfig",
    "PromptVersioningConfig",
    "get_eval_config",
]