"""
config/settings.py

Single source of truth for all runtime configuration.

Design principles:
  - Every module receives a `Settings` instance via constructor injection.
  - No module ever calls `os.environ` or `os.getenv` directly — ever.
  - Settings is instantiated ONCE at process startup (module-level singleton).
  - Sub-configs are nested Pydantic models — typed, validated, immutable after init.
  - `model_config = ConfigDict(frozen=True)` prevents accidental mutation at runtime.

Usage:
    from config.settings import get_settings
    settings = get_settings()           # returns cached singleton
    settings.gemini.api_key             # typed, validated, never None
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

# Absolute path to .env — always resolves to project root
# regardless of which directory the script is run from
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

from pydantic import (  # type: ignore[import-not-found]
    AnyHttpUrl,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for environments where pydantic-settings isn't installed.
    # Use pydantic.BaseModel as a stand-in for BaseSettings and a plain dict
    # for SettingsConfigDict to keep type-checkers and linters happy.
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore

# ---------------------------------------------------------------------------
# Type aliases — improves readability in field signatures
# ---------------------------------------------------------------------------
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]
AppEnv = Literal["development", "staging", "production"]
EmbeddingProvider = Literal["google", "huggingface"]


# ===========================================================================
# SUB-CONFIG MODELS
# Each section of .env maps to one of these nested models.
# Grouping by concern (LLM, storage, eval) makes injection granular —
# a module that only needs eval config receives EvalConfig, not all of Settings.
# ===========================================================================


class GeminiConfig(BaseSettings):
    """Google Gemini LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    api_key: Annotated[
        str,
        Field(
            min_length=10,
            description="Google Gemini API key. Required.",
        ),
    ]


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    provider: Annotated[
        EmbeddingProvider,
        Field(
            default="huggingface",
            description=(
                "Embedding backend. "
                "'huggingface' runs locally (zero cost). "
                "'google' uses text-embedding-004 API."
            ),
        ),
    ]

    model: Annotated[
        str,
        Field(
            default="all-MiniLM-L6-v2",
            description=(
                "Model name. "
                "HuggingFace: 'all-MiniLM-L6-v2' | 'all-mpnet-base-v2'. "
                "Google: 'models/text-embedding-004'."
            ),
        ),
    ]

    @model_validator(mode="after")
    def validate_google_model_name(self) -> EmbeddingConfig:
        """Google embedding models must be prefixed with 'models/'."""
        if self.provider == "google" and not self.model.startswith("models/"):
            raise ValueError(
                f"Google embedding model must start with 'models/', got: '{self.model}'. "
                "Use 'models/text-embedding-004'."
            )
        return self


class ChromaConfig(BaseSettings):
    """ChromaDB vector store configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CHROMA_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    persist_dir: Annotated[
        Path,
        Field(
            default=Path("./data/vectorstore"),
            description="Directory where ChromaDB persists its data.",
        ),
    ]

    default_collection: Annotated[
        str,
        Field(
            default="rag_eval_bench",
            min_length=1,
            max_length=64,
            description="Default ChromaDB collection name.",
        ),
    ]


class StorageConfig(BaseSettings):
    """Relational storage (SQLite / PostgreSQL) configuration."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    database_url: Annotated[
        str,
        Field(
            default="sqlite+aiosqlite:///./data/rag_eval.db",
            description=(
                "SQLAlchemy async database URL. "
                "SQLite: 'sqlite+aiosqlite:///./data/rag_eval.db'. "
                "PostgreSQL: 'postgresql+asyncpg://user:pass@host:5432/db'."
            ),
        ),
    ]

    raw_docs_dir: Annotated[
        Path,
        Field(default=Path("./data/raw_docs")),
    ]

    processed_docs_dir: Annotated[
        Path,
        Field(default=Path("./data/processed")),
    ]

    datasets_dir: Annotated[
        Path,
        Field(default=Path("./data/datasets")),
    ]

    comparison_results_dir: Annotated[
        Path,
        Field(default=Path("./data/comparison_results")),
    ]


class IngestionConfig(BaseSettings):
    """Document ingestion and chunking configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEFAULT_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    chunk_size: Annotated[
        PositiveInt,
        Field(
            default=1000,
            ge=100,
            le=8000,
            description="Max characters per chunk.",
        ),
    ]

    chunk_overlap: Annotated[
        PositiveInt,
        Field(
            default=200,
            ge=0,
            le=2000,
            description="Character overlap between adjacent chunks.",
        ),
    ]

    pdf_max_pages: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Max PDF pages to load. 0 = no limit.",
        ),
    ]

    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> IngestionConfig:
        """Overlap must be strictly less than chunk size or retrieval degrades."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size}). "
                "Equal or larger overlap causes infinite retrieval loops."
            )
        return self


class JudgeConfig(BaseSettings):
    """LLM-as-a-Judge evaluation engine configuration."""

    model_config = SettingsConfigDict(
        env_prefix="JUDGE_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    model: Annotated[
        str,
        Field(
            default="gemini-1.5-pro",
            description=(
                "LLM used as the evaluation judge. "
                "Use the strongest available model — judge quality "
                "directly determines metric reliability."
            ),
        ),
    ]

    temperature: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=2.0,
            description=(
                "Judge temperature. MUST be 0.0 for deterministic, "
                "reproducible scores. Never raise this for evaluation."
            ),
        ),
    ]

    max_retries: Annotated[
        PositiveInt,
        Field(
            default=3,
            description="Max retry attempts on API failure (tenacity).",
        ),
    ]

    retry_wait_seconds: Annotated[
        PositiveFloat,
        Field(
            default=2.0,
            description="Base wait between retries (exponential backoff base).",
        ),
    ]


class MetricWeightsConfig(BaseSettings):
    """
    Composite score weights for multi-metric aggregation.

    The composite score is the single number shown in comparison rankings.
    Weights are business-configurable — a compliance use case might weight
    Faithfulness at 0.50; a chatbot might weight Answer Relevance at 0.40.
    """

    model_config = SettingsConfigDict(
        env_prefix="WEIGHT_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    faithfulness: Annotated[
        float,
        Field(default=0.30, ge=0.0, le=1.0),
    ]

    answer_relevance: Annotated[
        float,
        Field(default=0.25, ge=0.0, le=1.0),
    ]

    context_precision: Annotated[
        float,
        Field(default=0.25, ge=0.0, le=1.0),
    ]

    correctness: Annotated[
        float,
        Field(default=0.20, ge=0.0, le=1.0),
    ]

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> MetricWeightsConfig:
        """
        Weights not summing to 1.0 produce composite scores outside [0, 5]
        which breaks dashboard scaling and comparisons.
        """
        total = (
            self.faithfulness
            + self.answer_relevance
            + self.context_precision
            + self.correctness
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Metric weights must sum to exactly 1.0. Current sum: {total:.4f}. "
                f"faithfulness={self.faithfulness}, answer_relevance={self.answer_relevance}, "
                f"context_precision={self.context_precision}, correctness={self.correctness}."
            )
        return self


class DatasetGenConfig(BaseSettings):
    """Synthetic test dataset generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATASET_GEN_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    model: Annotated[
        str,
        Field(
            default="gemini-1.5-flash",
            description=(
                "LLM used to generate synthetic Q&A pairs. "
                "Flash is sufficient — this is generation, not evaluation."
            ),
        ),
    ]

    temperature: Annotated[
        float,
        Field(
            default=0.4,
            ge=0.0,
            le=2.0,
            description=(
                "Temperature for Q&A generation. "
                "Slightly above 0 produces varied questions; "
                "too high produces nonsensical or unfaithful questions."
            ),
        ),
    ]

    max_pairs_per_chunk: Annotated[
        PositiveInt,
        Field(
            default=3,
            ge=1,
            le=10,
            description="Max Q&A pairs generated per document chunk.",
        ),
    ]


class ComparisonConfig(BaseSettings):
    """Multi-model comparison engine concurrency configuration."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    max_concurrent_runs: Annotated[
        PositiveInt,
        Field(
            default=3,
            description=(
                "Max model runs executing concurrently. "
                "Gemini free tier: ~60 RPM — keep at 2-3 to avoid 429s."
            ),
        ),
    ]

    max_concurrent_eval_calls: Annotated[
        PositiveInt,
        Field(
            default=5,
            description=(
                "Max concurrent judge LLM calls within a single eval run. "
                "Each question fires 4 async metric calls — this semaphore "
                "limits total in-flight judge requests."
            ),
        ),
    ]


class APIConfig(BaseSettings):
    """FastAPI backend server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    host: Annotated[str, Field(default="0.0.0.0")]
    port: Annotated[PositiveInt, Field(default=8000, ge=1024, le=65535)]
    reload: Annotated[bool, Field(default=True)]
    log_level: Annotated[str, Field(default="info")]

    # Plain string — pydantic-settings reads strings perfectly fine.
    # Parsed into a list via the property below.
    # Rename avoids pydantic-settings trying to coerce it into list[str].
    cors_origins_raw: Annotated[
        str,
        Field(
            default="http://localhost:8501,http://127.0.0.1:8501",
            alias="API_CORS_ORIGINS",
        ),
    ] = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def cors_origins(self) -> list[str]:
        """
        Parse cors_origins_raw into a list at access time.
        Handles both comma-separated and JSON array formats.
        Never touches pydantic-settings field parsing.
        """
        import json

        raw = self.cors_origins_raw.strip()

        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        return [o.strip() for o in raw.split(",") if o.strip()]


class LoggingConfig(BaseSettings):
    """Loguru structured logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    level: Annotated[LogLevel, Field(default="DEBUG")]
    format: Annotated[LogFormat, Field(default="text")]
    file: Annotated[Path, Field(default=Path("./logs/app.log"))]
    rotation: Annotated[str, Field(default="10 MB")]
    retention: Annotated[str, Field(default="7 days")]


class PhoenixConfig(BaseSettings):
    """Arize Phoenix open-source LLM tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PHOENIX_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    enabled: Annotated[bool, Field(default=True)]
    host: Annotated[str, Field(default="localhost")]
    port: Annotated[PositiveInt, Field(default=6006)]
    project_name: Annotated[str, Field(default="rag-eval-bench")]

    @property
    def collector_endpoint(self) -> str:
        """OTLP gRPC endpoint Phoenix listens on."""
        return f"http://{self.host}:{self.port}"


class LangSmithConfig(BaseSettings):
    """LangSmith optional tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LANGSMITH_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    enabled: Annotated[bool, Field(default=False)]
    api_key: Annotated[str, Field(default="")]
    project: Annotated[str, Field(default="rag-eval-bench")]


# ===========================================================================
# ROOT SETTINGS — aggregates all sub-configs
# ===========================================================================


class Settings(BaseSettings):
    """
    Root application settings.

    Aggregates all sub-configs into a single injectable object.
    Constructed once via get_settings() and cached for the process lifetime.

    Injection pattern:
        def __init__(self, settings: Settings) -> None:
            self._judge_model = settings.judge.model
            self._weights = settings.metric_weights

    Never do:
        import os; os.getenv("JUDGE_MODEL")   # bypasses validation entirely
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    # --- Runtime environment ---
    app_env: Annotated[AppEnv, Field(default="development")]
    secret_key: Annotated[
        str,
        Field(
            min_length=32,
            description="Secret key for token signing. Min 32 chars.",
        ),
    ]

    # --- Streamlit frontend ---
    api_base_url: Annotated[
        str,
        Field(
            default="http://localhost:8000",
            description="Base URL the Streamlit UI uses to call FastAPI.",
        ),
    ]

    streamlit_port: Annotated[PositiveInt, Field(default=8501)]

    # --- Sub-configs (composed, not inherited) ---
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metric_weights: MetricWeightsConfig = Field(default_factory=MetricWeightsConfig)
    dataset_gen: DatasetGenConfig = Field(default_factory=DatasetGenConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    phoenix: PhoenixConfig = Field(default_factory=PhoenixConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)

    @model_validator(mode="after")
    def warn_production_defaults(self) -> Settings:
        """
        Fail fast in production if dangerous defaults are present.

        In development, misconfigured secrets are an annoyance.
        In production, they are a security incident.
        """
        if self.app_env == "production":
            if self.secret_key == "replace_with_a_strong_random_secret_min_32_chars":
                raise ValueError(
                    "SECRET_KEY must not be the default placeholder value in production."
                )
            if self.api.reload:
                raise ValueError(
                    "API_RELOAD must be false in production. "
                    "Uvicorn reload mode is a development-only feature."
                )
            if self.logging.level == "DEBUG":
                raise ValueError(
                    "LOG_LEVEL=DEBUG in production leaks sensitive data. "
                    "Use INFO or WARNING."
                )
        return self

    @model_validator(mode="after")
    def create_required_directories(self) -> Settings:
        """
        Ensure all data directories exist at startup.

        Failing at dir-creation time is better than failing mid-run
        when writing the first output file.
        """
        dirs: list[Path] = [
            self.chroma.persist_dir,
            self.storage.raw_docs_dir,
            self.storage.processed_docs_dir,
            self.storage.datasets_dir,
            self.storage.comparison_results_dir,
            self.logging.file.parent,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def tracing_enabled(self) -> bool:
        """True if any observability backend is active."""
        return self.phoenix.enabled or self.langsmith.enabled


# ===========================================================================
# SINGLETON ACCESSOR
# ===========================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-level Settings singleton.

    `lru_cache(maxsize=1)` ensures Settings is constructed and validated
    exactly once per process. Subsequent calls return the cached instance
    at O(1) cost with zero re-parsing.

    In tests, call `get_settings.cache_clear()` before patching .env
    to force re-construction with test values:

        def test_something(monkeypatch):
            monkeypatch.setenv("APP_ENV", "production")
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.is_production
    """
    try:
        return Settings()
    except Exception as exc:
        # Print to stderr directly — logging is not yet initialised at this point
        print(
            f"[FATAL] Settings validation failed. Fix your .env file.\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)