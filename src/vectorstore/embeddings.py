"""
src/vectorstore/embeddings.py

EmbeddingManager — single interface for all embedding backends.

Design contract:
  - One class, two backends: HuggingFace (local, free) and Google.
  - The backend is selected at construction time from EmbeddingConfig.
  - All callers receive np.ndarray — never provider-specific objects.
  - Batch size, normalisation, and retry logic are internal concerns.
    Callers pass text, receive vectors. Nothing else.

Why a dedicated manager class instead of calling the SDK directly:
  - The IngestionPipeline, RAGPipeline, and RAGRetriever all need
    embeddings. If each called the SDK directly, swapping models
    would require changes in three places.
  - EmbeddingManager is injected — every consumer depends on this
    class, never on SentenceTransformer or Google SDK directly.
  - Unit tests mock EmbeddingManager. No test ever loads a real model.

Embedding dimension contract:
  - dimension is set at model load time and is immutable.
  - ChromaVectorStore reads manager.dimension to validate that
    documents being added match the collection's existing dimension.
  - If you change the embedding model, delete existing ChromaDB
    collections — old vectors and new vectors are incomparable.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from config.settings import EmbeddingConfig

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmbeddingProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class EmbeddingResult(BaseModel):
    """
    Structured result from a batch embedding call.

    Returns more than just the array so callers have full observability:
      - token_count for cost estimation (Google charges per token)
      - latency_ms for performance monitoring
      - model_name to verify which model produced these vectors
      - dimension for downstream validation
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    embeddings: Annotated[
        np.ndarray,
        Field(description="Shape (N, embedding_dim). Row i = embedding for text i."),
    ]
    model_name: str
    provider: EmbeddingProvider
    dimension: int
    text_count: int
    latency_ms: float
    estimated_token_count: int          # Approximate — used for cost tracking

    @property
    def shape(self) -> tuple[int, int]:
        return (self.text_count, self.dimension)

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult("
            f"shape={self.shape}, "
            f"model='{self.model_name}', "
            f"latency={self.latency_ms:.1f}ms)"
        )

# ===========================================================================
# EMBEDDING MANAGER
# ===========================================================================


class EmbeddingManager:
    """
    Unified interface for embedding text into dense vectors.

    Supports two backends:
      1. HuggingFace SentenceTransformer — runs locally, zero API cost.
         Best for development and free-tier deployments.
         Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).

      2. Google text-embedding-004 — API call, costs per token.
         Best for production when embedding quality is critical.
         Requires GEMINI_API_KEY.

    Usage:
        manager = EmbeddingManager(settings.embedding)

        # Embed a batch of chunks
        result = manager.embed(["text one", "text two", "text three"])
        vectors = result.embeddings    # np.ndarray shape (3, 384)

        # Embed a single query
        query_vec = manager.embed_query("What is RAG?")  # shape (384,)

    Thread safety:
        SentenceTransformer.encode() is thread-safe for read operations.
        Google API calls are stateless. EmbeddingManager instances can be
        shared across threads without locking.
    """

    # Average characters per token — used for approximate token counting
    # without running a real tokeniser. Conservative estimate.
    _CHARS_PER_TOKEN: float = 4.0

    # Maximum texts per batch for HuggingFace encode()
    # Larger batches use more RAM; smaller batches are slower.
    # 64 is a good balance for CPU inference on consumer hardware.
    _HF_BATCH_SIZE: int = 64

    # Maximum texts per batch for Google embedding API
    # Google recommends max 100 per request.
    _GOOGLE_BATCH_SIZE: int = 100

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialise the embedding manager and load the model.

        Model loading happens here, not lazily, so startup failures
        are caught immediately rather than during the first embed call.

        Args:
            config: EmbeddingConfig from Settings. Determines which
                    backend and model to use.
        """
        self._config = config
        self._provider = EmbeddingProvider(config.provider)
        self._model_name = config.model
        self._dimension: int | None = None
        self._model: object | None = None   # SentenceTransformer or None

        self._load_model()

        logger.info(
            f"EmbeddingManager initialised. "
            f"provider={self._provider.value}, "
            f"model={self._model_name}, "
            f"dimension={self._dimension}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the embedding model based on configured provider."""
        if self._provider == EmbeddingProvider.HUGGINGFACE:
            self._load_huggingface_model()
        elif self._provider == EmbeddingProvider.GOOGLE:
            self._load_google_model()
        else:
            raise EmbeddingError(
                provider=self._provider.value,
                reason=f"Unknown provider '{self._provider.value}'. "
                       f"Valid options: huggingface, google.",
            )

    def _load_huggingface_model(self) -> None:
        """
        Load SentenceTransformer model into memory.

        First call downloads the model from HuggingFace Hub (~90MB for
        all-MiniLM-L6-v2). Subsequent runs use the local cache at
        ~/.cache/huggingface/hub/.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading HuggingFace model '{self._model_name}'. "
                f"First run downloads ~90MB to HuggingFace cache."
            )
            self._model = SentenceTransformer(self._model_name)
            # Probe dimension by embedding a single dummy string
            probe: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                ["probe"],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._dimension = int(probe.shape[1])
            logger.info(
                f"HuggingFace model loaded. "
                f"dimension={self._dimension}"
            )
        except ImportError as exc:
            raise EmbeddingError(
                provider="huggingface",
                reason=(
                    "sentence-transformers is not installed. "
                    "Run: poetry add sentence-transformers"
                ),
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                provider="huggingface",
                reason=f"Failed to load model '{self._model_name}': {exc}",
                original_exception=exc,
            ) from exc

    def _load_google_model(self) -> None:
        """
        Validate Google embedding configuration.

        Google's embedding API is stateless — no model object to load.
        We validate the API key is present and probe the dimension
        with a single test embedding call.
        """
        try:
            import google.generativeai as genai
            from config.settings import get_settings

            settings = get_settings()
            genai.configure(api_key=settings.gemini.api_key)

            logger.info(
                f"Probing Google embedding model '{self._model_name}' "
                f"to determine dimension..."
            )
            probe_result = genai.embed_content(
                model=self._model_name,
                content="probe",
                task_type="retrieval_document",
            )
            self._dimension = len(probe_result["embedding"])
            logger.info(
                f"Google embedding model ready. "
                f"dimension={self._dimension}"
            )
        except ImportError as exc:
            raise EmbeddingError(
                provider="google",
                reason=(
                    "google-generativeai is not installed. "
                    "Run: poetry add google-generativeai"
                ),
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                provider="google",
                reason=(
                    f"Failed to initialise Google embedding model "
                    f"'{self._model_name}': {exc}. "
                    f"Check GEMINI_API_KEY in .env."
                ),
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Public embedding interface
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Embed a batch of texts into dense vectors.

        Args:
            texts:          List of strings to embed. Must be non-empty.
                            Empty strings are replaced with a single space
                            to avoid model errors — logged as a warning.
            show_progress:  Show tqdm progress bar (HuggingFace only).
                            Use True for large ingestion batches.

        Returns:
            EmbeddingResult with shape (len(texts), dimension).
            Row i corresponds to texts[i].

        Raises:
            EmbeddingError: on model inference failure.
            ValueError: if texts is empty.
        """
        if not texts:
            raise ValueError(
                "EmbeddingManager.embed() received an empty texts list. "
                "Pass at least one string."
            )

        # Replace empty strings — embedding models handle them poorly
        sanitised, empty_indices = self._sanitise_texts(texts)

        start_ms = time.monotonic() * 1000

        if self._provider == EmbeddingProvider.HUGGINGFACE:
            embeddings = self._embed_huggingface(sanitised, show_progress)
        else:
            embeddings = self._embed_google(sanitised)

        latency_ms = time.monotonic() * 1000 - start_ms

        # Validate output shape
        if embeddings.shape != (len(sanitised), self._dimension):
            raise EmbeddingError(
                provider=self._provider.value,
                reason=(
                    f"Unexpected embedding shape {embeddings.shape}. "
                    f"Expected ({len(sanitised)}, {self._dimension})."
                ),
            )

        estimated_tokens = self._estimate_tokens(sanitised)

        logger.debug(
            f"Embedded {len(texts)} texts in {latency_ms:.1f}ms. "
            f"shape={embeddings.shape}, "
            f"~{estimated_tokens} tokens"
        )

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._model_name,
            provider=self._provider,
            dimension=self._dimension,  # type: ignore[arg-type]
            text_count=len(texts),
            latency_ms=latency_ms,
            estimated_token_count=estimated_tokens,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns 1D array of shape (dim,).

        Convenience wrapper around embed() for the common case of
        embedding a single user query at retrieval time.

        The query is treated differently from documents in some models
        (e.g. asymmetric search). HuggingFace uses the same encoder
        for both — Google uses task_type="retrieval_query" vs
        task_type="retrieval_document" for asymmetric embedding.

        Returns:
            1D np.ndarray of shape (embedding_dim,).
        """
        if not query or not query.strip():
            raise ValueError(
                "embed_query() received an empty query string."
            )

        if self._provider == EmbeddingProvider.GOOGLE:
            # Use query-specific task type for asymmetric search
            return self._embed_google_query(query)

        result = self.embed([query], show_progress=False)
        return result.embeddings[0]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_huggingface(
        self,
        texts: list[str],
        show_progress: bool,
    ) -> np.ndarray:
        """
        Run SentenceTransformer inference in batches.

        Processes texts in _HF_BATCH_SIZE chunks to avoid OOM on
        large ingestion jobs. Results are concatenated into a single
        array before returning.
        """
        if self._model is None:
            raise EmbeddingError(
                provider="huggingface",
                reason="Model not loaded. Call _load_model() first.",
            )

        all_embeddings: list[np.ndarray] = []

        for batch_start in range(0, len(texts), self._HF_BATCH_SIZE):
            batch = texts[batch_start: batch_start + self._HF_BATCH_SIZE]
            try:
                batch_embeddings: np.ndarray = (
                    self._model.encode(  # type: ignore[union-attr]
                        batch,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                        normalize_embeddings=True,   # L2 normalise for cosine sim
                    )
                )
                all_embeddings.append(batch_embeddings)
            except Exception as exc:
                raise EmbeddingError(
                    provider="huggingface",
                    reason=f"Inference failed on batch starting at index "
                           f"{batch_start}: {exc}",
                    original_exception=exc,
                ) from exc

        return np.vstack(all_embeddings)

    def _embed_google(self, texts: list[str]) -> np.ndarray:
        """
        Call Google embedding API in batches.

        Google recommends max 100 texts per request.
        Uses task_type="retrieval_document" for document indexing.
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise EmbeddingError(
                provider="google",
                reason="google-generativeai not installed.",
                original_exception=exc,
            ) from exc

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self._GOOGLE_BATCH_SIZE):
            batch = texts[batch_start: batch_start + self._GOOGLE_BATCH_SIZE]
            try:
                result = genai.embed_content(
                    model=self._model_name,
                    content=batch,
                    task_type="retrieval_document",
                )
                # result["embedding"] is list[float] for single,
                # list[list[float]] for batch
                raw = result["embedding"]
                if isinstance(raw[0], float):
                    # Single text — wrap in list
                    all_embeddings.append(raw)
                else:
                    all_embeddings.extend(raw)

            except Exception as exc:
                raise EmbeddingError(
                    provider="google",
                    reason=f"API call failed on batch starting at index "
                           f"{batch_start}: {exc}. "
                           f"Check rate limits and API key.",
                    original_exception=exc,
                ) from exc

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_google_query(self, query: str) -> np.ndarray:
        """
        Embed a single query using Google's query-specific task type.

        Google's text-embedding-004 supports asymmetric search:
          - Documents use task_type="retrieval_document"
          - Queries use task_type="retrieval_query"
        This improves retrieval quality vs using the same task type
        for both. HuggingFace models are typically symmetric.
        """
        try:
            import google.generativeai as genai

            result = genai.embed_content(
                model=self._model_name,
                content=query,
                task_type="retrieval_query",
            )
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(
                provider="google",
                reason=f"Query embedding API call failed: {exc}",
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sanitise_texts(
        self,
        texts: list[str],
    ) -> tuple[list[str], list[int]]:
        """
        Replace empty strings with a single space.

        Returns:
            (sanitised_texts, indices_of_empty_strings)
        """
        sanitised: list[str] = []
        empty_indices: list[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning(
                    f"Empty string at index {i} in embed() call. "
                    f"Replacing with single space to avoid model error."
                )
                sanitised.append(" ")
                empty_indices.append(i)
            else:
                sanitised.append(text)

        return sanitised, empty_indices

    def _estimate_tokens(self, texts: list[str]) -> int:
        """
        Approximate total token count for cost estimation.

        Uses character count / 4 as a conservative estimate.
        Not used for billing — only for dashboard cost display.
        """
        total_chars = sum(len(t) for t in texts)
        return max(1, int(total_chars / self._CHARS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """
        Embedding vector dimension.

        Set at model load time. Read by ChromaVectorStore to validate
        that incoming embeddings match the collection's dimension.

        Raises RuntimeError if called before model is loaded
        (should never happen — _load_model() is called in __init__).
        """
        if self._dimension is None:
            raise RuntimeError(
                "EmbeddingManager.dimension accessed before model was loaded. "
                "This is a bug — _load_model() should have set _dimension."
            )
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"EmbeddingManager("
            f"provider={self._provider.value}, "
            f"model='{self._model_name}', "
            f"dimension={self._dimension})"
        )

# ===========================================================================
# CUSTOM EXCEPTION
# ===========================================================================

class EmbeddingError(Exception):
    """
    Raised when an embedding operation fails unrecoverably.
    
    Distinct from ValueError (bad input) and RuntimeError (bug).
    EmbeddingError means: "the input was valid but the model or
    API failed to produce embeddings."

    The IngestionPipeline catches this and records the failure
    without crashing the entire ingestion job.
    """
    def __init__(
        self,
        provider: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[EmbeddingManager/{provider}] {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingManager",
    "EmbeddingError",
]