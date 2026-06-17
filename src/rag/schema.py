"""
src/rag/schema.py

Pydantic schemas for the RAG pipeline layer.

These schemas represent the inputs and outputs of the RAG pipeline:
  - RAGRequest    — one question to answer (input)
  - RAGResponse   — one answered question with context (output)
  - RAGBatchResult— results from running a full dataset through RAG
  - ModelRunConfig— configuration for one model run in a comparison

Data flow:
    QAPair (status=PENDING)
        ↓ RAGPipeline.answer_pair()
        ↓ RAGResponse
        ↓ pair.set_answer()
    QAPair (status=ANSWERED)
        ↓ EvaluationEngine.arun()
        ↓ EvalReport

Design rules:
  - All schemas are frozen (immutable after construction).
  - RAGResponse carries enough information for the evaluation engine
    to score all 4 metrics without re-querying the vector store.
  - No schema here imports from src/evaluation — the RAG layer
    is upstream of evaluation and must not depend on it.
  - ModelRunConfig is the single source of truth for one model's
    parameters in a comparison run. The ComparisonRunner reads it
    and passes the correct values to both RAGPipeline and
    EvaluationEngine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ===========================================================================
# RAG REQUEST
# ===========================================================================

class RAGRequest(BaseModel):
    """
    A single question to be answered by the RAG pipeline.

    Wraps a question string with retrieval parameters so the pipeline
    can be called uniformly whether the source is a QAPair, a UI
    text box, or a batch job.

    Immutable — parameters are fixed at request creation time.
    The pipeline never modifies a request.
    """

    model_config = ConfigDict(frozen=True)

    question: Annotated[
        str,
        Field(
            min_length=3,
            description="The question to answer via RAG.",
        ),
    ]

    collection_name: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "ChromaDB collection to retrieve from. "
                "Must exist in the vector store."
            ),
        ),
    ]

    top_k: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=20,
            description=(
                "Number of context chunks to retrieve. "
                "Higher values provide more context but increase "
                "prompt size and cost."
            ),
        ),
    ] = 5

    score_threshold: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description=(
                "Minimum similarity score for retrieved chunks. "
                "0.0 = no filtering. "
                "0.5 = only retrieve chunks with >50% similarity."
            ),
        ),
    ] = 0.0

    temperature: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=2.0,
            description=(
                "Generation temperature. "
                "0.0 = deterministic (recommended for evaluation). "
                "Higher values produce more varied answers."
            ),
        ),
    ] = 0.0

    max_output_tokens: Annotated[
        int,
        Field(
            default=1024,
            ge=64,
            le=8192,
            description="Maximum tokens in the generated answer.",
        ),
    ] = 1024

    pair_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "QAPair ID this request was created from. "
                "Set by answer_pair() for traceability. "
                "None for ad-hoc queries."
            ),
        ),
    ] = None

    metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Optional key-value metadata attached to the request. "
                "Passed through to RAGResponse for logging."
            ),
        ),
    ] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RAGRequest("
            f"question='{self.question[:50]}...', "
            f"collection='{self.collection_name}', "
            f"top_k={self.top_k})"
        )

# ===========================================================================
# RETRIEVED CHUNK
# ===========================================================================

class RetrievedChunk(BaseModel):
    """
    A single chunk retrieved from the vector store.

    Richer than a plain string — carries source metadata so the
    evaluation engine can populate EvalResult.retrieved_chunk_sources
    and the dashboard can show which document/page each chunk came from.

    Frozen — retrieved content is immutable.
    """

    model_config = ConfigDict(frozen=True)

    content: Annotated[
        str,
        Field(description="Text content of the retrieved chunk."),
    ]

    source_file: Annotated[
        str,
        Field(description="Filename of the source document."),
    ]

    source_page: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Page number within the source document. "
                "None for formats without page structure."
            ),
        ),
    ] = None

    chunk_index: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="0-based chunk index within the source document.",
        ),
    ] = None

    similarity_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Cosine similarity score from vector search. "
                "1.0 = identical, 0.0 = unrelated."
            ),
        ),
    ]

    rank: Annotated[
        int,
        Field(
            ge=1,
            description="1-based rank in the retrieval result list.",
        ),
    ]

    collection_name: str = Field(
        description="ChromaDB collection this chunk was retrieved from."
    )

    @property
    def citation(self) -> str:
        """
        Human-readable citation string for this chunk.

        Used in generated answers and evaluation dashboard:
        "policy.pdf p.21" or "report.txt"
        """
        if self.source_page:
            return f"{self.source_file} p.{self.source_page}"
        return self.source_file

    @property
    def as_numbered_entry(self) -> str:
        """
        Format as a numbered entry for judge prompts.

        "1. [policy.pdf p.21] chunk content here..."
        """
        return f"[{self.citation}] {self.content}"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk("
            f"source='{self.citation}', "
            f"score={self.similarity_score:.3f}, "
            f"rank={self.rank})"
        )

# ===========================================================================
# RAG RESPONSE
# ===========================================================================

class RAGResponse(BaseModel):
    """
    The output of the RAG pipeline for a single question.

    Contains:
      - generated_answer    — the LLM's answer given the context
      - retrieved_chunks    — the context used to generate the answer
      - performance data    — latency, tokens, cost estimate
      - quality signals     — empty_context, answer_refused

    RAGResponse is the handoff object between RAGPipeline and
    EvaluationEngine. The engine reads retrieved_chunks to construct
    judge prompts — it never re-queries the vector store.

    Frozen — pipeline outputs are immutable.
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    request: RAGRequest

    rag_model: Annotated[
        str,
        Field(description="Model ID used to generate the answer."),
    ]

    answered_at: Annotated[
        datetime,
        Field(description="UTC timestamp when answer was generated."),
    ]

    # ------------------------------------------------------------------
    # Core output
    # ------------------------------------------------------------------

    generated_answer: Annotated[
        str,
        Field(
            description=(
                "The LLM-generated answer. "
                "May be empty if answer_refused=True or "
                "context was empty."
            )
        ),
    ]

    retrieved_chunks: Annotated[
        list[RetrievedChunk],
        Field(
            default_factory=list,
            description=(
                "Retrieved context chunks, sorted by similarity score. "
                "Empty if no relevant chunks were found."
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # Performance data
    # ------------------------------------------------------------------

    retrieval_latency_ms: Annotated[
        float,
        Field(
            ge=0.0,
            description="Time spent on vector store query.",
        ),
    ] = 0.0

    generation_latency_ms: Annotated[
        float,
        Field(
            ge=0.0,
            description="Time spent on LLM generation.",
        ),
    ] = 0.0

    total_latency_ms: Annotated[
        float,
        Field(
            ge=0.0,
            description="Total wall-clock time for this response.",
        ),
    ] = 0.0

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)

    # ------------------------------------------------------------------
    # Quality signals
    # ------------------------------------------------------------------

    empty_context: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "True if no chunks were retrieved or all were below "
                "score_threshold. The answer was generated without "
                "any context — unreliable and likely to score low "
                "on Faithfulness and Context Precision."
            ),
        ),
    ] = False

    answer_refused: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "True if the LLM refused to answer "
                "(safety filter, content policy, or explicit refusal). "
                "Generated answer will be empty or contain a refusal message."
            ),
        ),
    ] = False

    prompt_used: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "The full prompt sent to the LLM. "
                "Stored for debugging and prompt inspection. "
                "None if prompt logging is disabled."
            ),
        ),
    ] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def total_latency_is_sum(self) -> RAGResponse:
        """
        Warn if total_latency_ms is less than the sum of its parts.

        Wall-clock time should always be >= retrieval + generation.
        If it is significantly less, the caller likely forgot to include
        overhead (embedding latency, prompt construction, etc.).
        """
        expected_min = (
            self.retrieval_latency_ms + self.generation_latency_ms
        )
        if (
            expected_min > 0
            and self.total_latency_ms < expected_min * 0.9
        ):
            import warnings
            warnings.warn(
                f"RAGResponse: total_latency_ms={self.total_latency_ms:.1f} "
                f"is less than retrieval + generation = "
                f"{expected_min:.1f}ms. "
                f"Ensure total_latency_ms is measured as wall-clock time.",
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def question(self) -> str:
        """Convenience accessor — avoids .request.question chains."""
        return self.request.question

    @property
    def collection_name(self) -> str:
        return self.request.collection_name

    @property
    def top_k(self) -> int:
        return self.request.top_k

    @property
    def n_retrieved(self) -> int:
        """Actual number of chunks retrieved (may be < top_k)."""
        return len(self.retrieved_chunks)

    @property
    def chunk_contents(self) -> list[str]:
        """
        Plain text content of all retrieved chunks.

        Used by pair.set_answer() to populate QAPair.retrieved_chunks.
        """
        return [chunk.content for chunk in self.retrieved_chunks]

    @property
    def chunk_sources(self) -> list[str]:
        """
        Citation strings for all retrieved chunks.

        Used by pair.set_answer() to populate
        QAPair.retrieved_chunk_sources.
        """
        return [chunk.citation for chunk in self.retrieved_chunks]

    @property
    def context_as_numbered_list(self) -> str:
        """
        Format retrieved chunks as a numbered list.

        Identical format to QAPair.context_as_numbered_list.
        Used to build the RAG prompt and verify context content.
        """
        if not self.retrieved_chunks:
            return "No context retrieved."
        return "\n\n".join(
            f"{chunk.rank}. {chunk.as_numbered_entry}"
            for chunk in self.retrieved_chunks
        )

    @property
    def avg_similarity_score(self) -> float:
        """
        Average similarity score across retrieved chunks.

        Quality signal: low average score suggests the question
        is outside the knowledge base coverage.
        """
        if not self.retrieved_chunks:
            return 0.0
        return sum(
            c.similarity_score for c in self.retrieved_chunks
        ) / len(self.retrieved_chunks)

    @property
    def best_chunk_score(self) -> float:
        """Highest similarity score in retrieved chunks."""
        if not self.retrieved_chunks:
            return 0.0
        return max(c.similarity_score for c in self.retrieved_chunks)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_reliable(self) -> bool:
        """
        True if the response is unlikely to have quality issues.

        Unreliable if:
          - No context was retrieved (empty_context=True)
          - LLM refused to answer (answer_refused=True)
          - Answer is empty
        """
        return (
            not self.empty_context
            and not self.answer_refused
            and bool(self.generated_answer.strip())
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Serialise to a flat dict for logging and CSV export.
        """
        return {
            "question":             self.question,
            "collection_name":      self.collection_name,
            "top_k":                self.top_k,
            "rag_model":            self.rag_model,
            "answered_at":          self.answered_at.isoformat(),
            "generated_answer":     self.generated_answer,
            "n_retrieved":          self.n_retrieved,
            "avg_similarity_score": round(self.avg_similarity_score, 4),
            "best_chunk_score":     round(self.best_chunk_score, 4),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 1),
            "generation_latency_ms":round(self.generation_latency_ms, 1),
            "total_latency_ms":     round(self.total_latency_ms, 1),
            "input_tokens":         self.input_tokens,
            "output_tokens":        self.output_tokens,
            "estimated_cost_usd":   self.estimated_cost_usd,
            "empty_context":        self.empty_context,
            "answer_refused":       self.answer_refused,
            "is_reliable":          self.is_reliable,
            "pair_id":              self.request.pair_id,
        }

    def __repr__(self) -> str:
        return (
            f"RAGResponse("
            f"model='{self.rag_model}', "
            f"n_retrieved={self.n_retrieved}, "
            f"latency={self.total_latency_ms:.0f}ms, "
            f"reliable={self.is_reliable})"
        )

# ===========================================================================
# RAG BATCH RESULT
# ===========================================================================

class RAGBatchResult(BaseModel):
    """
    Result of running an entire EvalDataset through the RAG pipeline.

    Produced by RAGPipeline.answer_dataset().
    Passed to EvaluationEngine.arun() after all pairs are answered.

    Stores both the modified dataset (pairs now ANSWERED) and
    aggregate performance stats for dashboard display.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    rag_model: str
    collection_name: str
    top_k: int
    temperature: float

    total_pairs: int = Field(ge=0)
    answered_pairs: int = Field(ge=0)
    failed_pairs: int = Field(ge=0)
    skipped_pairs: int = Field(ge=0)

    # Timing
    total_latency_ms: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)

    # Tokens and cost
    total_input_tokens: int = Field(ge=0, default=0)
    total_output_tokens: int = Field(ge=0, default=0)
    total_estimated_cost_usd: float = Field(ge=0.0, default=0.0)

    # Quality signals
    empty_context_count: int = Field(
        ge=0,
        default=0,
        description=(
            "Number of pairs where no context was retrieved. "
            "High counts indicate the question set does not match "
            "the indexed document collection."
        ),
    )

    refused_count: int = Field(
        ge=0,
        default=0,
        description="Number of pairs where the LLM refused to answer.",
    )

    avg_similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Mean similarity score across all retrieved chunks.",
    )

    responses: Annotated[
        list[RAGResponse],
        Field(
            default_factory=list,
            description=(
                "One RAGResponse per answered pair. "
                "Parallel list to the dataset's answered pairs."
            ),
        ),
    ] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.answered_pairs / self.total_pairs

    @property
    def empty_context_rate(self) -> float:
        if self.answered_pairs == 0:
            return 0.0
        return self.empty_context_count / self.answered_pairs

    @property
    def is_healthy(self) -> bool:
        """
        True if the batch result has no major quality concerns.

        Unhealthy if:
          - More than 20% of pairs had empty context
          - More than 10% of pairs failed entirely
          - Success rate < 80%
        """
        return (
            self.success_rate >= 0.8
            and self.empty_context_rate <= 0.2
            and self.failed_pairs / max(self.total_pairs, 1) <= 0.1
        )
        
    def summary(self) -> dict[str, Any]:
        """Flat dict for logging and dashboard summary cards."""
        return {
            "rag_model":              self.rag_model,
            "collection_name":        self.collection_name,
            "top_k":                  self.top_k,
            "temperature":            self.temperature,
            "total_pairs":            self.total_pairs,
            "answered_pairs":         self.answered_pairs,
            "failed_pairs":           self.failed_pairs,
            "success_rate":           round(self.success_rate, 4),
            "empty_context_count":    self.empty_context_count,
            "empty_context_rate":     round(self.empty_context_rate, 4),
            "refused_count":          self.refused_count,
            "avg_latency_ms":         round(self.avg_latency_ms, 1),
            "total_latency_ms":       round(self.total_latency_ms, 1),
            "total_input_tokens":     self.total_input_tokens,
            "total_output_tokens":    self.total_output_tokens,
            "total_cost_usd":         round(
                self.total_estimated_cost_usd, 6
            ),
            "avg_similarity_score":   round(self.avg_similarity_score, 4),
            "is_healthy":             self.is_healthy,
        }

    def __repr__(self) -> str:
        return (
            f"RAGBatchResult("
            f"model='{self.rag_model}', "
            f"answered={self.answered_pairs}/{self.total_pairs}, "
            f"success_rate={self.success_rate:.0%})"
        )

# ===========================================================================
# MODEL RUN CONFIG
# ===========================================================================

class ModelRunConfig(BaseModel):
    """
    Configuration for one model run in a multi-model comparison.

    ComparisonRunner creates one ModelRunConfig per (model × parameter)
    combination from the comparison_grid in models.yaml.

    Immutable — run configurations are fixed before execution starts.
    The runner never modifies a config mid-run.

    Example:
        ModelRunConfig(
            model_id="gemini-1.5-flash",
            collection_name="q3_report",
            top_k=5,
            temperature=0.0,
        )
    """

    model_config = ConfigDict(frozen=True)

    model_id: Annotated[
        str,
        Field(
            description=(
                "Model ID from models.yaml. "
                "Must match an enabled model in the registry."
            )
        ),
    ]

    collection_name: Annotated[
        str,
        Field(description="ChromaDB collection to retrieve from."),
    ]

    top_k: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=20,
            description="Number of context chunks to retrieve.",
        ),
    ] = 5

    temperature: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=2.0,
            description="RAG model generation temperature.",
        ),
    ] = 0.0

    max_output_tokens: Annotated[
        int,
        Field(
            default=1024,
            ge=64,
            le=8192,
        ),
    ] = 1024

    score_threshold: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Minimum similarity score for retrieval.",
        ),
    ] = 0.0

    run_label: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional human-readable label for this run config. "
                "Displayed in the dashboard instead of the model ID "
                "when set. Useful for distinguishing runs with the "
                "same model but different parameters: "
                "'flash-top3' vs 'flash-top8'."
            ),
        ),
    ] = None

    @property
    def display_label(self) -> str:
        """
        Human-readable label for dashboard display.

        Uses run_label if set, otherwise constructs from model_id
        and key parameters.
        """
        if self.run_label:
            return self.run_label
        return f"{self.model_id} (k={self.top_k}, t={self.temperature})"

    @property
    def config_hash(self) -> str:
        """
        Short hash of this config for deduplication.

        Two configs with the same model_id, top_k, temperature, and
        collection produce the same hash — used to detect duplicate
        runs in the comparison grid before execution.
        """
        import hashlib
        fingerprint = (
            f"{self.model_id}|{self.collection_name}|"
            f"{self.top_k}|{self.temperature}|{self.score_threshold}"
        )
        return hashlib.sha256(
            fingerprint.encode()
        ).hexdigest()[:12]

    def to_rag_request(self, question: str, pair_id: str | None = None) -> RAGRequest:
        """
        Build a RAGRequest from this config for a given question.

        Convenience factory used by RAGPipeline.answer_pair().
        """
        return RAGRequest(
            question=question,
            collection_name=self.collection_name,
            top_k=self.top_k,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            score_threshold=self.score_threshold,
            pair_id=pair_id,
        )

    def __repr__(self) -> str:
        return (
            f"ModelRunConfig("
            f"model='{self.model_id}', "
            f"collection='{self.collection_name}', "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature})"
        )

# ===========================================================================
# PROMPT SCHEMA
# ===========================================================================

class RAGPromptComponents(BaseModel):
    """
    Structured representation of the RAG generation prompt.

    Separating prompt components into a typed object allows:
      - Logging the prompt without mixing it into response fields.
      - Testing prompt construction independently of LLM calls.
      - Inspecting what context was injected via the UI drilldown.

    Frozen — prompt is constructed once per request.
    """

    model_config = ConfigDict(frozen=True)

    system_instruction: Annotated[
        str,
        Field(description="System instruction for the RAG model."),
    ]

    context_block: Annotated[
        str,
        Field(
            description=(
                "Formatted retrieved chunks block injected "
                "into the prompt."
            )
        ),
    ]

    question: str

    full_prompt: Annotated[
        str,
        Field(
            description=(
                "Complete prompt string sent to the LLM. "
                "Concatenation of system_instruction + "
                "context_block + question."
            )
        ),
    ]

    @property
    def estimated_token_count(self) -> int:
        """Approximate token count: chars / 4."""
        return max(1, len(self.full_prompt) // 4)

    def __repr__(self) -> str:
        return (
            f"RAGPromptComponents("
            f"context_chunks={self.context_block.count(chr(10) + chr(10))}, "
            f"prompt_length={len(self.full_prompt)} chars, "
            f"~{self.estimated_token_count} tokens)"
        )

__all__ = [
    "RAGRequest",
    "RetrievedChunk",
    "RAGResponse",
    "RAGBatchResult",
    "ModelRunConfig",
    "RAGPromptComponents",
]