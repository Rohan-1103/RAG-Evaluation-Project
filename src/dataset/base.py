"""
src/dataset/base.py

Abstract base class for all test dataset generators.

Design contract:
  - Every generator receives document chunks and produces QAPairs.
  - No generator touches the vector store, evaluation engine, or RAG
    pipeline. Single responsibility: chunks → Q&A pairs.
  - The ABC defines the full generation interface. GeminiDatasetGenerator
    and any future generator (GPT-4o, manual, CSV import) must satisfy
    every method here.
  - Generation is inherently non-deterministic (LLM output varies).
    The ABC enforces a seed/temperature contract so callers can request
    more deterministic output when needed (e.g. regression testing).

Why abstract the generator:
  - The EvaluationEngine and DatasetStore depend on EvalDataset, not
    on how it was created. Swapping Gemini for GPT-4o as the generator
    requires zero changes in any downstream module.
  - Unit tests mock BaseDatasetGenerator to return fixture QAPairs
    without making any API calls.
  - Future generators (e.g. a human-annotation interface, a CSV importer)
    implement this interface and plug in without pipeline changes.

Generation quality contract:
  Every generated QAPair must satisfy:
    1. question  — answerable from source_chunk alone
    2. ground_truth_answer — derived only from source_chunk content
    3. source_chunk — the exact chunk text used for generation
    4. source_file  — filename of the source document
    5. status = QAPairStatus.PENDING (never pre-answered)

  Generators must NOT:
    - Generate questions that require cross-chunk reasoning
      (those are valid eval questions but cannot be auto-generated
      from a single chunk reliably)
    - Hallucinate answers beyond the source chunk content
    - Generate duplicate questions within one dataset
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.dataset.schema import (
    EvalDataset,
    GenerationMethod,
    QAPair,
    create_eval_dataset,
    make_pair_id,
)
from src.ingestion.base import Document

# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass(frozen=True)
class GeneratorConfig:
    """
    Configuration for a dataset generation run.

    Frozen — a generator's run config is immutable. To vary parameters,
    construct a new GeneratorConfig.

    Fields:
      n_pairs_per_chunk   — how many Q&A pairs to attempt per chunk.
                            Not all chunks produce n pairs (short chunks
                            may only yield 1 good question).
      min_question_length — reject generated questions shorter than this.
      min_answer_length   — reject generated answers shorter than this.
      max_retries_per_chunk — how many times to retry a chunk if the LLM
                              returns malformed JSON or empty output.
      deduplicate         — if True, discard questions that are too similar
                            to already-generated questions in this run.
      similarity_threshold — cosine similarity above which two questions
                             are considered duplicates (0.0-1.0).
                             Only used if deduplicate=True.
      filter_short_chunks — skip chunks shorter than min_chunk_length.
                            Very short chunks rarely produce good questions.
      min_chunk_length    — minimum character count for a chunk to be
                            considered for generation.
      temperature         — LLM temperature for generation.
                            Slightly above 0 produces varied questions.
                            0.0 produces more formulaic but reproducible output.
      max_pairs_total     — hard cap on total pairs generated regardless
                            of how many chunks are available. Prevents
                            accidentally generating 10,000 pairs on large
                            corpora during testing.
    """

    n_pairs_per_chunk: Annotated[int, "Pairs attempted per chunk"] = 3
    min_question_length: int = 15
    min_answer_length: int = 10
    max_retries_per_chunk: int = 2
    deduplicate: bool = True
    similarity_threshold: float = 0.85
    filter_short_chunks: bool = True
    min_chunk_length: int = 100
    max_pairs_total: int = 200
    temperature: float = 0.4

    def __post_init__(self) -> None:
        if self.n_pairs_per_chunk < 1 or self.n_pairs_per_chunk > 10:
            raise ValueError(
                f"n_pairs_per_chunk must be in [1, 10], "
                f"got {self.n_pairs_per_chunk}."
            )
        if self.min_chunk_length < 50:
            raise ValueError(
                f"min_chunk_length must be >= 50, "
                f"got {self.min_chunk_length}. "
                f"Chunks shorter than 50 chars rarely contain "
                f"enough content for a meaningful question."
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], "
                f"got {self.similarity_threshold}."
            )
        if self.max_pairs_total < 1:
            raise ValueError(
                f"max_pairs_total must be >= 1, "
                f"got {self.max_pairs_total}."
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], "
                f"got {self.temperature}."
            )

# ===========================================================================
# RESULT TYPES
# ===========================================================================

@dataclass
class ChunkGenerationResult:
    """
    Result of generating Q&A pairs from a single chunk.

    Granular per-chunk result so the caller can identify which chunks
    produced good pairs, which were skipped, and which failed —
    without parsing log messages.
    """

    chunk_index: int
    source_file: str
    source_page: int | None
    status: str                         # "success" | "skipped" | "failed"
    pairs_generated: int = 0
    pairs_rejected: int = 0             # Failed quality checks
    error_message: str | None = None
    retry_count: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    def __repr__(self) -> str:
        return (
            f"ChunkGenerationResult("
            f"chunk={self.chunk_index}, "
            f"source='{self.source_file}', "
            f"status={self.status}, "
            f"pairs={self.pairs_generated})"
        )


@dataclass
class GenerationStats:
    """
    Aggregate statistics from a complete dataset generation run.

    Returned alongside the EvalDataset so callers have full
    observability without re-scanning the pair list.
    """

    chunks_attempted: int = 0
    chunks_succeeded: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    pairs_generated: int = 0
    pairs_rejected: int = 0             # Failed quality/dedup checks
    pairs_deduplicated: int = 0         # Removed as too similar
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    chunk_results: list[ChunkGenerationResult] = field(
        default_factory=list
    )

    @property
    def success_rate(self) -> float:
        if self.chunks_attempted == 0:
            return 0.0
        return self.chunks_succeeded / self.chunks_attempted

    @property
    def avg_pairs_per_chunk(self) -> float:
        if self.chunks_succeeded == 0:
            return 0.0
        return self.pairs_generated / self.chunks_succeeded

    @property
    def avg_latency_per_chunk_ms(self) -> float:
        if self.chunks_attempted == 0:
            return 0.0
        return self.total_latency_ms / self.chunks_attempted

    @property
    def estimated_cost_usd(self) -> float:
        """
        Placeholder — actual cost computed by the generator using
        the model's cost_per_1k from models.yaml.
        """
        return 0.0

    def __repr__(self) -> str:
        return (
            f"GenerationStats("
            f"chunks={self.chunks_succeeded}/{self.chunks_attempted}, "
            f"pairs={self.pairs_generated}, "
            f"rejected={self.pairs_rejected}, "
            f"latency={self.total_latency_ms:.0f}ms)"
        )

# ===========================================================================
# RAW PAIR (intermediate representation)
# ===========================================================================

class RawQAPair(BaseModel):
    """
    Intermediate representation of a Q&A pair as returned by the LLM.

    The generator parses LLM JSON output into RawQAPair first,
    then validates and promotes to QAPair. This two-step approach:
      1. Isolates JSON parsing failures from schema validation failures.
      2. Allows the generator to apply quality filters (min length,
         duplicate detection) before constructing the full QAPair.
      3. Makes the generator's parsing logic testable independently
         of the full QAPair schema.

    Fields are all Optional because LLM output may be partially malformed.
    The generator validates completeness before promoting.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    question: str | None = None
    answer: str | None = None

    @property
    def is_complete(self) -> bool:
        """True if both fields are non-empty strings."""
        return (
            bool(self.question and self.question.strip())
            and bool(self.answer and self.answer.strip())
        )

    @property
    def is_valid_length(
        self,
        min_q: int = 15,
        min_a: int = 10,
    ) -> bool:
        """True if both fields meet minimum length requirements."""
        return (
            len(self.question or "") >= min_q
            and len(self.answer or "") >= min_a
        )

    def validate_lengths(
        self,
        min_question_length: int,
        min_answer_length: int,
    ) -> tuple[bool, str]:
        """
        Validate question and answer lengths.

        Returns (is_valid, rejection_reason).
        rejection_reason is empty string when valid.
        """
        if not self.question or not self.question.strip():
            return False, "Empty question"

        if not self.answer or not self.answer.strip():
            return False, "Empty answer"

        if len(self.question.strip()) < min_question_length:
            return False, (
                f"Question too short: {len(self.question.strip())} chars "
                f"(min {min_question_length})"
            )

        if len(self.answer.strip()) < min_answer_length:
            return False, (
                f"Answer too short: {len(self.answer.strip())} chars "
                f"(min {min_answer_length})"
            )

        return True, ""

# ===========================================================================
# ABSTRACT BASE CLASS
# ===========================================================================

class BaseDatasetGenerator(ABC):
    """
    Abstract interface for all test dataset generators.

    Subclasses implement:
      _generate_from_chunk  — core LLM call for one chunk
      generation_method     — declares what type of generation this is
      model_name            — which LLM this generator uses

    Subclasses must NOT implement:
      generate()            — orchestration (defined here as template method)
      _filter_chunks        — chunk pre-filtering (defined here)
      _deduplicate_pairs    — duplicate detection (defined here)
      _build_dataset        — EvalDataset construction (defined here)

    The template method pattern is applied to generate():
      - Common concerns (filtering, dedup, stats, dataset construction)
        are handled once here.
      - LLM-specific logic lives only in _generate_from_chunk().
      - Adding a new generator means implementing one method.

    Usage:
        generator = GeminiDatasetGenerator(
            config=GeneratorConfig(n_pairs_per_chunk=3),
            settings=get_settings(),
        )
        dataset, stats = generator.generate(
            chunks=chunks,
            dataset_name="Q3 Financial Eval",
            source_collection="q3_report",
        )
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Abstract interface — implement in subclasses
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def generation_method(self) -> GenerationMethod:
        """
        Declare what type of generation this class performs.

        Return GenerationMethod.SYNTHETIC for LLM-based generators.
        Return GenerationMethod.IMPORTED for CSV/JSON importers.
        Return GenerationMethod.MANUAL for human-annotation interfaces.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the model ID used by this generator.

        For LLM-based generators: "gemini-1.5-flash", "gpt-4o-mini", etc.
        For non-LLM generators: a descriptive string like "csv_importer".
        Used in DatasetMetadata.generator_model.
        """
        ...

    @abstractmethod
    def _generate_from_chunk(
        self,
        chunk: Document,
        n_pairs: int,
        dataset_id: str,
    ) -> tuple[list[RawQAPair], ChunkGenerationResult]:
        """
        Generate n_pairs Q&A pairs from a single document chunk.

        Args:
            chunk:      Document chunk to generate questions from.
                        chunk.page_content is the text to use.
            n_pairs:    Target number of pairs to generate.
                        The generator should attempt this many but
                        may return fewer if the chunk is short or
                        if the LLM produces low-quality output.
            dataset_id: ID of the dataset being constructed.
                        Set on all returned RawQAPairs' parent context.

        Returns:
            (raw_pairs, chunk_result) tuple:
              raw_pairs    — parsed but not yet validated QAPair candidates
              chunk_result — ChunkGenerationResult with status and stats

        Contract:
          - Never raises — all exceptions are caught and returned
            as chunk_result.status = "failed" with error_message set.
          - raw_pairs may be empty (short chunk, bad LLM response, etc.)
          - chunk_result.status must be "success", "skipped", or "failed"
          - chunk_result.latency_ms, input_tokens, output_tokens must
            be set for cost tracking.

        Implementations must NOT:
          - Call generate() recursively.
          - Persist anything to disk or database.
          - Call the vector store or evaluation engine.
        """
        ...

    # ------------------------------------------------------------------
    # Template method — do not override in subclasses
    # ------------------------------------------------------------------

    def generate(
        self,
        chunks: list[Document],
        dataset_name: str,
        source_collection: str | None = None,
    ) -> tuple[EvalDataset, GenerationStats]:
        """
        Generate an EvalDataset from a list of document chunks.

        This is the public interface. Do not override in subclasses.
        Subclasses implement _generate_from_chunk() only.

        Args:
            chunks:            Document chunks from RecursiveChunker.
            dataset_name:      Human-readable name for the dataset.
            source_collection: ChromaDB collection name (for metadata).

        Returns:
            (dataset, stats) tuple:
              dataset — fully constructed EvalDataset with all pairs
              stats   — GenerationStats for dashboard display

        Raises:
            ValueError: if chunks is empty.
            GenerationError: if generation fails so completely that
                             no pairs at all could be produced.
        """
        from loguru import logger

        if not chunks:
            raise ValueError(
                "BaseDatasetGenerator.generate() received empty chunks list. "
                "Run the ingestion pipeline first to produce chunks."
            )

        logger.info(
            f"{self.__class__.__name__}: Starting generation. "
            f"chunks={len(chunks)}, "
            f"n_pairs_per_chunk={self._config.n_pairs_per_chunk}, "
            f"max_pairs_total={self._config.max_pairs_total}, "
            f"model={self.model_name}"
        )

        # Stage 1: Filter chunks
        eligible_chunks = self._filter_chunks(chunks)

        if not eligible_chunks:
            raise GenerationError(
                generator=self.__class__.__name__,
                reason=(
                    f"No eligible chunks after filtering. "
                    f"All {len(chunks)} chunks were shorter than "
                    f"min_chunk_length={self._config.min_chunk_length} chars. "
                    f"Consider reducing min_chunk_length in GeneratorConfig "
                    f"or using larger chunk_size in RecursiveChunker."
                ),
            )

        logger.info(
            f"{self.__class__.__name__}: "
            f"{len(eligible_chunks)}/{len(chunks)} chunks eligible "
            f"after length filtering."
        )

        # Stage 2: Generate pairs per chunk
        from src.dataset.schema import make_dataset_id

        dataset_id = make_dataset_id()
        all_raw_pairs: list[tuple[RawQAPair, Document]] = []
        stats = GenerationStats()

        for i, chunk in enumerate(eligible_chunks):
            # Check total cap
            if len(all_raw_pairs) >= self._config.max_pairs_total:
                logger.info(
                    f"{self.__class__.__name__}: Reached max_pairs_total="
                    f"{self._config.max_pairs_total}. "
                    f"Stopping generation early at chunk {i}/{len(eligible_chunks)}."
                )
                break

            remaining_quota = (
                self._config.max_pairs_total - len(all_raw_pairs)
            )
            n_pairs = min(self._config.n_pairs_per_chunk, remaining_quota)

            logger.debug(
                f"{self.__class__.__name__}: Generating from chunk "
                f"{i + 1}/{len(eligible_chunks)} "
                f"('{chunk.metadata.source_file}' "
                f"p={chunk.metadata.page_number})..."
            )

            raw_pairs, chunk_result = self._generate_from_chunk(
                chunk=chunk,
                n_pairs=n_pairs,
                dataset_id=dataset_id,
            )

            # Update stats
            stats.chunks_attempted += 1
            stats.total_latency_ms += chunk_result.latency_ms
            stats.total_input_tokens += chunk_result.input_tokens
            stats.total_output_tokens += chunk_result.output_tokens
            stats.chunk_results.append(chunk_result)

            if chunk_result.succeeded:
                stats.chunks_succeeded += 1
            elif chunk_result.skipped:
                stats.chunks_skipped += 1
                continue
            else:
                stats.chunks_failed += 1
                continue

            # Stage 3: Quality filter raw pairs
            valid_pairs, rejected_count = self._quality_filter(
                raw_pairs=raw_pairs,
                chunk=chunk,
            )
            stats.pairs_rejected += rejected_count

            for raw_pair in valid_pairs:
                all_raw_pairs.append((raw_pair, chunk))

        if not all_raw_pairs:
            raise GenerationError(
                generator=self.__class__.__name__,
                reason=(
                    f"No valid pairs generated from {len(eligible_chunks)} "
                    f"eligible chunks. "
                    f"Check that source documents contain sufficient "
                    f"factual content for question generation."
                ),
            )

        # Stage 4: Deduplicate
        if self._config.deduplicate and len(all_raw_pairs) > 1:
            all_raw_pairs, dedup_count = self._deduplicate_pairs(
                all_raw_pairs
            )
            stats.pairs_deduplicated = dedup_count
            logger.info(
                f"{self.__class__.__name__}: Deduplication removed "
                f"{dedup_count} similar pairs."
            )

        # Stage 5: Promote RawQAPairs → QAPairs
        qa_pairs = self._promote_to_qa_pairs(
            raw_pairs_with_chunks=all_raw_pairs,
            dataset_id=dataset_id,
        )

        stats.pairs_generated = len(qa_pairs)

        # Stage 6: Build EvalDataset
        source_files = list({
            chunk.metadata.source_file
            for _, chunk in all_raw_pairs
        })

        dataset = create_eval_dataset(
            name=dataset_name,
            pairs=qa_pairs,
            source_collection=source_collection,
            source_files=source_files,
            generator_model=self.model_name,
            generation_method=self.generation_method,
        )

        logger.info(
            f"{self.__class__.__name__}: Generation complete. "
            f"pairs={stats.pairs_generated}, "
            f"rejected={stats.pairs_rejected}, "
            f"deduped={stats.pairs_deduplicated}, "
            f"latency={stats.total_latency_ms:.0f}ms"
        )

        return dataset, stats

    # ------------------------------------------------------------------
    # Concrete helpers — used by template method, available to subclasses
    # ------------------------------------------------------------------

    def _filter_chunks(
        self,
        chunks: list[Document],
    ) -> list[Document]:
        """
        Filter out chunks that are too short for meaningful generation.

        Short chunks (headers, page numbers, captions) rarely contain
        enough content for a factual Q&A pair. Filtering them out
        reduces LLM calls and improves pair quality.
        """
        if not self._config.filter_short_chunks:
            return chunks

        eligible = [
            chunk for chunk in chunks
            if len(chunk.page_content.strip()) >= self._config.min_chunk_length
        ]
        return eligible

    def _quality_filter(
        self,
        raw_pairs: list[RawQAPair],
        chunk: Document,
    ) -> tuple[list[RawQAPair], int]:
        """
        Apply quality filters to raw pairs from a single chunk.

        Filters:
          1. Completeness — both question and answer present
          2. Minimum length — question and answer meet length thresholds
          3. Answer grounded — answer not longer than chunk
             (very long answers likely hallucinated beyond chunk content)

        Returns (valid_pairs, rejected_count).
        """
        valid: list[RawQAPair] = []
        rejected = 0

        for raw_pair in raw_pairs:
            if not raw_pair.is_complete:
                rejected += 1
                continue

            is_valid, reason = raw_pair.validate_lengths(
                min_question_length=self._config.min_question_length,
                min_answer_length=self._config.min_answer_length,
            )

            if not is_valid:
                rejected += 1
                continue

            # Heuristic: reject answers that are implausibly long
            # relative to source chunk — likely hallucinated
            chunk_len = len(chunk.page_content)
            answer_len = len(raw_pair.answer or "")
            if answer_len > chunk_len * 1.5:
                rejected += 1
                continue

            valid.append(raw_pair)

        return valid, rejected

    def _deduplicate_pairs(
        self,
        pairs_with_chunks: list[tuple[RawQAPair, Document]],
    ) -> tuple[list[tuple[RawQAPair, Document]], int]:
        """
        Remove near-duplicate questions using simple token overlap.

        Uses Jaccard similarity on question word sets as a lightweight
        deduplication heuristic. Avoids the overhead of embedding-based
        deduplication for this stage.

        Jaccard similarity = |A ∩ B| / |A ∪ B|
        Two questions are duplicates if their word-set Jaccard
        similarity exceeds self._config.similarity_threshold.

        More sophisticated deduplication (embedding cosine similarity)
        can be added by overriding this method in a subclass.
        """
        if len(pairs_with_chunks) <= 1:
            return pairs_with_chunks, 0

        kept: list[tuple[RawQAPair, Document]] = []
        removed_count = 0

        # Precompute word sets for all questions
        question_sets: list[set[str]] = [
            set((pair.question or "").lower().split())
            for pair, _ in pairs_with_chunks
        ]

        for i, (pair, chunk) in enumerate(pairs_with_chunks):
            is_duplicate = False
            q_set_i = question_sets[i]

            # Compare against all already-kept pairs
            for j, (kept_pair, _) in enumerate(kept):
                q_set_j = question_sets[
                    pairs_with_chunks.index((kept_pair, _))
                    if (kept_pair, _) in pairs_with_chunks
                    else 0
                ]
                # Recompute for kept pair to be safe
                q_set_j = set(
                    (kept_pair.question or "").lower().split()
                )

                union = q_set_i | q_set_j
                if not union:
                    continue

                intersection = q_set_i & q_set_j
                jaccard = len(intersection) / len(union)

                if jaccard >= self._config.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    break

            if not is_duplicate:
                kept.append((pair, chunk))

        return kept, removed_count

    def _promote_to_qa_pairs(
        self,
        raw_pairs_with_chunks: list[tuple[RawQAPair, Document]],
        dataset_id: str,
    ) -> list[QAPair]:
        """
        Convert validated RawQAPairs to full QAPair objects.

        Called after all filtering and deduplication. At this point
        every RawQAPair has passed quality checks and is ready for
        promotion to the full schema.
        """
        from datetime import datetime, timezone

        from src.dataset.schema import QAPairStatus, make_pair_id

        pairs: list[QAPair] = []

        for raw_pair, chunk in raw_pairs_with_chunks:
            pair = QAPair(
                id=make_pair_id(),
                dataset_id=dataset_id,
                created_at=datetime.now(timezone.utc),
                source_chunk=chunk.page_content,
                source_file=chunk.metadata.source_file,
                source_page=chunk.metadata.page_number,
                chunk_index=chunk.chunk_index,
                question=str(raw_pair.question).strip(),
                ground_truth_answer=str(raw_pair.answer).strip(),
                generation_method=self.generation_method,
                status=QAPairStatus.PENDING,
            )
            pairs.append(pair)

        return pairs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> GeneratorConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"n_pairs_per_chunk={self._config.n_pairs_per_chunk}, "
            f"max_pairs_total={self._config.max_pairs_total})"
        )

# ===========================================================================
# CUSTOM EXCEPTIONS
# ===========================================================================

class GenerationError(Exception):
    """
    Raised when dataset generation fails so completely that no
    pairs could be produced.

    Distinct from per-chunk failures (which are recorded in
    ChunkGenerationResult and do not abort the run) — GenerationError
    means the entire generation job produced nothing usable.

    The pipeline catches this and surfaces it to the UI as a
    user-actionable error with a clear message.
    """

    def __init__(
        self,
        generator: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.generator = generator
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[{generator}] Generation failed: {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )


class ChunkGenerationError(Exception):
    """
    Raised within _generate_from_chunk() implementations
    when a single chunk cannot be processed.

    Caught by the template method's per-chunk try/except.
    Never propagates to the caller — recorded in ChunkGenerationResult.
    """

    def __init__(
        self,
        chunk_index: int,
        source_file: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.source_file = source_file
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[chunk={chunk_index}, source='{source_file}'] {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )

__all__ = [
    "GeneratorConfig",
    "ChunkGenerationResult",
    "GenerationStats",
    "RawQAPair",
    "BaseDatasetGenerator",
    "GenerationError",
    "ChunkGenerationError",
]