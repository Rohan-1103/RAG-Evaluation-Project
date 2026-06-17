"""
src/ingestion/chunker.py

RecursiveChunker — splits Documents into smaller chunks for RAG indexing.

Design contract:
  - Input:  list[Document] from any loader
  - Output: list[Document] with chunk-level fields populated
  - The chunker never touches embeddings, storage, or LLM calls.
  - Every output Document carries a reference to its parent via
    parent_doc_id so the evaluation pipeline can trace a retrieved
    chunk back to its source page and document.

Why RecursiveCharacterTextSplitter:
  Splitting strategies in order of preference:
    1. \n\n  — paragraph boundaries (best semantic unit)
    2. \n    — line boundaries
    3. ". "  — sentence boundaries
    4. " "   — word boundaries (last resort)
    5. ""    — character-level (never ideal, avoids hard truncation)

  The splitter tries each separator in order and only falls back
  to the next when a chunk would exceed chunk_size. This means
  well-structured documents (markdown, DOCX with headings) split
  at paragraph/section boundaries almost always.

Overlap rationale:
  chunk_overlap=200 means the last 200 characters of chunk N
  appear at the start of chunk N+1. This ensures a sentence
  split across a boundary is fully present in at least one chunk.
  Without overlap, a question about content spanning a boundary
  may not be answerable from any single retrieved chunk.

Chunk size guidance for Gemini:
  - gemini-1.5-flash context: 1M tokens
  - Each token ≈ 4 characters
  - chunk_size=1000 chars ≈ 250 tokens per chunk
  - top_k=5 chunks ≈ 1250 tokens of context per query
  - Well within limits while keeping context focused.
  - For dense technical content, reduce to chunk_size=500.
  - For narrative text, chunk_size=1500 works well.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from loguru import logger
from config.settings import IngestionConfig
from src.ingestion.base import Document, DocumentMetadata

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkerConfig:
    """
    Chunking parameters. Frozen — a chunker's config is immutable
    after construction. Change config by constructing a new chunker.

    Separators are tried in order — first match wins for each split.
    Override for domain-specific content:
      - Code files:      ["\nclass ", "\ndef ", "\n\n", "\n"]
      - Legal documents: ["\nSection ", "\n\n", "\n", ". "]
      - CSV/tables:      ["\n"]
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: tuple[str, ...] = (
        "\n\n",    # Paragraph break — highest priority
        "\n",      # Line break
        ". ",      # Sentence boundary
        "? ",      # Question boundary
        "! ",      # Exclamation boundary
        "; ",      # Clause boundary
        ", ",      # Phrase boundary
        " ",       # Word boundary
        "",        # Character-level fallback — avoids hard truncation
    )
    length_function: str = "char"    # "char" | "word" — how size is measured
    strip_whitespace: bool = True    # Strip leading/trailing whitespace per chunk

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"ChunkerConfig: chunk_overlap ({self.chunk_overlap}) "
                f"must be strictly less than chunk_size ({self.chunk_size}). "
                f"Equal or larger overlap causes infinite splitting loops."
            )
        if self.chunk_size < 50:
            raise ValueError(
                f"ChunkerConfig: chunk_size ({self.chunk_size}) is too small. "
                f"Minimum is 50 characters. "
                f"Very small chunks lose semantic meaning."
            )
        if self.length_function not in ("char", "word"):
            raise ValueError(
                f"ChunkerConfig: length_function must be 'char' or 'word', "
                f"got '{self.length_function}'."
            )

    @classmethod
    def from_ingestion_config(cls, config: IngestionConfig) -> ChunkerConfig:
        """Construct from Settings.ingestion — the standard factory."""
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

# ---------------------------------------------------------------------------
# Chunk statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingStats:
    """
    Statistics from a chunking operation.

    Returned alongside chunks so the pipeline can log quality signals
    without re-scanning the output list.
    """

    input_document_count: int
    output_chunk_count: int
    skipped_empty_count: int
    avg_chunk_size: float           # Characters
    min_chunk_size: int
    max_chunk_size: int
    total_characters: int

    @property
    def avg_chunks_per_document(self) -> float:
        if self.input_document_count == 0:
            return 0.0
        return self.output_chunk_count / self.input_document_count

    @property
    def expansion_ratio(self) -> float:
        """
        How much the chunk count grew relative to input document count.

        Ratio > 20 may indicate chunk_size is too small.
        Ratio < 1.5 may indicate chunk_size is too large.
        """
        return self.avg_chunks_per_document

    def __repr__(self) -> str:
        return (
            f"ChunkingStats("
            f"input_docs={self.input_document_count}, "
            f"output_chunks={self.output_chunk_count}, "
            f"avg_size={self.avg_chunk_size:.0f}chars, "
            f"avg_per_doc={self.avg_chunks_per_document:.1f})"
        )

# ===========================================================================
# RECURSIVE CHUNKER
# ===========================================================================

class RecursiveChunker:
    """
    Splits Documents into overlapping chunks using recursive
    character-based text splitting.

    Wraps LangChain's RecursiveCharacterTextSplitter for the
    splitting algorithm while maintaining our own Document type
    and metadata propagation contract.

    Why wrap LangChain here but not elsewhere:
      RecursiveCharacterTextSplitter is a well-tested, battle-hardened
      implementation of a non-trivial algorithm. Reimplementing it
      would add risk with no benefit. The wrapper ensures:
        1. Input/output types are our Document (not LangChain's).
        2. Metadata propagation is under our control.
        3. The chunker can be swapped without any caller changes.

    Usage:
        chunker = RecursiveChunker(ChunkerConfig.from_ingestion_config(
            settings.ingestion
        ))
        chunks, stats = chunker.chunk(documents)
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config
        self._splitter = self._build_splitter()
        logger.debug(
            f"RecursiveChunker initialised. "
            f"chunk_size={config.chunk_size}, "
            f"chunk_overlap={config.chunk_overlap}, "
            f"length_function={config.length_function}"
        )

    # ------------------------------------------------------------------
    # Splitter construction
    # ------------------------------------------------------------------

    def _build_splitter(self) -> object:
        """
        Construct LangChain RecursiveCharacterTextSplitter.

        The length function is selected based on config:
          "char" — len(text) — fast, standard
          "word" — len(text.split()) — better for variable-length tokens
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import (
                    RecursiveCharacterTextSplitter,
                )
            except ImportError as exc:
                raise ImportError(
                    "RecursiveCharacterTextSplitter not found. "
                    "Run: poetry add langchain-text-splitters"
                ) from exc

        length_fn = (
            len
            if self._config.length_function == "char"
            else lambda t: len(t.split())
        )

        return RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            length_function=length_fn,
            separators=list(self._config.separators),
            strip_whitespace=self._config.strip_whitespace,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(
        self,
        documents: list[Document],
    ) -> tuple[list[Document], ChunkingStats]:
        """
        Split a list of Documents into chunks.

        Args:
            documents: Documents from any loader. Can be a mix of
                       formats — each is chunked independently.

        Returns:
            (chunks, stats) tuple:
              chunks — flat list of chunk Documents sorted by
                       source_file, page_number, chunk_index.
              stats  — ChunkingStats for logging and monitoring.

        Contract:
          - Every output chunk has chunk_index, chunk_of, parent_doc_id set.
          - Every output chunk inherits its parent's DocumentMetadata.
          - Empty input returns ([], stats_with_zeros).
          - Documents that are already small enough to fit in one chunk
            are returned as a single chunk (chunk_index=0, chunk_of=1).
        """
        if not documents:
            logger.warning("RecursiveChunker.chunk() called with empty list.")
            return [], ChunkingStats(
                input_document_count=0,
                output_chunk_count=0,
                skipped_empty_count=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                total_characters=0,
            )

        all_chunks: list[Document] = []
        skipped_empty = 0

        for doc in documents:
            if doc.is_empty:
                logger.debug(
                    f"RecursiveChunker: Skipping empty document "
                    f"'{doc.metadata.source_file}' "
                    f"page={doc.metadata.page_number}."
                )
                skipped_empty += 1
                continue

            doc_chunks = self._chunk_single_document(doc)
            all_chunks.extend(doc_chunks)

        stats = self._compute_stats(
            input_count=len(documents),
            chunks=all_chunks,
            skipped_empty=skipped_empty,
        )

        logger.info(
            f"RecursiveChunker: {stats.input_document_count} docs → "
            f"{stats.output_chunk_count} chunks. "
            f"avg_size={stats.avg_chunk_size:.0f}chars, "
            f"avg_per_doc={stats.avg_chunks_per_document:.1f}"
        )

        if stats.avg_chunk_size < 100:
            logger.warning(
                f"RecursiveChunker: Average chunk size is very small "
                f"({stats.avg_chunk_size:.0f} chars). "
                f"Consider increasing chunk_size to at least 500."
            )

        if stats.expansion_ratio > 30:
            logger.warning(
                f"RecursiveChunker: High expansion ratio "
                f"({stats.expansion_ratio:.1f} chunks/doc). "
                f"Consider increasing chunk_size or reducing overlap."
            )

        return all_chunks, stats

    def chunk_text(
        self,
        text: str,
        source_label: str = "inline_text",
    ) -> list[str]:
        """
        Split a raw string into chunk strings.

        Convenience method for cases where a Document wrapper is not
        needed — e.g. dataset generation splitting a context string.

        Returns list of strings, not Documents.
        """
        if not text or not text.strip():
            return []

        from langchain_core.documents import Document as LCDocument

        lc_docs = self._splitter.split_documents(  # type: ignore[union-attr]
            [LCDocument(page_content=text, metadata={"source": source_label})]
        )
        return [d.page_content for d in lc_docs if d.page_content.strip()]

    # ------------------------------------------------------------------
    # Single document chunking
    # ------------------------------------------------------------------

    def _chunk_single_document(
        self,
        doc: Document,
    ) -> list[Document]:
        """
        Split one Document into chunk Documents.

        If the document's content fits within chunk_size, it is returned
        as a single chunk without splitting. This avoids unnecessary
        LangChain overhead for short documents.

        Each chunk:
          - Inherits parent's DocumentMetadata exactly
          - Has chunk_index (0-based) and chunk_of (total) set
          - Has parent_doc_id derived from parent's fingerprint
        """
        content = doc.page_content

        # Fast path: content fits in one chunk
        if self._measure_length(content) <= self._config.chunk_size:
            parent_id = self._document_fingerprint(doc)
            return [
                Document(
                    page_content=content,
                    metadata=doc.metadata,
                    chunk_index=0,
                    chunk_of=1,
                    parent_doc_id=parent_id,
                )
            ]

        # Split via LangChain splitter
        from langchain_core.documents import Document as LCDocument

        lc_input = LCDocument(
            page_content=content,
            metadata={},   # Metadata is managed by us, not LangChain
        )

        try:
            lc_chunks = self._splitter.split_documents(  # type: ignore[union-attr]
                [lc_input]
            )
        except Exception as exc:
            logger.error(
                f"RecursiveChunker: LangChain splitter failed for "
                f"'{doc.metadata.source_file}' "
                f"page={doc.metadata.page_number}: {exc}. "
                f"Returning document as single chunk."
            )
            parent_id = self._document_fingerprint(doc)
            return [
                Document(
                    page_content=content,
                    metadata=doc.metadata,
                    chunk_index=0,
                    chunk_of=1,
                    parent_doc_id=parent_id,
                )
            ]

        # Filter empty chunks from splitter output
        raw_texts = [
            c.page_content
            for c in lc_chunks
            if c.page_content and c.page_content.strip()
        ]

        if not raw_texts:
            logger.warning(
                f"RecursiveChunker: Splitter produced no content for "
                f"'{doc.metadata.source_file}' "
                f"page={doc.metadata.page_number}. "
                f"Returning as single chunk."
            )
            parent_id = self._document_fingerprint(doc)
            return [
                Document(
                    page_content=content,
                    metadata=doc.metadata,
                    chunk_index=0,
                    chunk_of=1,
                    parent_doc_id=parent_id,
                )
            ]

        parent_id = self._document_fingerprint(doc)
        total_chunks = len(raw_texts)
        result: list[Document] = []

        for idx, chunk_text in enumerate(raw_texts):
            # Build enriched metadata for this chunk
            # Chunk-level fields go on the Document, not in metadata
            chunk_metadata = self._enrich_chunk_metadata(
                parent_metadata=doc.metadata,
                chunk_index=idx,
                chunk_of=total_chunks,
                chunk_text=chunk_text,
            )

            result.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_index=idx,
                    chunk_of=total_chunks,
                    parent_doc_id=parent_id,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Metadata enrichment
    # ------------------------------------------------------------------

    def _enrich_chunk_metadata(
        self,
        parent_metadata: DocumentMetadata,
        chunk_index: int,
        chunk_of: int,
        chunk_text: str,
    ) -> DocumentMetadata:
        """
        Produce DocumentMetadata for a chunk derived from a parent.

        Inherits all parent fields. Adds chunk-level extra fields:
          - chunk_char_count: precise character count of this chunk
          - chunk_word_count: approximate word count

        These are stored in extra so they appear in the evaluation
        dashboard's drilldown table without schema changes.
        """
        parent_extra = dict(parent_metadata.extra)
        parent_extra["chunk_char_count"] = len(chunk_text)
        parent_extra["chunk_word_count"] = len(chunk_text.split())
        parent_extra["chunk_index"] = chunk_index
        parent_extra["chunk_of"] = chunk_of

        return DocumentMetadata(
            source_file=parent_metadata.source_file,
            source_path=parent_metadata.source_path,
            file_type=parent_metadata.file_type,
            file_size_bytes=parent_metadata.file_size_bytes,
            total_pages=parent_metadata.total_pages,
            page_number=parent_metadata.page_number,
            title=parent_metadata.title,
            author=parent_metadata.author,
            created_at=parent_metadata.created_at,
            loader_class=parent_metadata.loader_class,
            extra=parent_extra,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _measure_length(self, text: str) -> int:
        """Apply the configured length function to a string."""
        if self._config.length_function == "char":
            return len(text)
        return len(text.split())

    @staticmethod
    def _document_fingerprint(doc: Document) -> str:
        """
        Stable identifier for a parent document.

        Used as parent_doc_id on all chunks derived from this document.
        Combines source_file + page_number + first 100 chars of content
        for uniqueness without storing full content in the ID.
        """
        parts = [
            doc.metadata.source_file,
            str(doc.metadata.page_number or ""),
            doc.page_content[:100],
        ]
        fingerprint = "|".join(parts)
        return hashlib.sha256(
            fingerprint.encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _compute_stats(
        input_count: int,
        chunks: list[Document],
        skipped_empty: int,
    ) -> ChunkingStats:
        """Compute ChunkingStats from the output chunk list."""
        if not chunks:
            return ChunkingStats(
                input_document_count=input_count,
                output_chunk_count=0,
                skipped_empty_count=skipped_empty,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                total_characters=0,
            )

        sizes = [len(c.page_content) for c in chunks]
        total_chars = sum(sizes)

        return ChunkingStats(
            input_document_count=input_count,
            output_chunk_count=len(chunks),
            skipped_empty_count=skipped_empty,
            avg_chunk_size=total_chars / len(chunks),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            total_characters=total_chars,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ChunkerConfig:
        return self._config

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._config.chunk_overlap

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker("
            f"chunk_size={self._config.chunk_size}, "
            f"chunk_overlap={self._config.chunk_overlap}, "
            f"length_fn={self._config.length_function})"
        )

__all__ = [
    "ChunkerConfig",
    "ChunkingStats",
    "RecursiveChunker",
]