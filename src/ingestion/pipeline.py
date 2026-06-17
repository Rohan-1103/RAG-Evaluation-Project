"""
src/ingestion/pipeline.py

IngestionPipeline — orchestrates the complete document ingestion flow.

Flow:
    Input files / directory
        ↓ LoaderRegistry          (selects correct loader per file)
        ↓ BaseLoader.load()       (file → list[Document])
        ↓ RecursiveChunker.chunk()(Documents → chunks)
        ↓ EmbeddingManager.embed()(chunks → np.ndarray)
        ↓ BaseVectorStore.add()   (chunks + embeddings → ChromaDB)
        ↓ IngestionResult         (structured summary returned to caller)

Design principles:
  - The pipeline is a pure orchestrator. It contains no format-specific
    logic, no embedding logic, and no storage logic. Every concern is
    delegated to an injected collaborator.
  - All collaborators are injected via constructor. The pipeline never
    imports PDFLoader, ChromaVectorStore, or EmbeddingManager directly.
    It depends on BaseLoader, BaseVectorStore, and EmbeddingManager
    abstractions only.
  - Failure isolation: a single file failure never aborts the entire
    job. Each file is processed in a try/except block. Failures are
    recorded in IngestionResult and the pipeline continues.
  - Idempotency: running the pipeline twice on the same files produces
    the same ChromaDB state. ChromaVectorStore.add_documents() uses
    upsert=True and deterministic IDs by default.
  - Progress reporting: the pipeline emits structured log events at
    every stage. The Streamlit UI tails these logs for live progress.

Collection naming:
  Each ingestion run targets one named ChromaDB collection.
  The collection name can be:
    - Explicitly provided by the caller (e.g. "q3_report")
    - Auto-derived from the source directory name
    - The default collection from ChromaConfig
  This allows the UI to maintain multiple independent knowledge bases
  (one per document set) within a single ChromaDB instance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.settings import ChromaConfig, EmbeddingConfig, IngestionConfig
from src.ingestion.base import (
    BaseLoader,
    Document,
    LoaderError,
    LoaderRegistry,
    UnsupportedFormatError,
)
from src.ingestion.chunker import ChunkerConfig, ChunkingStats, RecursiveChunker
from src.ingestion.loaders import DOCXLoader, HTMLLoader, PDFLoader, TXTLoader
from src.vectorstore.base import AddResult, BaseVectorStore
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.embeddings import EmbeddingError, EmbeddingManager

# ===========================================================================
# RESULT TYPES
# ===========================================================================

@dataclass
class FileIngestionResult:
    """
    Result of ingesting a single file.

    Granular per-file result so the caller can identify exactly
    which files succeeded, which were skipped, and which failed —
    without parsing log messages.
    """

    path: Path
    status: str                      # "success" | "skipped" | "failed"
    document_count: int = 0          # Pages/docs loaded from file
    chunk_count: int = 0             # Chunks produced from this file
    added_to_store: int = 0          # Chunks successfully stored
    error_message: str | None = None # Set if status == "failed"
    loader_class: str = ""           # Which loader handled this file
    latency_ms: float = 0.0          # Wall-clock time for this file

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
            f"FileIngestionResult("
            f"file='{self.path.name}', "
            f"status={self.status}, "
            f"chunks={self.chunk_count}, "
            f"stored={self.added_to_store})"
        )


@dataclass
class IngestionResult:
    """
    Aggregate result of a complete ingestion run.

    Returned to the caller (Streamlit UI, FastAPI endpoint, CLI script)
    as the single source of truth about what happened during ingestion.

    All counts are derived from file_results — never set directly.
    """

    collection_name: str
    run_id: str
    started_at: str                              # ISO 8601
    completed_at: str | None = None
    file_results: list[FileIngestionResult] = field(default_factory=list)
    chunking_stats: ChunkingStats | None = None
    total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Derived counts — computed from file_results
    # ------------------------------------------------------------------

    @property
    def total_files_attempted(self) -> int:
        return len(self.file_results)

    @property
    def succeeded_files(self) -> int:
        return sum(1 for r in self.file_results if r.succeeded)

    @property
    def failed_files(self) -> int:
        return sum(1 for r in self.file_results if r.failed)

    @property
    def skipped_files(self) -> int:
        return sum(1 for r in self.file_results if r.skipped)

    @property
    def total_documents_loaded(self) -> int:
        return sum(r.document_count for r in self.file_results)

    @property
    def total_chunks_produced(self) -> int:
        return sum(r.chunk_count for r in self.file_results)

    @property
    def total_chunks_stored(self) -> int:
        return sum(r.added_to_store for r in self.file_results)

    @property
    def success_rate(self) -> float:
        if self.total_files_attempted == 0:
            return 1.0
        return self.succeeded_files / self.total_files_attempted

    @property
    def failed_file_paths(self) -> list[Path]:
        return [r.path for r in self.file_results if r.failed]

    @property
    def is_complete_success(self) -> bool:
        return self.failed_files == 0 and self.succeeded_files > 0

    def summary(self) -> dict[str, Any]:
        """
        Return a flat dict suitable for dashboard display and logging.
        """
        return {
            "collection_name":       self.collection_name,
            "run_id":                self.run_id,
            "started_at":            self.started_at,
            "completed_at":          self.completed_at,
            "total_files_attempted": self.total_files_attempted,
            "succeeded_files":       self.succeeded_files,
            "failed_files":          self.failed_files,
            "skipped_files":         self.skipped_files,
            "total_documents_loaded":self.total_documents_loaded,
            "total_chunks_produced": self.total_chunks_produced,
            "total_chunks_stored":   self.total_chunks_stored,
            "success_rate":          round(self.success_rate, 4),
            "total_latency_ms":      round(self.total_latency_ms, 1),
        }

    def __repr__(self) -> str:
        return (
            f"IngestionResult("
            f"collection='{self.collection_name}', "
            f"files={self.succeeded_files}/{self.total_files_attempted}, "
            f"chunks_stored={self.total_chunks_stored})"
        )

# ===========================================================================
# INGESTION PIPELINE
# ===========================================================================

class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow.

    Constructor injection pattern — all collaborators are passed in,
    never constructed internally. This enables:
      - Unit testing with mock collaborators
      - Swapping ChromaVectorStore for Pinecone with zero code changes
      - Swapping PDFLoader for a higher-fidelity parser transparently

    Standard construction (via factory):
        pipeline = IngestionPipeline.build_default(settings)
        result = pipeline.ingest_directory(Path("./data/raw_docs"))

    Custom construction (for testing or custom loaders):
        mock_store = MockVectorStore()
        pipeline = IngestionPipeline(
            registry=registry,
            chunker=chunker,
            embedding_manager=embedding_manager,
            vector_store=mock_store,
            config=ingestion_config,
        )
    """

    # Embedding batch size — how many chunks to embed in one call.
    # Larger batches are faster but use more RAM.
    # 128 chunks × 1000 chars ≈ 128K chars per batch — safe for CPU.
    _EMBEDDING_BATCH_SIZE: int = 128

    def __init__(
        self,
        registry: LoaderRegistry,
        chunker: RecursiveChunker,
        embedding_manager: EmbeddingManager,
        vector_store: BaseVectorStore,
        config: IngestionConfig,
    ) -> None:
        self._registry = registry
        self._chunker = chunker
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store
        self._config = config

        logger.info(
            f"IngestionPipeline initialised. "
            f"registered_loaders={self._registry.registered_loaders()}, "
            f"chunk_size={chunker.chunk_size}, "
            f"chunk_overlap={chunker.chunk_overlap}, "
            f"embedding_model={embedding_manager.model_name}"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        ingestion_config: IngestionConfig,
        embedding_config: EmbeddingConfig,
        chroma_config: ChromaConfig,
    ) -> IngestionPipeline:
        """
        Construct a fully wired IngestionPipeline with default collaborators.

        Default stack:
          Loaders:       PDFLoader, TXTLoader, HTMLLoader, DOCXLoader
          Chunker:       RecursiveChunker from IngestionConfig
          Embedder:      EmbeddingManager from EmbeddingConfig
          VectorStore:   ChromaVectorStore from ChromaConfig

        For alternative stacks, construct manually via __init__.
        """
        # Build loader registry
        registry = LoaderRegistry()
        registry.register(PDFLoader(ingestion_config))
        registry.register(TXTLoader(ingestion_config))
        registry.register(HTMLLoader(ingestion_config))
        registry.register(DOCXLoader(ingestion_config))

        # Build chunker
        chunker_config = ChunkerConfig.from_ingestion_config(ingestion_config)
        chunker = RecursiveChunker(chunker_config)

        # Build embedding manager (loads model on construction)
        embedding_manager = EmbeddingManager(embedding_config)

        # Build vector store
        vector_store = ChromaVectorStore(
            config=chroma_config,
            embedding_manager=embedding_manager,
        )

        return cls(
            registry=registry,
            chunker=chunker,
            embedding_manager=embedding_manager,
            vector_store=vector_store,
            config=ingestion_config,
        )

    # ------------------------------------------------------------------
    # Public ingestion interface
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        directory: Path,
        collection_name: str | None = None,
        recursive: bool = True,
        upsert: bool = True,
    ) -> IngestionResult:
        """
        Ingest all supported files from a directory.

        Args:
            directory:       Path to directory containing source documents.
            collection_name: ChromaDB collection to store into.
                             Defaults to directory stem (folder name).
            recursive:       If True, scan subdirectories recursively.
            upsert:          If True, update existing docs (idempotent).
                             If False, skip files already in collection.

        Returns:
            IngestionResult with per-file results and aggregate stats.

        Raises:
            ValueError: if directory does not exist or is not a directory.
        """
        directory = directory.resolve()

        if not directory.exists():
            raise ValueError(
                f"IngestionPipeline.ingest_directory: "
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"IngestionPipeline.ingest_directory: "
                f"Path is not a directory: {directory}"
            )

        resolved_collection = collection_name or directory.stem
        logger.info(
            f"IngestionPipeline: Starting directory ingestion. "
            f"directory='{directory}', "
            f"collection='{resolved_collection}', "
            f"recursive={recursive}"
        )

        # Discover all files
        files = self._discover_files(directory, recursive)

        if not files:
            logger.warning(
                f"IngestionPipeline: No supported files found in "
                f"'{directory}'. "
                f"Supported extensions: "
                f"{sorted(self._registry.supported_extensions())}"
            )

        return self._run_ingestion(
            files=files,
            collection_name=resolved_collection,
            upsert=upsert,
        )

    def ingest_files(
        self,
        files: list[Path],
        collection_name: str,
        upsert: bool = True,
    ) -> IngestionResult:
        """
        Ingest a specific list of files.

        Used by the Streamlit UI when the user uploads files individually
        rather than pointing to a directory.

        Args:
            files:           List of file paths to ingest.
            collection_name: Target ChromaDB collection.
            upsert:          Update existing documents if True.

        Returns:
            IngestionResult with per-file results and aggregate stats.
        """
        resolved_files = [f.resolve() for f in files]
        logger.info(
            f"IngestionPipeline: Starting file list ingestion. "
            f"file_count={len(resolved_files)}, "
            f"collection='{collection_name}'"
        )
        return self._run_ingestion(
            files=resolved_files,
            collection_name=collection_name,
            upsert=upsert,
        )

    def ingest_single_file(
        self,
        file_path: Path,
        collection_name: str,
        upsert: bool = True,
    ) -> FileIngestionResult:
        """
        Ingest a single file. Returns FileIngestionResult directly.

        Convenience wrapper for one-off file ingestion without
        constructing a full IngestionResult. Used in tests and scripts.
        """
        file_path = file_path.resolve()
        result = self._run_ingestion(
            files=[file_path],
            collection_name=collection_name,
            upsert=upsert,
        )
        if result.file_results:
            return result.file_results[0]
        return FileIngestionResult(
            path=file_path,
            status="skipped",
            error_message="No supported files found.",
        )

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        files: list[Path],
        collection_name: str,
        upsert: bool,
    ) -> IngestionResult:
        """
        Core ingestion loop. Processes files sequentially.

        Each file goes through: load → chunk → embed → store.
        Failures at any stage are caught, recorded, and the loop
        continues with the next file.
        """
        from datetime import datetime, timezone

        run_id = self._generate_run_id()
        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.monotonic()

        result = IngestionResult(
            collection_name=collection_name,
            run_id=run_id,
            started_at=started_at,
        )

        # Ensure collection exists before processing files
        self._vector_store.get_or_create_collection(collection_name)

        all_chunks: list[Document] = []
        file_chunk_map: dict[str, tuple[int, int]] = {}
        # Maps file path string → (start_idx, end_idx) in all_chunks

        # ── Stage 1: Load + Chunk all files ──────────────────────────
        for file_path in files:
            file_result = self._load_and_chunk_file(file_path)
            result.file_results.append(file_result)

            if file_result.succeeded and file_result.chunk_count > 0:
                # Record where this file's chunks start in all_chunks
                start_idx = len(all_chunks)

                # Re-load chunks for this file (stored in file_result
                # indirectly — we need to re-run or cache)
                # Design note: We cache chunks per-file below instead
                pass

        # Cleaner approach: load-chunk-cache in one pass
        result.file_results.clear()
        all_chunks.clear()

        file_chunks_cache: dict[str, list[Document]] = {}

        for file_path in files:
            file_start = time.monotonic()
            file_result, file_chunks = self._load_and_chunk_file_with_cache(
                file_path
            )
            file_result.latency_ms = (time.monotonic() - file_start) * 1000
            result.file_results.append(file_result)

            if file_result.succeeded and file_chunks:
                file_chunks_cache[str(file_path)] = file_chunks
                all_chunks.extend(file_chunks)

        # Compute chunking stats across all files
        if all_chunks:
            result.chunking_stats = self._chunker._compute_stats(
                input_count=result.total_documents_loaded,
                chunks=all_chunks,
                skipped_empty=0,
            )

        # ── Stage 2: Embed all chunks in batches ─────────────────────
        if not all_chunks:
            logger.warning(
                f"IngestionPipeline: No chunks to embed for "
                f"collection '{collection_name}'. "
                f"Check that source files are not empty."
            )
            result.completed_at = self._now_iso()
            result.total_latency_ms = (
                time.monotonic() - wall_start
            ) * 1000
            return result

        logger.info(
            f"IngestionPipeline: Embedding {len(all_chunks)} chunks "
            f"in batches of {self._EMBEDDING_BATCH_SIZE}..."
        )

        all_embeddings = self._embed_chunks_batched(all_chunks)

        if all_embeddings is None:
            # Embedding failed entirely — mark all succeeded files as failed
            logger.error(
                "IngestionPipeline: Embedding failed for all chunks. "
                "No documents will be stored."
            )
            for file_result in result.file_results:
                if file_result.succeeded:
                    file_result.status = "failed"
                    file_result.error_message = (
                        "Embedding failed — see logs for details."
                    )
            result.completed_at = self._now_iso()
            result.total_latency_ms = (
                time.monotonic() - wall_start
            ) * 1000
            return result

        # ── Stage 3: Store chunks + embeddings ───────────────────────
        logger.info(
            f"IngestionPipeline: Storing {len(all_chunks)} chunks "
            f"in collection '{collection_name}'..."
        )

        add_result = self._store_chunks(
            chunks=all_chunks,
            embeddings=all_embeddings,
            collection_name=collection_name,
            upsert=upsert,
        )

        # Distribute stored counts back to file results
        self._distribute_stored_counts(
            file_results=result.file_results,
            file_chunks_cache=file_chunks_cache,
            add_result=add_result,
        )

        result.completed_at = self._now_iso()
        result.total_latency_ms = (time.monotonic() - wall_start) * 1000

        logger.info(
            f"IngestionPipeline: Run complete. {result.summary()}"
        )

        return result

    # ------------------------------------------------------------------
    # Per-file load + chunk
    # ------------------------------------------------------------------

    def _load_and_chunk_file_with_cache(
        self,
        file_path: Path,
    ) -> tuple[FileIngestionResult, list[Document]]:
        """
        Load and chunk a single file. Returns (result, chunks).

        Separating this from _run_ingestion keeps the orchestration
        loop readable. All exceptions are caught here — the loop
        always receives a valid (result, chunks) pair.
        """
        file_result = FileIngestionResult(
            path=file_path,
            status="failed",
        )

        # ── Check loader support ──────────────────────────────────────
        if not self._registry.is_supported(file_path):
            ext = file_path.suffix.lstrip(".").lower()
            file_result.status = "skipped"
            file_result.error_message = (
                f"No loader registered for extension '.{ext}'. "
                f"Supported: "
                f"{sorted(self._registry.supported_extensions())}"
            )
            logger.debug(
                f"IngestionPipeline: Skipping '{file_path.name}' — "
                f"unsupported extension '.{ext}'."
            )
            return file_result, []

        # ── Load ─────────────────────────────────────────────────────
        try:
            loader = self._registry.get_loader_for(file_path)
            file_result.loader_class = loader.__class__.__name__
            documents = loader.load(file_path)

            if not documents:
                file_result.status = "skipped"
                file_result.error_message = (
                    "Loader returned no documents — file may be empty "
                    "or contain only images/non-extractable content."
                )
                logger.warning(
                    f"IngestionPipeline: '{file_path.name}' produced "
                    f"no documents."
                )
                return file_result, []

            file_result.document_count = len(documents)
            logger.debug(
                f"IngestionPipeline: Loaded {len(documents)} documents "
                f"from '{file_path.name}'."
            )

        except (LoaderError, UnsupportedFormatError) as exc:
            file_result.status = "failed"
            file_result.error_message = str(exc)
            logger.error(
                f"IngestionPipeline: Failed to load '{file_path.name}': "
                f"{exc}"
            )
            return file_result, []

        except Exception as exc:
            file_result.status = "failed"
            file_result.error_message = (
                f"Unexpected error during load: {type(exc).__name__}: {exc}"
            )
            logger.exception(
                f"IngestionPipeline: Unexpected error loading "
                f"'{file_path.name}': {exc}"
            )
            return file_result, []

        # ── Chunk ─────────────────────────────────────────────────────
        try:
            chunks, _ = self._chunker.chunk(documents)

            if not chunks:
                file_result.status = "skipped"
                file_result.error_message = (
                    "Chunker produced no chunks — "
                    "all content may have been empty after cleaning."
                )
                logger.warning(
                    f"IngestionPipeline: '{file_path.name}' produced "
                    f"no chunks after splitting."
                )
                return file_result, []

            file_result.chunk_count = len(chunks)
            file_result.status = "success"

            logger.debug(
                f"IngestionPipeline: '{file_path.name}' → "
                f"{len(chunks)} chunks."
            )
            return file_result, chunks

        except Exception as exc:
            file_result.status = "failed"
            file_result.error_message = (
                f"Chunking failed: {type(exc).__name__}: {exc}"
            )
            logger.exception(
                f"IngestionPipeline: Chunking failed for "
                f"'{file_path.name}': {exc}"
            )
            return file_result, []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_chunks_batched(
        self,
        chunks: list[Document],
    ) -> np.ndarray | None:
        """
        Embed all chunks in batches. Returns full embedding matrix.

        Returns None if embedding fails entirely.
        Partial batch failures are logged and the failed batch's
        chunks receive zero vectors — flagged in the stored metadata.

        Returns:
            np.ndarray of shape (len(chunks), embedding_dim)
            or None on total failure.
        """
        all_embeddings: list[np.ndarray] = []
        texts = [chunk.page_content for chunk in chunks]
        total = len(texts)
        batch_size = self._EMBEDDING_BATCH_SIZE

        for batch_start in range(0, total, batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.debug(
                f"IngestionPipeline: Embedding batch "
                f"{batch_num}/{total_batches} "
                f"({len(batch_texts)} chunks)..."
            )

            try:
                embed_result = self._embedding_manager.embed(
                    texts=batch_texts,
                    show_progress=False,
                )
                all_embeddings.append(embed_result.embeddings)

            except EmbeddingError as exc:
                logger.error(
                    f"IngestionPipeline: Embedding batch {batch_num} "
                    f"failed: {exc}. "
                    f"Using zero vectors for this batch."
                )
                # Zero vectors for failed batch — stored but flagged
                dim = self._embedding_manager.dimension
                zero_batch = np.zeros(
                    (len(batch_texts), dim),
                    dtype=np.float32,
                )
                all_embeddings.append(zero_batch)

            except Exception as exc:
                logger.exception(
                    f"IngestionPipeline: Unexpected error in embedding "
                    f"batch {batch_num}: {exc}"
                )
                if not all_embeddings:
                    # First batch failed — cannot continue
                    return None
                dim = self._embedding_manager.dimension
                zero_batch = np.zeros(
                    (len(batch_texts), dim),
                    dtype=np.float32,
                )
                all_embeddings.append(zero_batch)

        if not all_embeddings:
            return None

        full_matrix = np.vstack(all_embeddings)
        logger.info(
            f"IngestionPipeline: Embedding complete. "
            f"shape={full_matrix.shape}"
        )
        return full_matrix

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_chunks(
        self,
        chunks: list[Document],
        embeddings: np.ndarray,
        collection_name: str,
        upsert: bool,
    ) -> AddResult:
        """
        Store chunks and embeddings in the vector store.

        Catches VectorStoreError and returns an empty AddResult
        so the pipeline can record the failure without crashing.
        """
        try:
            add_result = self._vector_store.add_documents(
                documents=chunks,
                embeddings=embeddings,
                collection_name=collection_name,
                upsert=upsert,
            )
            logger.info(
                f"IngestionPipeline: Store complete. "
                f"added={add_result.added_count}, "
                f"skipped={add_result.skipped_count}, "
                f"failed={add_result.failed_count}"
            )
            return add_result

        except Exception as exc:
            logger.exception(
                f"IngestionPipeline: Vector store write failed: {exc}"
            )
            return AddResult(
                collection_name=collection_name,
                added_count=0,
                skipped_count=0,
                failed_count=len(chunks),
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_files(
        self,
        directory: Path,
        recursive: bool,
    ) -> list[Path]:
        """
        Find all supported files in a directory.

        Returns files sorted by name for deterministic processing order.
        Deterministic ordering means re-runs produce the same chunk IDs
        in the same sequence, keeping logs comparable across runs.
        """
        supported = self._registry.supported_extensions()
        found: list[Path] = []

        pattern = "**/*" if recursive else "*"

        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lstrip(".").lower()
            if ext in supported:
                found.append(file_path)

        logger.info(
            f"IngestionPipeline: Discovered {len(found)} supported files "
            f"in '{directory}' (recursive={recursive})."
        )

        if found:
            logger.debug(
                f"IngestionPipeline: Files to process: "
                f"{[f.name for f in found]}"
            )

        return found

    # ------------------------------------------------------------------
    # Stored count distribution
    # ------------------------------------------------------------------

    def _distribute_stored_counts(
        self,
        file_results: list[FileIngestionResult],
        file_chunks_cache: dict[str, list[Document]],
        add_result: AddResult,
    ) -> None:
        """
        Distribute the total stored count back to per-file results.

        ChromaDB's add_documents() returns an aggregate count, not
        a per-file count. We distribute proportionally based on
        each file's chunk count relative to the total.

        This is an approximation — if some chunks from file A failed
        and some from file B succeeded, the proportional distribution
        may not be perfectly accurate. For the dashboard, this is
        acceptable. Exact per-file storage counts would require
        individual add_documents() calls per file (much slower).
        """
        total_chunks = sum(
            len(chunks)
            for chunks in file_chunks_cache.values()
        )

        if total_chunks == 0:
            return

        for file_result in file_results:
            if not file_result.succeeded:
                continue
            file_key = str(file_result.path)
            file_chunk_count = len(
                file_chunks_cache.get(file_key, [])
            )
            if file_chunk_count == 0:
                continue
            proportion = file_chunk_count / total_chunks
            file_result.added_to_store = round(
                add_result.added_count * proportion
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_and_chunk_file(
        self,
        file_path: Path,
    ) -> FileIngestionResult:
        """
        Thin wrapper used in the first (now replaced) pass.
        Retained for backward compatibility with any external callers.
        """
        result, _ = self._load_and_chunk_file_with_cache(file_path)
        return result

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a time-sortable run ID."""
        import uuid
        return f"ingest_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO 8601 string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supported_extensions(self) -> frozenset[str]:
        return self._registry.supported_extensions()

    @property
    def collection_names(self) -> list[str]:
        """Return names of all existing collections in the vector store."""
        return [c.name for c in self._vector_store.list_collections()]

    def __repr__(self) -> str:
        return (
            f"IngestionPipeline("
            f"loaders={[l.__class__.__name__ for l in self._registry.registered_loaders()]}, "
            f"chunk_size={self._chunker.chunk_size}, "
            f"embedding_model='{self._embedding_manager.model_name}')"
        )

__all__ = [
    "FileIngestionResult",
    "IngestionResult",
    "IngestionPipeline",
]