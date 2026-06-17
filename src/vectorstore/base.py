"""
src/vectorstore/base.py

Abstract base class for all vector store implementations.

Design contract:
  - Every vector store receives Documents and returns Documents.
  - No vector store ever touches chunking, embedding, or LLM logic.
    Single responsibility: store vectors → retrieve by similarity.
  - All vector stores are collection-aware. A single store instance
    can manage multiple named collections (one per ingestion job).
  - The ABC defines the full interface contract. ChromaVectorStore,
    PineconeVectorStore, or any future implementation must satisfy
    every method here — no partial implementations allowed.

Why abstract the vector store:
  - ChromaDB is the default (free, local, zero infra).
  - In production a team may switch to Pinecone or Weaviate.
  - The EvaluationEngine, RAGPipeline, and IngestionPipeline all
    depend on BaseVectorStore — never on ChromaVectorStore directly.
  - Swapping the store requires one line change in dependency wiring,
    zero changes in any business logic module.

Embedding responsibility boundary:
  - The vector store accepts pre-computed embeddings (np.ndarray).
  - It does NOT compute embeddings internally.
  - EmbeddingManager owns embedding computation.
  - This separation means the embedding model can be swapped
    independently of the storage backend.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from src.ingestion.base import Document

# ===========================================================================
# VALUE OBJECTS
# Typed return types for vector store operations.
# Using dataclasses instead of raw dicts means callers always know
# what fields are available without reading implementation code.
# ===========================================================================

@dataclass(frozen=True)
class QueryResult:
    """
    A single document returned by a similarity search.

    Frozen — query results are immutable after retrieval.
    The RAGPipeline reads these; it never mutates them.

    Fields:
      document        — the retrieved Document with full metadata
      similarity_score — 1.0 = identical, 0.0 = completely unrelated
                         Always in [0.0, 1.0] regardless of underlying
                         distance metric (cosine, L2, dot product).
                         Normalisation is the store's responsibility.
      distance        — raw distance value from the underlying index
                         (implementation-specific, for debugging only)
      rank            — 1-based position in the result list (1 = most similar)
      collection_name — which collection this result came from
    """

    document: Document
    similarity_score: float
    distance: float
    rank: int
    collection_name: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"QueryResult.similarity_score must be in [0.0, 1.0], "
                f"got {self.similarity_score}. "
                f"The vector store implementation must normalise distances "
                f"before constructing QueryResult."
            )
        if self.rank < 1:
            raise ValueError(
                f"QueryResult.rank must be >= 1 (1-based), got {self.rank}."
            )

    @property
    def content(self) -> str:
        """Convenience accessor — avoids .document.page_content chains."""
        return self.document.page_content

    @property
    def source_file(self) -> str:
        """Convenience accessor for the source filename."""
        return self.document.metadata.source_file

    def __repr__(self) -> str:
        return (
            f"QueryResult("
            f"rank={self.rank}, "
            f"score={self.similarity_score:.4f}, "
            f"source='{self.source_file}', "
            f"content='{self.content[:60]}...')"
        )

@dataclass(frozen=True)
class CollectionInfo:
    """
    Metadata about a vector store collection.

    Returned by get_collection_info() and list_collections().
    Provides a uniform view of collection state regardless of
    the underlying store implementation.
    """

    name: str
    document_count: int
    embedding_dimension: int | None       # None if store cannot report it
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None         # ISO 8601 if available
    last_modified: str | None = None      # ISO 8601 if available

    @property
    def is_empty(self) -> bool:
        return self.document_count == 0

    def __repr__(self) -> str:
        return (
            f"CollectionInfo("
            f"name='{self.name}', "
            f"docs={self.document_count}, "
            f"dim={self.embedding_dimension})"
        )

@dataclass(frozen=True)
class AddResult:
    """
    Result of a batch add operation.

    Returned by add_documents() to give the caller full visibility
    into what was stored vs skipped vs failed.
    """

    collection_name: str
    added_count: int
    skipped_count: int                    # Duplicates skipped if upsert=False
    failed_count: int
    document_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_attempted(self) -> int:
        return self.added_count + self.skipped_count + self.failed_count

    @property
    def success_rate(self) -> float:
        if self.total_attempted == 0:
            return 1.0
        return self.added_count / self.total_attempted

    @property
    def had_failures(self) -> bool:
        return self.failed_count > 0

    def __repr__(self) -> str:
        return (
            f"AddResult("
            f"collection='{self.collection_name}', "
            f"added={self.added_count}, "
            f"skipped={self.skipped_count}, "
            f"failed={self.failed_count})"
        )

# ===========================================================================
# ABSTRACT BASE CLASS
# ===========================================================================

class BaseVectorStore(ABC):
    """
    Abstract interface for all vector store backends.

    Method categories:
      WRITE:  add_documents, delete_documents, delete_collection
      READ:   query, get_document_by_id, get_collection_info
      ADMIN:  list_collections, collection_exists, get_or_create_collection

    All write operations are synchronous in the base interface.
    Async variants are provided as optional overrides (aquery, aadd_documents)
    for implementations that support native async I/O.

    Implementations must guarantee:
      1. Idempotency on add: adding the same document_id twice with
         upsert=True updates the record, does not duplicate it.
      2. Normalised scores: query() always returns similarity_score
         in [0.0, 1.0], regardless of internal distance metric.
      3. Rank ordering: query() results are always sorted by
         similarity_score descending (rank 1 = most similar).
      4. Collection isolation: documents in collection A are never
         returned by queries against collection B.
    """

    # ------------------------------------------------------------------
    # WRITE OPERATIONS
    # ------------------------------------------------------------------

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        collection_name: str,
        upsert: bool = True,
    ) -> AddResult:
        """
        Store documents with their pre-computed embeddings.

        Args:
            documents:       List of Document objects to store.
                             len(documents) must equal embeddings.shape[0].
            embeddings:      2D numpy array, shape (N, embedding_dim).
                             Row i corresponds to documents[i].
            collection_name: Target collection. Created if not exists.
            upsert:          If True, update existing docs with same ID.
                             If False, skip duplicates (log as skipped).

        Returns:
            AddResult with counts of added, skipped, and failed documents.

        Raises:
            VectorStoreError: on unrecoverable storage failure.
            DimensionMismatchError: if embedding dim conflicts with
                                    the collection's existing dimension.
            ValueError: if len(documents) != embeddings.shape[0].
        """
        ...

    @abstractmethod
    def delete_documents(
        self,
        document_ids: list[str],
        collection_name: str,
    ) -> int:
        """
        Delete documents by ID from a collection.

        Args:
            document_ids:    List of document IDs to delete.
            collection_name: Collection to delete from.

        Returns:
            Count of documents actually deleted.
            IDs not found are silently ignored (not an error).

        Raises:
            VectorStoreError: if the collection does not exist.
        """
        ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an entire collection and all its documents.

        Args:
            collection_name: Collection to delete.

        Returns:
            True if deleted, False if collection did not exist.

        Raises:
            VectorStoreError: on storage-level deletion failure.
        """
        ...

    # ------------------------------------------------------------------
    # READ OPERATIONS
    # ------------------------------------------------------------------

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Retrieve the top_k most similar documents to the query embedding.

        Args:
            query_embedding:  1D numpy array of shape (embedding_dim,).
            collection_name:  Collection to search.
            top_k:            Maximum results to return.
            score_threshold:  Minimum similarity_score. Results below
                              this threshold are excluded. Range [0.0, 1.0].
            filter_metadata:  Optional key-value filter applied before
                              similarity ranking. Implementation-specific
                              support — may be ignored if unsupported.

        Returns:
            List of QueryResult, sorted by similarity_score descending.
            May return fewer than top_k results if:
              - Collection has fewer than top_k documents.
              - score_threshold filters some results.
            Never returns more than top_k results.

        Raises:
            VectorStoreError: if collection does not exist.
            DimensionMismatchError: if query_embedding dim does not match
                                    the collection's embedding dimension.
        """
        ...

    @abstractmethod
    def get_document_by_id(
        self,
        document_id: str,
        collection_name: str,
    ) -> Document | None:
        """
        Retrieve a single document by its exact ID.

        Returns None if not found — never raises on missing ID.
        Used by the evaluation engine to fetch specific chunks
        referenced in eval results.

        Raises:
            VectorStoreError: if the collection does not exist.
        """
        ...

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Return metadata about a collection.

        Raises:
            VectorStoreError: if the collection does not exist.
        """
        ...

    # ------------------------------------------------------------------
    # ADMIN OPERATIONS
    # ------------------------------------------------------------------

    @abstractmethod
    def list_collections(self) -> list[CollectionInfo]:
        """
        Return info for all collections in this store.

        Returns empty list if no collections exist.
        Never raises — an empty store is valid state.
        """
        ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Return True if the named collection exists.

        Used for pre-flight checks before querying or deleting.
        Never raises.
        """
        ...

    @abstractmethod
    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """
        Return collection info, creating the collection if it does not exist.

        Idempotent — safe to call on every pipeline run.
        Used by IngestionPipeline before add_documents() to ensure
        the collection exists without checking existence separately.

        Args:
            collection_name: Collection name.
            metadata:        Optional metadata to attach on creation.
                             Ignored if collection already exists.

        Returns:
            CollectionInfo for the existing or newly created collection.
        """
        ...

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """
        Return the number of documents in a collection.

        Raises:
            VectorStoreError: if the collection does not exist.
        """
        ...

    # ------------------------------------------------------------------
    # OPTIONAL ASYNC INTERFACE
    # Default implementations delegate to sync methods.
    # Implementations with native async support should override these.
    # ------------------------------------------------------------------

    async def aquery(
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Async variant of query().

        Default implementation wraps the sync method.
        Override in implementations with native async I/O support
        (e.g. a Pinecone async client).
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.query(
                query_embedding,
                collection_name,
                top_k,
                score_threshold,
                filter_metadata,
            ),
        )

    async def aadd_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        collection_name: str,
        upsert: bool = True,
    ) -> AddResult:
        """
        Async variant of add_documents().

        Default implementation wraps the sync method.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.add_documents(
                documents,
                embeddings,
                collection_name,
                upsert,
            ),
        )

    # ------------------------------------------------------------------
    # CONCRETE HELPERS
    # Utility methods built on the abstract interface.
    # Available to all implementations without reimplementation.
    # ------------------------------------------------------------------

    def query_multiple_collections(
        self,
        query_embedding: np.ndarray,
        collection_names: list[str],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[QueryResult]:
        """
        Query multiple collections and return merged, re-ranked results.

        Results from all collections are pooled and re-sorted by
        similarity_score. Final result list is capped at top_k.

        Useful for multi-document evaluation runs where documents
        are stored in separate collections by source file.
        """
        all_results: list[QueryResult] = []
        for collection_name in collection_names:
            if not self.collection_exists(collection_name):
                continue
            results = self.query(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            all_results.extend(results)

        # Re-sort by similarity score descending, re-assign ranks
        all_results.sort(key=lambda r: r.similarity_score, reverse=True)
        reranked: list[QueryResult] = []
        for i, result in enumerate(all_results[:top_k]):
            reranked.append(
                QueryResult(
                    document=result.document,
                    similarity_score=result.similarity_score,
                    distance=result.distance,
                    rank=i + 1,
                    collection_name=result.collection_name,
                )
            )
        return reranked

    def get_total_document_count(self) -> int:
        """
        Return total document count across all collections.

        Convenience method for dashboard stats display.
        """
        return sum(
            info.document_count
            for info in self.list_collections()
        )

    def __repr__(self) -> str:
        collections = self.list_collections()
        total_docs = sum(c.document_count for c in collections)
        return (
            f"{self.__class__.__name__}("
            f"collections={len(collections)}, "
            f"total_documents={total_docs})"
        )

# ===========================================================================
# CUSTOM EXCEPTIONS
# ===========================================================================

class VectorStoreError(Exception):
    """
    Base exception for all vector store failures.

    Raised for storage-level errors that are not the caller's fault
    (e.g. ChromaDB internal error, disk full, connection refused).
    The IngestionPipeline catches this and records the failure.
    """

    def __init__(
        self,
        store: str,
        operation: str,
        collection: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.store = store
        self.operation = operation
        self.collection = collection
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[{store}] {operation} failed on collection '{collection}': {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )

class DimensionMismatchError(VectorStoreError):
    """
    Raised when an embedding's dimension conflicts with a collection.

    A collection's embedding dimension is fixed at first insert.
    Subsequent inserts or queries must use the same dimension.

    This is always a programming error — either the wrong embedding
    model was used, or the collection was not cleared before switching
    models. The error message names both dimensions to aid debugging.
    """

    def __init__(
        self,
        store: str,
        collection: str,
        expected_dim: int,
        received_dim: int,
    ) -> None:
        self.expected_dim = expected_dim
        self.received_dim = received_dim
        super().__init__(
            store=store,
            operation="dimension_check",
            collection=collection,
            reason=(
                f"Embedding dimension mismatch. "
                f"Collection '{collection}' expects {expected_dim}-dim vectors "
                f"but received {received_dim}-dim vectors. "
                f"Either use the same embedding model that created this "
                f"collection, or delete the collection and re-index."
            ),
        )

class CollectionNotFoundError(VectorStoreError):
    """
    Raised when an operation targets a collection that does not exist.

    Distinct from VectorStoreError so callers can explicitly handle
    the "collection missing" case (e.g. prompt user to run ingestion).
    """

    def __init__(self, store: str, collection: str) -> None:
        super().__init__(
            store=store,
            operation="collection_lookup",
            collection=collection,
            reason=(
                f"Collection '{collection}' does not exist. "
                f"Run the ingestion pipeline to create it."
            ),
        )

__all__ = [
    "QueryResult",
    "CollectionInfo",
    "AddResult",
    "BaseVectorStore",
    "VectorStoreError",
    "DimensionMismatchError",
    "CollectionNotFoundError",
]