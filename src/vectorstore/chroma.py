"""
src/vectorstore/chroma.py

ChromaVectorStore — ChromaDB implementation of BaseVectorStore.

Design decisions:
  - PersistentClient only. In-memory ChromaDB is never used in this
    project — data must survive process restarts between ingestion
    and evaluation runs.
  - Document IDs are deterministic: sha256(source_file + page_number
    + chunk_index). Same content always produces the same ID so
    re-ingesting a document is idempotent (upsert=True by default).
  - Cosine distance is the default metric. score = 1 - distance
    maps cleanly to [0, 1] similarity for normalised vectors.
  - All ChromaDB exceptions are caught and re-raised as typed
    VectorStoreError subclasses. No ChromaDB type ever leaks
    past this module's boundary.
  - Collection metadata stores embedding_model and embedding_dim
    at creation time. DimensionMismatchError is raised before
    any insert if the manager's dimension conflicts with the
    collection's recorded dimension.

Thread safety:
    ChromaDB's PersistentClient is not thread-safe for concurrent
    writes. The IngestionPipeline runs ingestion sequentially.
    Concurrent reads (query) are safe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.settings import ChromaConfig
from src.ingestion.base import Document, DocumentMetadata
from src.vectorstore.base import (
    AddResult,
    BaseVectorStore,
    CollectionInfo,
    CollectionNotFoundError,
    DimensionMismatchError,
    QueryResult,
    VectorStoreError,
)
from src.vectorstore.embeddings import EmbeddingManager

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store.

    Implements every abstract method of BaseVectorStore.
    No method here is called directly by business logic —
    all callers depend on BaseVectorStore.

    Collection naming convention:
        Collections are named by the caller (IngestionPipeline passes
        a collection name derived from the source document set).
        The default collection name comes from ChromaConfig.

    Embedding dimension enforcement:
        On first add_documents() to a new collection, the embedding
        dimension is recorded in the collection's metadata.
        All subsequent add_documents() and query() calls to that
        collection validate against this recorded dimension.
        This prevents silent corruption from mixed-model ingestion.
    """

    # Metadata key used to store the embedding dimension in ChromaDB
    _DIM_METADATA_KEY: str = "embedding_dim"
    _MODEL_METADATA_KEY: str = "embedding_model"
    _DESCRIPTION_METADATA_KEY: str = "description"

    def __init__(
        self,
        config: ChromaConfig,
        embedding_manager: EmbeddingManager,
    ) -> None:
        """
        Initialise ChromaDB persistent client.

        Args:
            config:            ChromaConfig — persist_dir, default_collection.
            embedding_manager: Used to validate dimensions and reconstruct
                               Document objects from stored metadata.
                               NOT used to compute embeddings here —
                               that is the caller's responsibility.

        Raises:
            VectorStoreError: if ChromaDB client cannot be initialised
                              (e.g. corrupt persist_dir, permission error).
        """
        self._config = config
        self._embedding_manager = embedding_manager
        self._persist_dir = Path(config.persist_dir)
        self._default_collection = config.default_collection
        self._client: Any = None   # chromadb.PersistentClient

        self._initialise_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_client(self) -> None:
        """Create ChromaDB PersistentClient and validate connectivity."""
        try:
            import chromadb

            self._persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
            )
            
            # self._client = chromadb.Client(chromadb.Settings(
            #     chroma_db_impl="duckd+parquet",
            #     persist_directory=str(self._persist_dir),
            #     anonymized_telemetry=False,))
            
            # Heartbeat validates the client is functional
            self._client.heartbeat()

            logger.info(
                f"ChromaVectorStore initialised. "
                f"persist_dir='{self._persist_dir}', "
                f"existing_collections={len(self._client.list_collections())}"
            )
        except ImportError as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="initialise",
                collection="",
                reason=(
                    "chromadb is not installed. "
                    "Run: poetry add chromadb"
                ),
                original_exception=exc,
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="initialise",
                collection="",
                reason=(
                    f"Failed to initialise ChromaDB client at "
                    f"'{self._persist_dir}': {exc}"
                ),
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # WRITE OPERATIONS
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        collection_name: str,
        upsert: bool = True,
    ) -> AddResult:
        """
        Store documents and their embeddings in a ChromaDB collection.

        Generates deterministic IDs from document content so that
        re-ingesting the same file is idempotent when upsert=True.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"add_documents: len(documents)={len(documents)} must equal "
                f"embeddings.shape[0]={embeddings.shape[0]}."
            )

        if len(documents) == 0:
            logger.warning(
                f"add_documents called with empty documents list "
                f"for collection '{collection_name}'. No-op."
            )
            return AddResult(
                collection_name=collection_name,
                added_count=0,
                skipped_count=0,
                failed_count=0,
            )

        # Validate / record embedding dimension
        self._validate_embedding_dimension(
            collection_name=collection_name,
            incoming_dim=embeddings.shape[1],
        )

        collection = self._get_or_create_chroma_collection(
            collection_name=collection_name,
            embedding_dim=embeddings.shape[1],
        )

        # Build ChromaDB batch payload
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents_text: list[str] = []
        embeddings_list: list[list[float]] = []
        failed_ids: list[str] = []
        failed_errors: list[str] = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            try:
                doc_id = self._generate_document_id(doc, i)
                ids.append(doc_id)
                metadatas.append(doc.metadata.to_dict())
                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())
            except Exception as exc:
                logger.error(
                    f"Failed to prepare document at index {i} "
                    f"for collection '{collection_name}': {exc}"
                )
                failed_ids.append(f"index_{i}")
                failed_errors.append(str(exc))

        if not ids:
            return AddResult(
                collection_name=collection_name,
                added_count=0,
                skipped_count=0,
                failed_count=len(failed_ids),
                failed_ids=failed_ids,
                errors=failed_errors,
            )

        # Execute ChromaDB write
        try:
            if upsert:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text,
                )
                added_count = len(ids)
                skipped_count = 0
            else:
                # Check which IDs already exist
                existing = collection.get(ids=ids, include=[])
                existing_ids = set(existing["ids"])
                new_ids: list[str] = []
                new_meta: list[dict[str, Any]] = []
                new_docs: list[str] = []
                new_embs: list[list[float]] = []

                for doc_id, meta, doc_text, emb in zip(
                    ids, metadatas, documents_text, embeddings_list
                ):
                    if doc_id in existing_ids:
                        skipped_count = skipped_count if hasattr(
                            self, "_skip_count"
                        ) else 0
                    else:
                        new_ids.append(doc_id)
                        new_meta.append(meta)
                        new_docs.append(doc_text)
                        new_embs.append(emb)

                skipped_count = len(ids) - len(new_ids)
                added_count = len(new_ids)

                if new_ids:
                    collection.add(
                        ids=new_ids,
                        embeddings=new_embs,
                        metadatas=new_meta,
                        documents=new_docs,
                    )

            logger.info(
                f"add_documents complete. "
                f"collection='{collection_name}', "
                f"added={added_count}, "
                f"skipped={skipped_count}, "
                f"failed={len(failed_ids)}, "
                f"total_in_collection={collection.count()}"
            )

            return AddResult(
                collection_name=collection_name,
                added_count=added_count,
                skipped_count=skipped_count,
                failed_count=len(failed_ids),
                document_ids=ids,
                failed_ids=failed_ids,
                errors=failed_errors,
            )

        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="add_documents",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    def delete_documents(
        self,
        document_ids: list[str],
        collection_name: str,
    ) -> int:
        """Delete documents by ID. Returns count of deleted documents."""
        collection = self._get_existing_collection(collection_name)

        try:
            # ChromaDB get() returns only existing IDs
            existing = collection.get(ids=document_ids, include=[])
            existing_ids = existing["ids"]

            if not existing_ids:
                logger.warning(
                    f"delete_documents: none of {len(document_ids)} "
                    f"requested IDs exist in '{collection_name}'."
                )
                return 0

            collection.delete(ids=existing_ids)
            logger.info(
                f"Deleted {len(existing_ids)} documents "
                f"from '{collection_name}'."
            )
            return len(existing_ids)

        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="delete_documents",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection. Returns True if deleted."""
        if not self.collection_exists(collection_name):
            logger.warning(
                f"delete_collection: '{collection_name}' does not exist."
            )
            return False

        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'.")
            return True
        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="delete_collection",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # READ OPERATIONS
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Retrieve top_k most similar documents to the query vector.

        ChromaDB returns cosine distances in [0, 2] for unnormalised
        vectors and [0, 1] for normalised vectors.
        EmbeddingManager normalises HuggingFace vectors at encode time.
        Google vectors are not explicitly normalised but are close to
        unit length in practice.

        Similarity score = 1 - distance. Clipped to [0, 1].
        """
        collection = self._get_existing_collection(collection_name)

        # Validate query dimension
        self._validate_query_dimension(
            collection_name=collection_name,
            query_dim=query_embedding.shape[0],
        )

        # Cap top_k at collection size to avoid ChromaDB warnings
        doc_count = collection.count()
        if doc_count == 0:
            logger.warning(
                f"query: collection '{collection_name}' is empty."
            )
            return []

        effective_top_k = min(top_k, doc_count)

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": effective_top_k,
                "include": ["documents", "metadatas", "distances"],
            }

            if filter_metadata:
                query_kwargs["where"] = filter_metadata

            raw = collection.query(**query_kwargs)

        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="query",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

        # Parse results
        results: list[QueryResult] = []

        if not raw["documents"] or not raw["documents"][0]:
            return results

        raw_docs = raw["documents"][0]
        raw_metas = raw["metadatas"][0]
        raw_distances = raw["distances"][0]
        raw_ids = raw["ids"][0]

        for rank, (doc_id, content, meta, distance) in enumerate(
            zip(raw_ids, raw_docs, raw_metas, raw_distances), start=1
        ):
            # Convert distance → similarity, clip to [0.0, 1.0]
            similarity = float(np.clip(1.0 - distance, 0.0, 1.0))

            if similarity < score_threshold:
                continue

            document = self._reconstruct_document(
                doc_id=doc_id,
                content=content,
                meta=meta,
            )

            results.append(
                QueryResult(
                    document=document,
                    similarity_score=similarity,
                    distance=float(distance),
                    rank=rank,
                    collection_name=collection_name,
                )
            )

        logger.debug(
            f"query: collection='{collection_name}', "
            f"top_k={top_k}, "
            f"returned={len(results)}, "
            f"threshold={score_threshold}"
        )

        return results

    def get_document_by_id(
        self,
        document_id: str,
        collection_name: str,
    ) -> Document | None:
        """Retrieve a single document by exact ID. Returns None if not found."""
        collection = self._get_existing_collection(collection_name)

        try:
            result = collection.get(
                ids=[document_id],
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="get_document_by_id",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

        if not result["ids"]:
            return None

        return self._reconstruct_document(
            doc_id=result["ids"][0],
            content=result["documents"][0],
            meta=result["metadatas"][0],
        )

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """Return metadata about a collection."""
        collection = self._get_existing_collection(collection_name)

        try:
            meta = collection.metadata or {}
            return CollectionInfo(
                name=collection_name,
                document_count=collection.count(),
                embedding_dimension=meta.get(self._DIM_METADATA_KEY),
                metadata=meta,
            )
        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="get_collection_info",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # ADMIN OPERATIONS
    # ------------------------------------------------------------------

    def list_collections(self) -> list[CollectionInfo]:
        """Return CollectionInfo for all existing collections."""
        try:
            chroma_collections = self._client.list_collections()
            result: list[CollectionInfo] = []

            for col in chroma_collections:
                meta = col.metadata or {}
                result.append(
                    CollectionInfo(
                        name=col.name,
                        document_count=col.count(),
                        embedding_dimension=meta.get(
                            self._DIM_METADATA_KEY
                        ),
                        metadata=meta,
                    )
                )
            return result

        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="list_collections",
                collection="",
                reason=str(exc),
                original_exception=exc,
            ) from exc

    def collection_exists(self, collection_name: str) -> bool:
        """Return True if collection exists. Never raises."""
        try:
            self._client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """Return or create a collection. Idempotent."""
        collection = self._get_or_create_chroma_collection(
            collection_name=collection_name,
            embedding_dim=self._embedding_manager.dimension,
            extra_metadata=metadata,
        )
        col_meta = collection.metadata or {}
        return CollectionInfo(
            name=collection_name,
            document_count=collection.count(),
            embedding_dimension=col_meta.get(self._DIM_METADATA_KEY),
            metadata=col_meta,
        )

    def count(self, collection_name: str) -> int:
        """Return document count in a collection."""
        collection = self._get_existing_collection(collection_name)
        return collection.count()

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _get_or_create_chroma_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Get existing or create new ChromaDB collection.

        Records embedding_dim and model in collection metadata on
        creation. Existing collections are returned as-is.
        """
        try:
            meta: dict[str, Any] = {
                self._DIM_METADATA_KEY: embedding_dim,
                self._MODEL_METADATA_KEY: self._embedding_manager.model_name,
                self._DESCRIPTION_METADATA_KEY: (
                    "RAG Eval Bench document embeddings"
                ),
            }
            if extra_metadata:
                meta.update(extra_metadata)

            return self._client.get_or_create_collection(
                name=collection_name,
                metadata=meta,
            )
        except Exception as exc:
            raise VectorStoreError(
                store="ChromaVectorStore",
                operation="get_or_create_collection",
                collection=collection_name,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    def _get_existing_collection(self, collection_name: str) -> Any:
        """
        Get an existing ChromaDB collection.

        Raises CollectionNotFoundError (not ChromaDB's internal error)
        so callers get an actionable message.
        """
        try:
            return self._client.get_collection(collection_name)
        except Exception:
            raise CollectionNotFoundError(
                store="ChromaVectorStore",
                collection=collection_name,
            )

    def _validate_embedding_dimension(
        self,
        collection_name: str,
        incoming_dim: int,
    ) -> None:
        """
        Validate incoming embedding dimension against collection's
        recorded dimension (if collection already exists).

        Raises DimensionMismatchError before any write if dimensions
        conflict — prevents silent vector space corruption.
        """
        if not self.collection_exists(collection_name):
            return   # New collection — no existing dimension to check

        try:
            collection = self._client.get_collection(collection_name)
            meta = collection.metadata or {}
            stored_dim = meta.get(self._DIM_METADATA_KEY)

            if stored_dim is not None and int(stored_dim) != incoming_dim:
                raise DimensionMismatchError(
                    store="ChromaVectorStore",
                    collection=collection_name,
                    expected_dim=int(stored_dim),
                    received_dim=incoming_dim,
                )
        except DimensionMismatchError:
            raise
        except Exception as exc:
            logger.warning(
                f"Could not validate embedding dimension for "
                f"'{collection_name}': {exc}. Proceeding anyway."
            )

    def _validate_query_dimension(
        self,
        collection_name: str,
        query_dim: int,
    ) -> None:
        """Validate query vector dimension against collection metadata."""
        try:
            collection = self._client.get_collection(collection_name)
            meta = collection.metadata or {}
            stored_dim = meta.get(self._DIM_METADATA_KEY)

            if stored_dim is not None and int(stored_dim) != query_dim:
                raise DimensionMismatchError(
                    store="ChromaVectorStore",
                    collection=collection_name,
                    expected_dim=int(stored_dim),
                    received_dim=query_dim,
                )
        except DimensionMismatchError:
            raise
        except Exception:
            pass   # Non-fatal — let the query attempt and fail naturally

    def _generate_document_id(
        self,
        doc: Document,
        fallback_index: int,
    ) -> str:
        """
        Generate a deterministic document ID from content fingerprint.

        Uses sha256 of (source_file + page_number + chunk_index + content).
        Truncated to 16 hex chars for readability while maintaining
        collision resistance at the scale of any single RAG corpus.

        Deterministic IDs make re-ingestion idempotent:
          - Same file re-ingested → same IDs → upsert updates in place
          - No duplicate documents from repeated pipeline runs
        """
        fingerprint_parts = [
            doc.metadata.source_file,
            str(doc.metadata.page_number or ""),
            str(doc.chunk_index or fallback_index),
            doc.page_content[:200],   # First 200 chars for uniqueness
        ]
        fingerprint = "|".join(fingerprint_parts)
        hash_hex = hashlib.sha256(
            fingerprint.encode("utf-8")
        ).hexdigest()
        return f"doc_{hash_hex[:16]}"

    def _reconstruct_document(
        self,
        doc_id: str,
        content: str,
        meta: dict[str, Any],
    ) -> Document:
        """
        Reconstruct a Document from ChromaDB stored fields.

        ChromaDB stores metadata as flat dict[str, str|int|float|bool].
        This method reconstructs the typed DocumentMetadata from
        the flattened representation stored by DocumentMetadata.to_dict().
        """
        # Parse extra field back from JSON string
        extra_raw = meta.get("extra", "{}")
        try:
            extra = json.loads(extra_raw) if isinstance(
                extra_raw, str
            ) else {}
        except json.JSONDecodeError:
            extra = {}

        # Restore None for empty-string sentinel values
        def _or_none(val: Any) -> Any:
            return None if val == "" else val

        metadata = DocumentMetadata(
            source_file=str(meta.get("source_file", "")),
            source_path=str(meta.get("source_path", "")),
            file_type=str(meta.get("file_type", "")),
            file_size_bytes=int(meta.get("file_size_bytes", 0)),
            total_pages=_or_none(meta.get("total_pages")),
            page_number=_or_none(meta.get("page_number")),
            title=_or_none(meta.get("title")),
            author=_or_none(meta.get("author")),
            created_at=_or_none(meta.get("created_at")),
            loader_class=str(meta.get("loader_class", "")),
            extra=extra,
        )

        # Extract chunk fields if present
        chunk_index_raw = meta.get("chunk_index")
        chunk_of_raw = meta.get("chunk_of")
        parent_doc_id = _or_none(meta.get("parent_doc_id"))

        return Document(
            page_content=content,
            metadata=metadata,
            chunk_index=int(chunk_index_raw)
            if chunk_index_raw not in (None, "")
            else None,
            chunk_of=int(chunk_of_raw)
            if chunk_of_raw not in (None, "")
            else None,
            parent_doc_id=parent_doc_id,
        )

    def __repr__(self) -> str:
        try:
            n_collections = len(self._client.list_collections())
        except Exception:
            n_collections = -1
        return (
            f"ChromaVectorStore("
            f"persist_dir='{self._persist_dir}', "
            f"collections={n_collections})"
        )

__all__ = ["ChromaVectorStore"]