"""
src/api/routes/ingest.py

Document ingestion routes — file upload and ChromaDB collection
management.

Endpoints:
    POST   /api/v1/ingest/files                 — upload + ingest documents
    GET    /api/v1/ingest/collections            — list all collections
    GET    /api/v1/ingest/collections/{name}      — get one collection's info
    DELETE /api/v1/ingest/collections/{name}      — delete a collection
    GET    /api/v1/ingest/supported-formats       — list loader-supported extensions

Why this route constructs its own request-scoped IngestionPipeline
dependency (_get_ingestion_pipeline below) instead of adding a fifth
field to AppState in src/api/dependencies.py:

  IngestionPipeline is NOT expensive to construct — its constituent
  parts are either already application-scoped (EmbeddingManager,
  BaseVectorStore, retrieved via the existing get_embedding_manager
  and get_vector_store dependencies — no model reload, no new ChromaDB
  connection) or genuinely cheap to build fresh per request
  (LoaderRegistry is just four loader instances; RecursiveChunker
  wraps a stdlib text splitter — neither loads a model or opens a
  connection). This mirrors the get_dataset_store pattern in
  dependencies.py exactly: cheap composition is built per request;
  expensive resources (the embedding model, the DB connection) are
  retrieved from app.state, never rebuilt.

Why uploaded files are written to disk before ingestion, rather than
ingesting from the in-memory UploadFile stream directly:

  BaseLoader.load() (src/ingestion/base.py) has an explicit, deliberate
  contract: it accepts a Path and validates the file exists on disk.
  This was a design choice when the ingestion layer was first written
  — it keeps every loader implementation symmetric (PDFLoader,
  TXTLoader, HTMLLoader, DOCXLoader all just open a path) and means
  the exact same loaders work identically whether called from this API
  route, a CLI script, or a Streamlit file uploader. Persisting to
  settings.storage.raw_docs_dir before calling the pipeline is the one
  small adaptation this route makes to satisfy that existing contract
  — it does not change anything in src/ingestion/.

Why filenames are preserved exactly (not randomised/uuid-prefixed) when
saved to disk:

  ChromaVectorStore._generate_document_id() (src/vectorstore/chroma.py)
  derives a deterministic chunk ID from source_file + page_number +
  chunk_index + a content prefix. Re-uploading the identical file to
  the identical collection must produce the identical source_file
  value for upsert-based idempotency to work as designed — randomising
  the saved filename here would silently break that guarantee and turn
  every re-upload into a duplicate-chunk-producing operation instead of
  an in-place update.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings
from src.api.dependencies import get_app_settings, get_embedding_manager, get_vector_store
from src.ingestion.base import LoaderRegistry
from src.ingestion.chunker import ChunkerConfig, RecursiveChunker
from src.ingestion.loaders import DOCXLoader, HTMLLoader, PDFLoader, TXTLoader
from src.ingestion.pipeline import FileIngestionResult, IngestionPipeline, IngestionResult
from src.vectorstore.base import BaseVectorStore, CollectionInfo
from src.vectorstore.embeddings import EmbeddingManager

router = APIRouter()

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Safety cap on files per upload request — mirrors the philosophy of
# ComparisonRunner.max_total_runs and EvalRunConfig.max_questions_per_run:
# fail fast with a clear 400 before doing any work, rather than silently
# accepting and slowly choking on an unreasonable request (e.g. a
# misconfigured client looping a single-file upload 500 times).
_MAX_FILES_PER_REQUEST: int = 50

# ChromaDB collection naming rule (matches chromadb's own validation):
# 3-63 chars, alphanumeric/underscore/hyphen, must start and end with
# an alphanumeric character. Validated here so a malformed
# collection_name produces an immediate, clear 400 from THIS route
# rather than an opaque error surfacing from inside ChromaVectorStore
# several calls deep.
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")

# ===========================================================================
# RESPONSE SCHEMAS
# ===========================================================================

class FileIngestionResultOut(BaseModel):
    """API-facing view of FileIngestionResult — one row per uploaded file."""

    model_config = ConfigDict(frozen=True)

    filename: str
    status: str
    document_count: int
    chunk_count: int
    added_to_store: int
    error_message: str | None
    loader_class: str
    latency_ms: float

    @classmethod
    def from_domain(cls, result: FileIngestionResult) -> FileIngestionResultOut:
        return cls(
            filename=result.path.name,
            status=result.status,
            document_count=result.document_count,
            chunk_count=result.chunk_count,
            added_to_store=result.added_to_store,
            error_message=result.error_message,
            loader_class=result.loader_class,
            latency_ms=round(result.latency_ms, 1),
        )

class IngestionResultOut(BaseModel):
    """API-facing view of IngestionResult — the full job summary."""

    model_config = ConfigDict(frozen=True)

    collection_name: str
    run_id: str
    started_at: str
    completed_at: str | None
    total_files_attempted: int
    succeeded_files: int
    failed_files: int
    skipped_files: int
    total_documents_loaded: int
    total_chunks_produced: int
    total_chunks_stored: int
    success_rate: float
    total_latency_ms: float
    files: list[FileIngestionResultOut]

    @classmethod
    def from_domain(cls, result: IngestionResult) -> IngestionResultOut:
        return cls(
            collection_name=result.collection_name,
            run_id=result.run_id,
            started_at=result.started_at,
            completed_at=result.completed_at,
            total_files_attempted=result.total_files_attempted,
            succeeded_files=result.succeeded_files,
            failed_files=result.failed_files,
            skipped_files=result.skipped_files,
            total_documents_loaded=result.total_documents_loaded,
            total_chunks_produced=result.total_chunks_produced,
            total_chunks_stored=result.total_chunks_stored,
            success_rate=round(result.success_rate, 4),
            total_latency_ms=round(result.total_latency_ms, 1),
            files=[
                FileIngestionResultOut.from_domain(f)
                for f in result.file_results
            ],
        )

class CollectionInfoOut(BaseModel):
    """API-facing view of CollectionInfo."""

    model_config = ConfigDict(frozen=True)

    name: str
    document_count: int
    embedding_dimension: int | None
    metadata: dict[str, Any]
    is_empty: bool

    @classmethod
    def from_domain(cls, info: CollectionInfo) -> CollectionInfoOut:
        return cls(
            name=info.name,
            document_count=info.document_count,
            embedding_dimension=info.embedding_dimension,
            metadata=info.metadata,
            is_empty=info.is_empty,
        )

class CollectionListOut(BaseModel):
    """Response for GET /collections."""

    model_config = ConfigDict(frozen=True)

    collections: list[CollectionInfoOut]
    total_collections: int
    total_documents: int

class DeleteCollectionOut(BaseModel):
    """Response for DELETE /collections/{name}."""

    model_config = ConfigDict(frozen=True)

    collection_name: str
    deleted: bool

class SupportedFormatsOut(BaseModel):
    """Response for GET /supported-formats."""

    model_config = ConfigDict(frozen=True)

    extensions: list[str]

# ===========================================================================
# REQUEST-SCOPED INGESTION PIPELINE DEPENDENCY
# ===========================================================================
def get_ingestion_pipeline(
    settings: Settings = Depends(get_app_settings),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> IngestionPipeline:
    """
    Construct a request-scoped IngestionPipeline from shared,
    application-scoped resources.

    embedding_manager and vector_store are retrieved from app.state
    (see src/api/dependencies.py) — never reconstructed here. Only the
    LoaderRegistry and RecursiveChunker, both stateless and cheap, are
    built fresh per request.

    This is a deliberate departure from IngestionPipeline.build_default(),
    which constructs its OWN EmbeddingManager and ChromaVectorStore —
    that factory is correct for scripts and tests that want a fully
    self-contained pipeline, but would reload the ~90MB HuggingFace
    model on every single API request if used here.
    """
    registry = LoaderRegistry()
    registry.register(PDFLoader(settings.ingestion))
    registry.register(TXTLoader(settings.ingestion))
    registry.register(HTMLLoader(settings.ingestion))
    registry.register(DOCXLoader(settings.ingestion))

    chunker = RecursiveChunker(
        ChunkerConfig.from_ingestion_config(settings.ingestion)
    )

    return IngestionPipeline(
        registry=registry,
        chunker=chunker,
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        config=settings.ingestion,
    )

# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

def _validate_collection_name(collection_name: str) -> None:
    """
    Validate collection_name against ChromaDB's naming rules before
    attempting any ingestion work.

    Raises HTTPException(400) with a specific, actionable message
    rather than letting an invalid name surface as an opaque
    VectorStoreError several calls deep inside ChromaVectorStore.
    """
    if not _COLLECTION_NAME_PATTERN.match(collection_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid collection_name '{collection_name}'. "
                f"Must be 3-63 characters, contain only letters, "
                f"numbers, underscores, and hyphens, and start/end "
                f"with a letter or number."
            ),
        )

async def _save_upload(upload: UploadFile, destination: Path) -> Path:
    """
    Persist one UploadFile to disk at the given destination path.

    Reads the upload's content asynchronously (UploadFile.read() is
    already non-blocking), then writes to disk via run_in_executor —
    consistent with this codebase's established pattern of never
    letting synchronous I/O block the event loop directly (see
    ComparisonRunner._run_single_config wrapping RAGPipeline's
    synchronous answer_dataset() call the same way).
    """
    content = await upload.read()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, destination.write_bytes, content)
    return destination

# ===========================================================================
# ROUTES
# ===========================================================================

@router.post(
    "/files",
    response_model=IngestionResultOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest documents into a ChromaDB collection",
)
async def ingest_files(
    collection_name: str = Form(
        ...,
        description=(
            "Target ChromaDB collection. Created automatically if it "
            "does not yet exist. Re-using an existing collection name "
            "upserts matching content rather than duplicating it."
        ),
    ),
    upsert: bool = Form(
        True,
        description=(
            "If true (default), re-ingesting identical content updates "
            "existing chunks in place. If false, duplicate chunks "
            "(by deterministic ID) are skipped rather than updated."
        ),
    ),
    files: list[UploadFile] = File(
        ...,
        description=(
            "One or more documents to ingest. Supported formats: "
            "pdf, txt, text, md, markdown, html, htm, xhtml, docx."
        ),
    ),
    settings: Settings = Depends(get_app_settings),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestionResultOut:
    """
    Upload one or more documents and run them through the full
    ingestion pipeline (load -> chunk -> embed -> store).

    Per-file failures (corrupt PDF, unsupported extension, empty file)
    do NOT abort the batch — IngestionPipeline.ingest_files() isolates
    failures per file by design (see src/ingestion/pipeline.py). The
    response's `files` array always reports every uploaded file's
    individual outcome, even when collection-level status is overall
    "succeeded" for most files and "failed" for a few.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in the upload request.",
        )

    if len(files) > _MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many files in one request: {len(files)} "
                f"(max {_MAX_FILES_PER_REQUEST}). "
                f"Split into multiple upload requests."
            ),
        )

    _validate_collection_name(collection_name)

    # Persist uploads under a collection-specific subdirectory so
    # filenames are preserved exactly (required for upsert idempotency
    # via ChromaVectorStore's deterministic ID generation) while still
    # avoiding cross-collection filename collisions on disk.
    upload_dir = settings.storage.raw_docs_dir / collection_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for upload in files:
        if not upload.filename:
            logger.warning(
                "ingest_files: Skipping an upload with no filename."
            )
            continue
        destination = upload_dir / upload.filename
        saved_path = await _save_upload(upload, destination)
        saved_paths.append(saved_path)

    if not saved_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files were saved from the upload request.",
        )

    logger.info(
        f"ingest_files: Saved {len(saved_paths)} file(s) to "
        f"'{upload_dir}'. Starting ingestion into "
        f"collection='{collection_name}'."
    )

    # IngestionPipeline.ingest_files() is fully synchronous internally
    # (sequential load -> chunk -> batched embed -> store). Wrapped in
    # run_in_executor so a large multi-file upload doesn't block the
    # event loop for other concurrent requests — same rationale as
    # ComparisonRunner wrapping RAGPipeline.answer_dataset().
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.ingest_files(
            files=saved_paths,
            collection_name=collection_name,
            upsert=upsert,
        ),
    )

    return IngestionResultOut.from_domain(result)

@router.get(
    "/collections",
    response_model=CollectionListOut,
    summary="List all ChromaDB collections",
)
async def list_collections(
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> CollectionListOut:
    """
    List every collection currently in the vector store, with document
    counts and embedding dimension.

    Used by the Streamlit dataset-generation page to populate the
    "source collection" dropdown, and by the comparison-run
    configuration page to populate available retrieval targets.
    """
    collections = vector_store.list_collections()
    return CollectionListOut(
        collections=[
            CollectionInfoOut.from_domain(c) for c in collections
        ],
        total_collections=len(collections),
        total_documents=sum(c.document_count for c in collections),
    )

@router.get(
    "/collections/{collection_name}",
    response_model=CollectionInfoOut,
    summary="Get details for one collection",
)
async def get_collection(
    collection_name: str,
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> CollectionInfoOut:
    """
    Retrieve metadata for a single collection.

    Raises 404 via CollectionNotFoundError's registered exception
    handler (src/api/app.py) if the collection does not exist —
    this route does not need its own try/except for that case.
    """
    info = vector_store.get_collection_info(collection_name)
    return CollectionInfoOut.from_domain(info)

@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionOut,
    summary="Delete a collection and all its documents",
)
async def delete_collection(
    collection_name: str,
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> DeleteCollectionOut:
    """
    Permanently delete a collection and every chunk stored in it.

    Irreversible. Does NOT delete any DatasetRecord/EvalReport that
    referenced this collection_name historically — those remain valid
    history entries; only future retrieval against this collection
    name will fail with CollectionNotFoundError until it is re-ingested.
    """
    deleted = vector_store.delete_collection(collection_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' does not exist.",
        )
    return DeleteCollectionOut(
        collection_name=collection_name,
        deleted=True,
    )

@router.get(
    "/supported-formats",
    response_model=SupportedFormatsOut,
    summary="List file extensions supported by the ingestion pipeline",
)
async def supported_formats(
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> SupportedFormatsOut:
    """
    Report every file extension the ingestion pipeline can currently
    load, sourced live from the constructed LoaderRegistry rather than
    a hardcoded list — guaranteed to stay correct if a loader is added
    or removed from get_ingestion_pipeline() above.
    """
    return SupportedFormatsOut(
        extensions=sorted(pipeline.supported_extensions)
    )

__all__ = ["router"]