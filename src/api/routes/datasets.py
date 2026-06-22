"""
src/api/routes/datasets.py

Test dataset routes — synthetic generation, browsing, inline editing,
and CSV export.

Endpoints:
    POST   /api/v1/datasets/generate                  — generate a dataset from a collection
    GET    /api/v1/datasets                            — list datasets (filtered, paginated)
    GET    /api/v1/datasets/{dataset_id}               — full dataset, all pairs
    GET    /api/v1/datasets/{dataset_id}/metadata      — metadata only (fast)
    DELETE /api/v1/datasets/{dataset_id}               — delete a dataset
    PATCH  /api/v1/datasets/{dataset_id}/pairs/{pair_id} — edit question/ground_truth on one pair
    DELETE /api/v1/datasets/{dataset_id}/pairs/{pair_id} — remove one pair
    GET    /api/v1/datasets/{dataset_id}/export.csv     — download per-pair CSV

Why chunk sampling for generation uses a similarity-search query rather
than "fetch all chunks in the collection":

  BaseVectorStore (src/vectorstore/base.py) deliberately has no
  "list/scan all documents" method in its abstract interface — only
  query() (similarity search against an embedding) and
  get_document_by_id() (exact lookup). This is a real constraint
  inherited from that design, not an oversight here: not every
  possible BaseVectorStore implementation (a future managed/hosted
  vector DB, for instance) is guaranteed to support an efficient full
  collection scan, so the abstraction was never given one. The exact
  same pattern — embed a representative query, retrieve top_k chunks —
  is what was manually verified working end-to-end in this project's
  Level 3 integration test, so this route reuses that proven approach
  rather than inventing a new, untested retrieval path. seed_query
  lets the caller bias sampling toward a topic; it defaults to a
  generic phrase when omitted, which is an explicit, documented
  trade-off, not a hidden one.

Why GeminiDatasetGenerator is constructed inline per-request rather
than via a Depends() function in src/api/dependencies.py:

  Unlike EmbeddingManager (loads a 90MB model) or the judge evaluators
  inside EvaluationEngine (probe the judge model's availability at
  construction), GeminiDatasetGenerator.__init__ -> _initialise_model()
  only calls genai.configure() and constructs a GenerativeModel
  wrapper — no network call happens until generate_content() is
  actually invoked. It is also configured per-request from
  GeneratorConfig fields that come directly from the request body
  (n_pairs_per_chunk, max_pairs_total, temperature) — there is no
  single shared instance that could correctly serve every request's
  distinct configuration, so application-scoping it would provide no
  benefit while adding complexity.

Why editing a pair is restricted to status == PENDING:

  QAPair.set_answer() (src/dataset/schema.py) enforces that an answer
  is generated for a specific, fixed question. Allowing a user to edit
  pair.question after RAGPipeline has already answered the ORIGINAL
  question (status == ANSWERED or EVALUATED) would leave a stored
  answer and metric scores that no longer correspond to the question
  text being displayed — a silent correctness bug in the dashboard.
  Editing is therefore a pre-evaluation-only operation, matching this
  project's original design intent ("Dataset preview UI — Allow
  editing/deletion before running evaluation").
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.settings import Settings
from src.api.dependencies import (
    get_app_settings,
    get_dataset_store,
    get_embedding_manager,
    get_run_repository,
    get_vector_store,
)
from src.dataset.base import GeneratorConfig, GenerationStats
from src.dataset.generator import GeminiDatasetGenerator
from src.dataset.schema import (
    DatasetStatus,
    EvalDataset,
    GenerationMethod,
    QAPair,
    QAPairStatus,
)
from src.dataset.store import (
    DatasetIndexEntry,
    DatasetNotFoundError,
    DatasetStore,
    DatasetStoreError,
)
from src.storage.repository import RunRepository
from src.ingestion.base import Document
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.embeddings import EmbeddingManager

router = APIRouter()

# ===========================================================================
# CONSTANTS
# ===========================================================================

_DEFAULT_SEED_QUERY: str = (
    "key information, important facts, and significant details"
)

# ===========================================================================
# REQUEST SCHEMAS
# ===========================================================================

class GenerateDatasetRequest(BaseModel):
    """Request body for POST /generate."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(
        min_length=3,
        max_length=63,
        description="ChromaDB collection to sample chunks from.",
    )
    dataset_name: str = Field(
        min_length=1,
        max_length=255,
        description="Human-readable name for the generated dataset.",
    )
    description: str | None = Field(default=None, max_length=1000)
    seed_query: str | None = Field(
        default=None,
        description=(
            "Optional query used to bias chunk sampling toward a "
            "topic via similarity search. Defaults to a generic "
            "representative phrase if omitted."
        ),
    )
    sample_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks to sample from the collection.",
    )
    n_pairs_per_chunk: int = Field(default=1, ge=1, le=5)
    max_pairs_total: int = Field(default=20, ge=1, le=100)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    tags: list[str] = Field(default_factory=list)

class EditPairRequest(BaseModel):
    """
    Request body for PATCH /{dataset_id}/pairs/{pair_id}.

    At least one field must be provided — an empty patch is rejected
    explicitly rather than silently being a no-op.
    """

    model_config = ConfigDict(frozen=True)

    question: str | None = Field(default=None, min_length=5)
    ground_truth_answer: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def at_least_one_field(self) -> EditPairRequest:
        if self.question is None and self.ground_truth_answer is None:
            raise ValueError(
                "At least one of 'question' or 'ground_truth_answer' "
                "must be provided."
            )
        return self

# ===========================================================================
# RESPONSE SCHEMAS
# ===========================================================================

class GenerationStatsOut(BaseModel):
    """API-facing view of GenerationStats."""

    model_config = ConfigDict(frozen=True)

    chunks_attempted: int
    chunks_succeeded: int
    chunks_skipped: int
    chunks_failed: int
    pairs_generated: int
    pairs_rejected: int
    pairs_deduplicated: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float

    @classmethod
    def from_domain(cls, stats: GenerationStats) -> GenerationStatsOut:
        return cls(
            chunks_attempted=stats.chunks_attempted,
            chunks_succeeded=stats.chunks_succeeded,
            chunks_skipped=stats.chunks_skipped,
            chunks_failed=stats.chunks_failed,
            pairs_generated=stats.pairs_generated,
            pairs_rejected=stats.pairs_rejected,
            pairs_deduplicated=stats.pairs_deduplicated,
            total_input_tokens=stats.total_input_tokens,
            total_output_tokens=stats.total_output_tokens,
            total_latency_ms=round(stats.total_latency_ms, 1),
        )

class GenerateDatasetResponse(BaseModel):
    """Response for POST /generate."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    status: str
    total_pairs: int
    source_collection: str
    generation: GenerationStatsOut

class QAPairOut(BaseModel):
    """API-facing view of a single QAPair."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    ground_truth_answer: str
    source_file: str
    source_page: int | None
    chunk_index: int | None
    status: str
    generated_answer: str | None
    composite_score: float | None
    created_at: str

    @classmethod
    def from_domain(cls, pair: QAPair) -> QAPairOut:
        return cls(
            id=pair.id,
            question=pair.question,
            ground_truth_answer=pair.ground_truth_answer,
            source_file=pair.source_file,
            source_page=pair.source_page,
            chunk_index=pair.chunk_index,
            status=pair.status.value,
            generated_answer=pair.generated_answer,
            composite_score=pair.composite_score,
            created_at=pair.created_at.isoformat(),
        )

class DatasetSummaryOut(BaseModel):
    """API-facing view of a DatasetIndexEntry — for list views."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    generation_method: str
    source_collection: str | None
    source_files: list[str]
    generator_model: str | None
    total_pairs: int
    evaluated_pairs: int
    status: str
    tags: list[str]
    version: str
    completion_rate: float

    @classmethod
    def from_domain(cls, entry: DatasetIndexEntry) -> DatasetSummaryOut:
        return cls(
            id=entry.id,
            name=entry.name,
            description=entry.description,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            generation_method=entry.generation_method,
            source_collection=entry.source_collection,
            source_files=entry.source_files,
            generator_model=entry.generator_model,
            total_pairs=entry.total_pairs,
            evaluated_pairs=entry.evaluated_pairs,
            status=entry.status,
            tags=entry.tags,
            version=entry.version,
            completion_rate=round(entry.completion_rate, 4),
        )

class DatasetListOut(BaseModel):
    """Response for GET /datasets."""

    model_config = ConfigDict(frozen=True)

    datasets: list[DatasetSummaryOut]
    total: int
    limit: int
    offset: int

class DatasetDetailOut(BaseModel):
    """Response for GET /datasets/{dataset_id} — full dataset with pairs."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    generation_method: str
    source_collection: str | None
    source_files: list[str]
    generator_model: str | None
    status: str
    tags: list[str]
    version: str
    pairs: list[QAPairOut]

    @classmethod
    def from_domain(cls, dataset: EvalDataset) -> DatasetDetailOut:
        meta = dataset.metadata
        return cls(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            created_at=meta.created_at.isoformat(),
            updated_at=meta.updated_at.isoformat(),
            generation_method=meta.generation_method.value,
            source_collection=meta.source_collection,
            source_files=meta.source_files,
            generator_model=meta.generator_model,
            status=meta.status.value,
            tags=meta.tags,
            version=meta.version,
            pairs=[QAPairOut.from_domain(p) for p in dataset.pairs],
        )

class DeleteDatasetOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: str
    deleted: bool

class DeletePairOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: str
    pair_id: str
    deleted: bool
    remaining_pairs: int

# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

def _parse_status_filter(raw: str | None) -> DatasetStatus | None:
    """Validate a status query param, raising 400 with valid options on error."""
    if raw is None:
        return None
    try:
        return DatasetStatus(raw)
    except ValueError as exc:
        valid = [s.value for s in DatasetStatus]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{raw}'. Valid options: {valid}.",
        ) from exc

def _parse_generation_method_filter(
    raw: str | None,
) -> GenerationMethod | None:
    """Validate a generation_method query param, raising 400 on error."""
    if raw is None:
        return None
    try:
        return GenerationMethod(raw)
    except ValueError as exc:
        valid = [m.value for m in GenerationMethod]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid generation_method '{raw}'. "
                f"Valid options: {valid}."
            ),
        ) from exc

def _load_dataset_or_404(store: DatasetStore, dataset_id: str) -> EvalDataset:
    """
    Load a dataset by ID, converting DatasetNotFoundError/DatasetStoreError
    into the appropriate HTTPException.

    Both exception types are caught explicitly here rather than relying
    on global handlers in src/api/app.py — neither is currently
    registered there, and this keeps the route fully self-contained
    regardless of what gets added to app.py's handler list later.
    """
    try:
        return store.load(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatasetStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

def _find_pair_or_404(dataset: EvalDataset, pair_id: str) -> QAPair:
    """Find a pair within a dataset by ID, or raise 404."""
    pair = dataset.get_pair_by_id(pair_id)
    if pair is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Pair '{pair_id}' not found in dataset "
                f"'{dataset.id}'."
            ),
        )
    return pair

# ===========================================================================
# CHUNK SAMPLING
# ===========================================================================

def _sample_chunks_sync(
    embedding_manager: EmbeddingManager,
    vector_store: BaseVectorStore,
    collection_name: str,
    seed_query: str,
    sample_size: int,
) -> list[Document]:
    """
    Synchronous chunk-sampling helper, run inside an executor by the
    route handler below.

    Embeds seed_query and retrieves up to sample_size chunks via
    similarity search. vector_store.query() raises CollectionNotFoundError
    if collection_name does not exist — that propagates up through
    run_in_executor to the route, which lets src/api/app.py's
    registered CollectionNotFoundError handler produce a 404, exactly
    as src/api/routes/ingest.py's get_collection route already relies
    on the same handler for the identical exception type.
    """
    query_vector = embedding_manager.embed_query(seed_query)
    results = vector_store.query(
        query_embedding=query_vector,
        collection_name=collection_name,
        top_k=sample_size,
    )
    return [r.document for r in results]

# ===========================================================================
# ROUTES
# ===========================================================================

@router.post(
    "/generate",
    response_model=GenerateDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a synthetic Q&A dataset from a collection",
)
async def generate_dataset(
    request: GenerateDatasetRequest,
    settings: Settings = Depends(get_app_settings),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_store: BaseVectorStore = Depends(get_vector_store),
    store: DatasetStore = Depends(get_dataset_store),
    repo: RunRepository = Depends(get_run_repository),
) -> GenerateDatasetResponse:
    """
    Sample chunks from a collection and generate a synthetic Q&A
    dataset via Gemini, then persist it through DatasetStore.

    The full pipeline (embedding the seed query, similarity search,
    and the generator's sequential Gemini calls) is synchronous and
    blocking — wrapped in run_in_executor so it never stalls other
    concurrent requests, matching the pattern already established in
    src/api/routes/ingest.py for the equally blocking ingestion call.
    """
    loop = asyncio.get_event_loop()

    chunks = await loop.run_in_executor(
        None,
        lambda: _sample_chunks_sync(
            embedding_manager,
            vector_store,
            request.collection_name,
            request.seed_query or _DEFAULT_SEED_QUERY,
            request.sample_size,
        ),
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No chunks retrieved from collection "
                f"'{request.collection_name}'. The collection may be "
                f"empty, or no chunks matched the sampling query "
                f"closely enough."
            ),
        )

    logger.info(
        f"generate_dataset: Sampled {len(chunks)} chunks from "
        f"'{request.collection_name}'. Starting generation..."
    )

    generator = GeminiDatasetGenerator(
        config=GeneratorConfig(
            n_pairs_per_chunk=request.n_pairs_per_chunk,
            max_pairs_total=request.max_pairs_total,
            temperature=request.temperature,
        ),
        dataset_gen_config=settings.dataset_gen,
        gemini_api_key=settings.gemini.api_key,
    )

    try:
        dataset, gen_stats = await loop.run_in_executor(
            None,
            lambda: generator.generate(
                chunks=chunks,
                dataset_name=request.dataset_name,
                source_collection=request.collection_name,
            ),
        )
    except Exception as exc:
        # GenerationError is registered in src/api/app.py and produces
        # a 422 automatically — re-raise rather than catching here so
        # that single registered handler stays the one source of truth
        # for this error's HTTP representation.
        raise

    if request.description or request.tags:
        dataset.metadata = dataset.metadata.model_copy(
            update={
                "description": request.description,
                "tags": request.tags,
            }
        )

    await loop.run_in_executor(None, store.save, dataset)

    await repo.upsert_dataset_record(
        metadata=dataset.metadata,
        dataset_dir=store.base_dir / dataset.metadata.id,
    )

    logger.info(
        f"generate_dataset: Saved dataset '{dataset.id}' "
        f"({len(dataset.pairs)} pairs)."
    )

    return GenerateDatasetResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        status=dataset.metadata.status.value,
        total_pairs=len(dataset.pairs),
        source_collection=request.collection_name,
        generation=GenerationStatsOut.from_domain(gen_stats),
    )

@router.get(
    "",
    response_model=DatasetListOut,
    summary="List datasets with filtering and pagination",
)
async def list_datasets(
    status_filter: str | None = Query(default=None, alias="status"),
    generation_method: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    descending: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    store: DatasetStore = Depends(get_dataset_store),
) -> DatasetListOut:
    """
    List dataset summaries from the fast SQL-free index — no full
    dataset.json files are loaded, so this scales to many datasets
    without a corresponding slowdown (see DatasetStore.list_datasets'
    docstring on why index.json exists at all).
    """
    parsed_status = _parse_status_filter(status_filter)
    parsed_method = _parse_generation_method_filter(generation_method)

    loop = asyncio.get_event_loop()
    all_entries = await loop.run_in_executor(
        None,
        lambda: store.list_datasets(
            status_filter=parsed_status,
            generation_method_filter=parsed_method,
            tag_filter=tag,
            sort_by=sort_by,
            descending=descending,
        ),
    )

    page = all_entries[offset: offset + limit]

    return DatasetListOut(
        datasets=[DatasetSummaryOut.from_domain(e) for e in page],
        total=len(all_entries),
        limit=limit,
        offset=offset,
    )

@router.get(
    "/{dataset_id}",
    response_model=DatasetDetailOut,
    summary="Get a complete dataset with all Q&A pairs",
)
async def get_dataset(
    dataset_id: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> DatasetDetailOut:
    """Retrieve the full dataset, including every QAPair's full content."""
    loop = asyncio.get_event_loop()
    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, store, dataset_id
    )
    return DatasetDetailOut.from_domain(dataset)

@router.get(
    "/{dataset_id}/metadata",
    response_model=DatasetSummaryOut,
    summary="Get dataset metadata only (fast — no pairs loaded)",
)
async def get_dataset_metadata(
    dataset_id: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> DatasetSummaryOut:
    """
    Retrieve only metadata, via DatasetStore.load_metadata() which
    reads the lightweight metadata.json sidecar file rather than the
    full dataset.json — used by UI screens (e.g. the evaluation-run
    configuration page) that need to display a dataset's name and
    pair count without paying the cost of deserialising every pair.
    """
    loop = asyncio.get_event_loop()
    try:
        meta = await loop.run_in_executor(
            None, store.load_metadata, dataset_id
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return DatasetSummaryOut(
        id=meta.id,
        name=meta.name,
        description=meta.description,
        created_at=meta.created_at.isoformat(),
        updated_at=meta.updated_at.isoformat(),
        generation_method=meta.generation_method.value,
        source_collection=meta.source_collection,
        source_files=meta.source_files,
        generator_model=meta.generator_model,
        total_pairs=meta.total_pairs,
        evaluated_pairs=0,
        status=meta.status.value,
        tags=meta.tags,
        version=meta.version,
        completion_rate=0.0,
    )

@router.delete(
    "/{dataset_id}",
    response_model=DeleteDatasetOut,
    summary="Delete a dataset and all its files",
)
async def delete_dataset(
    dataset_id: str,
    store: DatasetStore = Depends(get_dataset_store),
    repo: RunRepository = Depends(get_run_repository),
) -> DeleteDatasetOut:
    """
    Permanently delete a dataset's directory (dataset.json,
    metadata.json, and all archived versions) and remove it from the
    index. Does not delete any RunRecord/EvalReport that referenced
    this dataset historically — those remain valid history entries.
    """
    loop = asyncio.get_event_loop()
    deleted = await loop.run_in_executor(None, store.delete, dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    await repo.delete_dataset_record(dataset_id)
    return DeleteDatasetOut(dataset_id=dataset_id, deleted=True)

@router.patch(
    "/{dataset_id}/pairs/{pair_id}",
    response_model=QAPairOut,
    summary="Edit a pair's question or ground truth answer",
)
async def edit_pair(
    dataset_id: str,
    pair_id: str,
    request: EditPairRequest,
    store: DatasetStore = Depends(get_dataset_store),
    repo: RunRepository = Depends(get_run_repository),
) -> QAPairOut:
    """
    Update question and/or ground_truth_answer on a single pair.

    Restricted to pairs with status == PENDING — see this file's
    module docstring for why editing a pair that has already been
    answered or evaluated would leave inconsistent stored state.
    """
    loop = asyncio.get_event_loop()
    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, store, dataset_id
    )
    pair = _find_pair_or_404(dataset, pair_id)

    if pair.status != QAPairStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Pair '{pair_id}' has status='{pair.status.value}'. "
                f"Only PENDING pairs (not yet answered or evaluated) "
                f"can be edited. Generate a new dataset version if "
                f"you need to change an already-answered question."
            ),
        )

    if request.question is not None:
        pair.question = request.question
    if request.ground_truth_answer is not None:
        pair.ground_truth_answer = request.ground_truth_answer

    await loop.run_in_executor(None, store.save, dataset)
    await repo.upsert_dataset_record(
        metadata=dataset.metadata,
        dataset_dir=store.base_dir / dataset.id,
    )

    logger.info(f"edit_pair: Updated pair '{pair_id}' in '{dataset_id}'.")
    return QAPairOut.from_domain(pair)

@router.delete(
    "/{dataset_id}/pairs/{pair_id}",
    response_model=DeletePairOut,
    summary="Remove a single pair from a dataset",
)
async def delete_pair(
    dataset_id: str,
    pair_id: str,
    store: DatasetStore = Depends(get_dataset_store),
    repo: RunRepository = Depends(get_run_repository),
) -> DeletePairOut:
    """
    Remove one pair and persist the updated dataset.

    metadata.total_pairs is explicitly recomputed here before saving —
    DatasetStore.save() does NOT auto-derive total_pairs from
    len(pairs) on every save (see its docstring: a mismatch only
    triggers a warning, by design, to tolerate datasets mid-update).
    This route performs that recomputation itself since pair removal
    is exactly the kind of update that must keep the count accurate.
    """
    loop = asyncio.get_event_loop()
    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, store, dataset_id
    )
    _find_pair_or_404(dataset, pair_id)

    dataset.pairs = [p for p in dataset.pairs if p.id != pair_id]
    dataset.metadata = dataset.metadata.model_copy(
        update={"total_pairs": len(dataset.pairs)}
    )

    if not dataset.pairs:
        logger.warning(
            f"delete_pair: '{dataset_id}' has zero pairs remaining "
            f"after deletion. Dataset is now empty but not "
            f"auto-deleted — use DELETE /{dataset_id} explicitly "
            f"if it should be removed entirely."
        )

    await loop.run_in_executor(None, store.save, dataset)
    await repo.upsert_dataset_record(
        metadata=dataset.metadata,
        dataset_dir=store.base_dir / dataset.id,
    )

    return DeletePairOut(
        dataset_id=dataset_id,
        pair_id=pair_id,
        deleted=True,
        remaining_pairs=len(dataset.pairs),
    )

@router.get(
    "/{dataset_id}/export.csv",
    summary="Download a dataset's pairs as CSV",
)
async def export_dataset_csv(
    dataset_id: str,
    store: DatasetStore = Depends(get_dataset_store),
) -> Response:
    """
    Generate a CSV in-memory from EvalDataset.to_csv_rows() and return
    it as a file download.

    Deliberately does NOT call DatasetStore.export_csv() — that method
    writes a file to the dataset's own directory on disk, which is the
    right behaviour for a CLI script but would mean every single API
    download silently leaves a new file on the server's filesystem.
    Building the CSV directly from to_csv_rows() into an in-memory
    buffer keeps this endpoint side-effect-free.
    """
    loop = asyncio.get_event_loop()
    dataset = await loop.run_in_executor(
        None, _load_dataset_or_404, store, dataset_id
    )

    rows = dataset.to_csv_rows()
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset '{dataset_id}' has no pairs to export.",
        )

    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in dataset.name
    )
    filename = f"{safe_name}_{dataset_id[-8:]}.csv"

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

__all__ = ["router"]