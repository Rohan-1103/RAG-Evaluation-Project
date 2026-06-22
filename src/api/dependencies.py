"""
src/api/dependencies.py

FastAPI dependency injection wiring — the single import surface for
every route handler's `Depends(...)` parameters.

Two distinct dependency lifetimes are used here, and the distinction
matters for correctness, not just style:

  REQUEST-SCOPED (constructed fresh per request, via Depends):
    - Database session (get_db)
    - RunRepository (get_run_repository) — wraps the per-request session
    - DatasetStore (get_dataset_store) — cheap, stateless, file-based

  APPLICATION-SCOPED (constructed once at startup, stored on app.state,
  retrieved per-request via Request):
    - EmbeddingManager  — loads a ~90MB HuggingFace model into memory.
                          Reconstructing this per request would reload
                          the model on every single API call.
    - BaseVectorStore   — holds a ChromaDB PersistentClient connection.
    - RAGPipeline       — wraps the above two plus a Gemini client cache.
    - EvaluationEngine  — constructs and holds all 4 judge evaluators.
    - ComparisonRunner  — wraps RAGPipeline + EvaluationEngine.

Why application-scoped objects live on app.state and not module-level
globals (unlike get_settings()/get_model_registry() which use
lru_cache): FastAPI's TestClient and pytest fixtures construct a fresh
app instance per test session. Module-level globals would leak state
across test runs and make parallel test execution unsafe. app.state is
explicitly scoped to one app instance, constructed in app.py's lifespan
context manager, and torn down with it.

Routes NEVER import from config/, src/storage/database.py, or any
src/*/pipeline.py module directly — they import exclusively from this
file. This is what makes routes trivially testable with
app.dependency_overrides without touching any business logic module.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings, get_settings
from src.comparison.runner import ComparisonRunner
from src.dataset.store import DatasetStore
from src.evaluation.engine import EvaluationEngine
from src.rag.pipeline import RAGPipeline
from src.storage.database import Database, get_database
from src.storage.repository import RunRepository
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.embeddings import EmbeddingManager

# ===========================================================================
# APPLICATION STATE CONTRACT
# ===========================================================================

@dataclass(slots=True)
class AppState:
    """
    Typed contract for everything app.py's lifespan must attach to
    app.state before the application starts serving requests.

    This dataclass is never instantiated directly by route handlers —
    it exists so app.py's lifespan function has a single, type-checked
    place documenting exactly what application-scoped resources must
    be constructed, and in what order (embedding_manager and
    vector_store must exist before rag_pipeline can be built; database
    must exist before any route touches the DB).

    app.py constructs one of these per app instance:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            settings = get_settings()
            database = await startup_db()
            embedding_manager = EmbeddingManager(settings.embedding)
            vector_store = ChromaVectorStore(settings.chroma, embedding_manager)
            rag_pipeline = RAGPipeline(embedding_manager, vector_store, settings)
            evaluation_engine = EvaluationEngine.from_settings(settings)
            comparison_runner = ComparisonRunner(
                rag_pipeline, evaluation_engine, settings
            )
            app.state.resources = AppState(
                database=database,
                embedding_manager=embedding_manager,
                vector_store=vector_store,
                rag_pipeline=rag_pipeline,
                evaluation_engine=evaluation_engine,
                comparison_runner=comparison_runner,
            )
            yield
            await shutdown_db()
    """

    database: Database
    embedding_manager: EmbeddingManager
    vector_store: BaseVectorStore
    rag_pipeline: RAGPipeline
    evaluation_engine: EvaluationEngine
    comparison_runner: ComparisonRunner

def _get_app_state(request: Request) -> AppState:
    """
    Retrieve AppState from the running application, with a clear error
    if the lifespan never ran (e.g. app constructed without lifespan
    in a misconfigured test, or a route called before startup
    completes).

    Raising a 503 here rather than letting an AttributeError propagate
    means a misconfigured app produces an actionable HTTP error instead
    of a raw stack trace on the first request that touches any
    application-scoped resource.
    """
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Application resources not initialised. "
                "The server's lifespan startup may have failed or not "
                "completed. Check server logs for startup errors."
            ),
        )
    return resources

# ===========================================================================
# SETTINGS
# ===========================================================================

def get_app_settings() -> Settings:
    """
    Re-export of config.settings.get_settings() as a dependency.

    Routes inject settings via `Depends(get_app_settings)`, never by
    importing config.settings directly — this keeps the "routes only
    import from src.api.dependencies" rule total, with no exceptions
    for settings access.

    get_settings() is itself lru_cache'd, so this adds zero overhead —
    FastAPI calls this function per request, which returns the same
    cached Settings instance every time.
    """
    return get_settings()

# ===========================================================================
# DATABASE SESSION (request-scoped)
# ===========================================================================

async def get_db(
    settings: Settings = Depends(get_app_settings),
) -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a request-scoped AsyncSession with automatic commit/rollback.

    A fresh session is created for every request and closed when the
    request completes, regardless of success or failure. This is the
    standard FastAPI session-per-request pattern: it prevents one
    request's uncommitted changes or a stale transaction from leaking
    into the next request handled by the same worker.

    Delegates to Database.session_dependency() (src/storage/database.py)
    rather than duplicating commit/rollback/close logic here — this
    function exists solely to adapt that generator to FastAPI's
    Depends() calling convention with the get_database() singleton
    already resolved.

    Usage in a route:
        @router.get("/runs/{run_id}")
        async def get_run(
            run_id: str,
            db: AsyncSession = Depends(get_db),
        ):
            ...
    """
    database = get_database(settings)
    async for session in database.session_dependency():
        yield session

# ===========================================================================
# REPOSITORY (request-scoped, wraps the per-request session)
# ===========================================================================

def get_run_repository(
    db: AsyncSession = Depends(get_db),
) -> RunRepository:
    """
    Construct a RunRepository bound to the current request's session.

    Cheap — RunRepository is a thin wrapper holding only a session
    reference, with no state of its own. Constructing a fresh one per
    request (rather than caching) is correct and free, since the
    session itself is already request-scoped and the repository's
    entire purpose is to operate on exactly that session.

    Usage in a route:
        @router.get("/runs")
        async def list_runs(
            repo: RunRepository = Depends(get_run_repository),
        ):
            return await repo.list_runs()
    """
    return RunRepository(db)

# ===========================================================================
# DATASET STORE (request-scoped, but cheap — no caching needed)
# ===========================================================================

def get_dataset_store(
    settings: Settings = Depends(get_app_settings),
) -> DatasetStore:
    """
    Construct a DatasetStore for the current request.

    DatasetStore is file-based and holds only an in-memory index cache
    that is invalidated on every write — constructing a fresh instance
    per request means each request's first read re-parses index.json
    from disk if a prior request elsewhere already mutated it,
    guaranteeing no request ever sees a stale in-process cache from a
    previous request's instance. The cost of this (one small JSON
    parse) is negligible compared to the correctness guarantee.

    Usage in a route:
        @router.get("/datasets")
        async def list_datasets(
            store: DatasetStore = Depends(get_dataset_store),
        ):
            return store.list_datasets()
    """
    return DatasetStore.from_settings(settings)

# ===========================================================================
# APPLICATION-SCOPED RESOURCES (constructed once in app.py's lifespan,
# retrieved per-request from app.state — never reconstructed here)
# ===========================================================================

def get_embedding_manager(request: Request) -> EmbeddingManager:
    """
    Retrieve the application-wide EmbeddingManager.

    Constructed exactly once in app.py's lifespan (loading the
    HuggingFace model takes 2-5 seconds) and reused for every request
    for the lifetime of the process. Never reconstructed per request —
    doing so would reload the embedding model on every single API call.
    """
    return _get_app_state(request).embedding_manager

def get_vector_store(request: Request) -> BaseVectorStore:
    """
    Retrieve the application-wide vector store (ChromaDB connection).

    Typed as BaseVectorStore, not ChromaVectorStore, so routes and any
    future test override depend on the abstraction — swapping the
    concrete implementation in app.py's lifespan requires no change
    to any route signature.
    """
    return _get_app_state(request).vector_store

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """
    Retrieve the application-wide RAGPipeline.

    Holds a cache of Gemini GenerativeModel instances keyed by
    (model_id, temperature, max_output_tokens) — application scope
    means this cache is shared and warm across all requests, rather
    than rebuilt (and re-paying the first-call SDK initialisation
    cost) on every single question answered through the API.
    """
    return _get_app_state(request).rag_pipeline

def get_evaluation_engine(request: Request) -> EvaluationEngine:
    """
    Retrieve the application-wide EvaluationEngine.

    Holds all 4 instantiated metric evaluators (each with its own
    Gemini judge model client). Constructing this per request would
    re-run build_all_evaluators() — including a model availability
    probe — on every evaluation API call.
    """
    return _get_app_state(request).evaluation_engine

def get_comparison_runner(request: Request) -> ComparisonRunner:
    """
    Retrieve the application-wide ComparisonRunner.

    Wraps the same RAGPipeline and EvaluationEngine instances above —
    constructing a fresh ComparisonRunner per request is harmless
    (it's a thin orchestrator with no expensive internal state) but
    is still served from app.state for consistency with the rest of
    this section and to guarantee it always wraps the exact same
    pipeline/engine pair the rest of the app is using.
    """
    return _get_app_state(request).comparison_runner

__all__ = [
    "AppState",
    "get_app_settings",
    "get_db",
    "get_run_repository",
    "get_dataset_store",
    "get_embedding_manager",
    "get_vector_store",
    "get_rag_pipeline",
    "get_evaluation_engine",
    "get_comparison_runner",
]