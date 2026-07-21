"""
src/api/app.py

FastAPI application factory.

This is the ONLY module in the project that:
  1. Constructs all application-scoped resources (embedding manager,
     vector store, RAG pipeline, evaluation engine, comparison runner,
     database) and attaches them to app.state via the lifespan context
     manager.
  2. Wires CORS, exception handlers, and route routers together into
     one FastAPI instance.

Every resource constructed here exists for exactly one reason: routes
need to retrieve it via src.api.dependencies functions, which read from
app.state rather than constructing anything themselves. This file is
the single place where "what gets built, in what order, exactly once"
is decided — see AppState's docstring in dependencies.py for the full
rationale on why these specific resources are application-scoped
rather than per-request.

Construction order is NOT arbitrary — it follows the dependency graph:
    Settings
        -> EmbeddingManager        (needs settings.embedding)
        -> BaseVectorStore         (needs embedding_manager for dimension validation)
        -> RAGPipeline             (needs embedding_manager + vector_store)
        -> EvaluationEngine        (independent — only needs settings + eval.yaml)
        -> ComparisonRunner        (needs rag_pipeline + evaluation_engine)
    Database (independent of the above — only needs settings.storage)

Startup failures are NOT swallowed. If EmbeddingManager fails to load
the HuggingFace model, or ChromaDB fails to open its persist directory,
the lifespan raises and uvicorn refuses to start serving requests. A
partially-initialised app silently returning 503s for some endpoints
and 200s for others is a worse failure mode than refusing to start —
see Settings.get_settings()'s identical "fail fast at construction"
philosophy in config/settings.py.

create_app() is a factory function, not a module-level `app = FastAPI()`
instance, specifically so tests can construct multiple independent app
instances (each with its own AppState, its own temp ChromaDB directory,
its own in-memory SQLite database) without any shared global state
between test cases. scripts/seed_demo_data.py and uvicorn's ASGI
entrypoint both call create_app() — there is exactly one code path that
produces a runnable application, used identically in production and in
tests.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config.settings import Settings, get_settings
# ADD import at top of app.py
from src.api.auth import verify_api_key
from src.api.dependencies import AppState
from src.comparison.runner import ComparisonRunner, ComparisonRunnerError
from src.dataset.base import GenerationError
from src.evaluation.engine import EvaluationEngine
from src.ingestion.base import LoaderError, UnsupportedFormatError
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.storage.database import shutdown_db, startup_db
from src.vectorstore.base import (
    CollectionNotFoundError,
    DimensionMismatchError,
    VectorStoreError,
)
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.embeddings import EmbeddingError, EmbeddingManager

# ADD imports
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from src.api.ratelimit import limiter
from prometheus_fastapi_instrumentator import Instrumentator

# ADD import
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
from starlette.middleware.gzip import GZipMiddleware

from src.rag.clients import LLMClientError
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ADD import
from src.core.logging import setup_logging
from src.api.middleware import RequestIDMiddleware


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Reject requests exceeding MAX_UPLOAD_SIZE_MB before they reach
    any route handler. Without this, a 1GB file upload would be fully
    buffered into memory before FastAPI's own validation could reject it.
    """
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB

    async def dispatch(
        self, request: StarletteRequest, call_next: Any
    ) -> Response:
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.MAX_UPLOAD_BYTES:
                return Response(
                    content='{"error":"file_too_large","detail":"'
                    f'Upload exceeds maximum allowed size of '
                    f'{self.MAX_UPLOAD_BYTES // (1024*1024)}MB."}}',
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)


# ===========================================================================
# RESOURCE CONSTRUCTION — the dependency graph from AppState's docstring
# ===========================================================================

async def _build_app_state(settings: Settings) -> AppState:
    """Construct every application-scoped resource in dependency order.

    Each step logs its own success/failure so a startup failure points
    immediately at which resource failed to construct, rather than producing a
    single opaque traceback at the bottom of the chain.

    Raises whatever the underlying constructor raises — EmbeddingError,
    VectorStoreError, RuntimeError from evaluator initialisation, etc. None of
    these are caught here; lifespan() below is responsible for deciding what a
    startup failure means for the running process.
    """
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="rag_eval_worker",
    )
    loop.set_default_executor(executor)
    logger.info("Application startup: ThreadPoolExecutor(max_workers=2) set.")
    
    startup_start = time.monotonic()
    logger.info("Application startup: constructing resources...")

    database = await startup_db()
    logger.info("Application startup: database ready.")

    embedding_manager = EmbeddingManager(settings.embedding)
    logger.info(
        f"Application startup: embedding manager ready "
        f"(model={embedding_manager.model_name}, "
        f"dim={embedding_manager.dimension})."
    )

    vector_store = ChromaVectorStore(
        config=settings.chroma,
        embedding_manager=embedding_manager,
    )
    logger.info(
        f"Application startup: vector store ready "
        f"({len(vector_store.list_collections())} existing collections)."
    )

    rag_pipeline = RAGPipeline(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        settings=settings,
    )
    logger.info("Application startup: RAG pipeline ready.")
    
    try:
        collections = vector_store.list_collections()
        if collections:
            # Warm up against the first available collection.
            # warm_up() embeds a probe query + queries top_k=1 —
            # zero API calls, zero quota consumed.
            warmed = rag_pipeline.warm_up(
                collection_name=collections[0].name,
                model_id=settings.dataset_gen.model,
            )
            if warmed:
                logger.info(
                    f"Application startup: RAG pipeline warmed up "
                    f"against collection '{collections[0].name}'."
                )
            else:
                logger.warning(
                    "Application startup: RAG pipeline warm-up failed — "
                    "check embedding model and ChromaDB configuration."
                )
        else:
            logger.info(
                "Application startup: No collections yet — "
                "warm-up skipped (ingest documents to populate ChromaDB)."
            )
    except Exception as exc:
        # Warm-up failure is non-fatal — the server starts anyway.
        # The first real query will pay the cold-start cost instead.
        logger.warning(f"Application startup: warm-up skipped: {exc}")

    evaluation_engine = EvaluationEngine.from_settings(settings)
    logger.info(
        f"Application startup: evaluation engine ready "
        f"(metrics={evaluation_engine.metric_names})."
    )

    comparison_runner = ComparisonRunner(
        rag_pipeline=rag_pipeline,
        evaluation_engine=evaluation_engine,
        settings=settings,
    )
    logger.info("Application startup: comparison runner ready.")

    # Warn if CORS is too permissive
    if "*" in settings.api.cors_origins:
        logger.error(
            "SECURITY: CORS is set to '*' — this allows any origin to "
            "call your API. Set API_CORS_ORIGINS to your exact "
            "Streamlit URL before public deployment."
        )

    elapsed_ms = (time.monotonic() - startup_start) * 1000
    logger.info(
        f"Application startup: all resources ready in {elapsed_ms:.0f}ms."
    )

    return AppState(
        database=database,
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        rag_pipeline=rag_pipeline,
        evaluation_engine=evaluation_engine,
        comparison_runner=comparison_runner,
    )


# ===========================================================================
# LIFESPAN — startup and shutdown
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs once when uvicorn starts the worker,
    before any request is accepted. Everything after `yield` runs once
    when the worker shuts down (SIGTERM, or test client teardown).

    If resource construction raises, this function does NOT catch the
    exception — it propagates out of the context manager, which causes
    uvicorn to abort startup entirely and exit non-zero rather than
    serve traffic against a half-built application. This is the
    intended behaviour: a missing GEMINI_API_KEY or an unreadable
    ChromaDB directory should stop the process immediately, with the
    real exception and traceback visible in the startup logs, not
    surface later as a confusing 503 on the first request that happens
    to touch the broken resource.
    """
    settings = get_settings()
    setup_logging(app_env=settings.app_env)
    app.state.resources = await _build_app_state(settings)

    yield

    logger.info("Application shutdown: releasing resources...")
    await shutdown_db()
    logger.info("Application shutdown: complete.")

# ===========================================================================
# APP FACTORY
# ===========================================================================

def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Construct and fully wire a FastAPI application instance.

    Args:
        settings: Optional Settings override. Tests pass a Settings
                  instance pointed at a temp directory / in-memory
                  database; production and scripts/seed_demo_data.py
                  omit this and get the process-wide get_settings()
                  singleton via the lifespan function's own call.

                  Note this parameter is currently accepted for API
                  symmetry with other factories in this codebase
                  (RAGPipeline.from_settings, etc.) but the lifespan
                  closure above always re-resolves get_settings() at
                  startup time rather than closing over this parameter
                  directly — see the inline comment at the lifespan
                  call site below for why, and how to override it in
                  tests.

    Returns:
        Configured FastAPI instance, NOT yet running. uvicorn's
        ASGI server drives the lifespan and serves requests; this
        function only assembles the application object.
    """
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="RAG Evaluation Benchmarking Tool",
        description=(
            "Production-grade RAG evaluation with LLM-as-a-Judge "
            "scoring across Faithfulness, Answer Relevance, "
            "Context Precision, and Correctness metrics."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    _configure_cors(app, resolved_settings)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(MaxBodySizeMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    _register_exception_handlers(app)
    _register_routers(app)
    
    Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_group_untemplated=True,
    should_respect_env_var=True,
    env_var_name="ENABLE_METRICS",
    excluded_handlers=["/health", "/metrics"],).instrument(app).expose(app, include_in_schema=False)

    return app

# ===========================================================================
# CORS
# ===========================================================================

def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """
    Configure CORS to allow exactly the Streamlit frontend's origin(s).

    settings.api.cors_origins defaults to localhost:8501 (Streamlit's
    default port) — see APIConfig in config/settings.py for the full
    rationale on why this is parsed from a plain comma-separated
    string rather than a list field, which was a real
    pydantic-settings parsing issue worked through earlier in this
    project's setup.

    allow_credentials=True is required for the Streamlit app to send
    cookies/auth headers in the future; harmless to enable now even
    though no auth exists yet, since allow_origins is an explicit
    finite list, never "*".
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ===========================================================================
# EXCEPTION HANDLERS
# ===========================================================================

def _register_exception_handlers(app: FastAPI) -> None:
    """
    Map domain-specific exceptions from src/ to structured HTTP
    responses, rather than letting them surface as opaque 500s with
    raw Python tracebacks.

    Every exception type handled here is one explicitly defined and
    raised somewhere in src/ — this list is a direct map of "what can
    go wrong in business logic" to "what HTTP status code and message
    that should produce." Anything NOT in this list (a genuine bug)
    correctly falls through to FastAPI's default 500 handler, which is
    the right behaviour for truly unexpected errors: they should be
    loud, logged with full traceback, and visibly distinct from
    expected domain failures.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Pydantic request validation failures -> structured 422."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(CollectionNotFoundError)
    async def _handle_collection_not_found(
        request: Request, exc: CollectionNotFoundError
    ) -> JSONResponse:
        """Vector store collection doesn't exist -> 404, not 500."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "collection_not_found", "detail": str(exc)},
        )

    @app.exception_handler(DimensionMismatchError)
    async def _handle_dimension_mismatch(
        request: Request, exc: DimensionMismatchError
    ) -> JSONResponse:
        """
        Embedding dimension conflict -> 409 Conflict.

        This is always a configuration error (wrong embedding model
        used against an existing collection), not a transient failure
        — 409 communicates "the request is valid but conflicts with
        existing state" more precisely than a generic 400 or 500.
        """
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "dimension_mismatch", "detail": str(exc)},
        )

    @app.exception_handler(VectorStoreError)
    async def _handle_vector_store_error(
        request: Request, exc: VectorStoreError
    ) -> JSONResponse:
        """Generic vector store infra failure -> 502 Bad Gateway."""
        logger.error(f"VectorStoreError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "vector_store_error", "detail": str(exc)},
        )

    @app.exception_handler(EmbeddingError)
    async def _handle_embedding_error(
        request: Request, exc: EmbeddingError
    ) -> JSONResponse:
        """Embedding model/API failure -> 502 Bad Gateway."""
        logger.error(f"EmbeddingError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "embedding_error", "detail": str(exc)},
        )

    @app.exception_handler(LoaderError)
    async def _handle_loader_error(
        request: Request, exc: LoaderError
    ) -> JSONResponse:
        """Document parsing failure -> 422, the uploaded file is the issue."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "loader_error", "detail": str(exc)},
        )

    @app.exception_handler(UnsupportedFormatError)
    async def _handle_unsupported_format(
        request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
        """Unrecognised file extension -> 415 Unsupported Media Type."""
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": "unsupported_format", "detail": str(exc)},
        )

    @app.exception_handler(GenerationError)
    async def _handle_generation_error(
        request: Request, exc: GenerationError
    ) -> JSONResponse:
        """Dataset generation produced zero usable pairs -> 422."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "generation_error", "detail": str(exc)},
        )

    @app.exception_handler(RAGPipelineError)
    async def _handle_rag_pipeline_error(
        request: Request, exc: RAGPipelineError
    ) -> JSONResponse:
        """
        Unrecoverable RAG pipeline failure -> 502.

        Note: most RAG failures (empty context, model refusal,
        transient API errors) are deliberately NOT exceptions — see
        RAGPipeline.answer()'s docstring, which converts those into a
        normal RAGResponse with empty_context/answer_refused flags so
        the evaluation pipeline can score them rather than aborting.
        RAGPipelineError is reserved for true infra failures (embedding
        model unreachable, SDK missing) that prevent any response at all.
        """
        logger.error(f"RAGPipelineError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "rag_pipeline_error", "detail": str(exc)},
        )

    @app.exception_handler(ComparisonRunnerError)
    async def _handle_comparison_runner_error(
        request: Request, exc: ComparisonRunnerError
    ) -> JSONResponse:
        """
        Entire comparison job failed (every model config failed, or
        the requested grid exceeded max_total_runs) -> 422.

        Per-model failures within a partially-successful comparison
        are NOT exceptions — they're captured in ModelRunResult and
        surfaced as part of a normal 200 response with some entries
        missing. This handler only fires for the "nothing at all could
        be compared" case.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "comparison_runner_error", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        True catch-all for anything not explicitly mapped above.

        Logged with full exception context via logger.exception()
        (captures traceback) since, by definition, anything reaching
        this handler was not anticipated by the domain-specific
        handlers above and represents either a genuine bug or a new
        failure mode that should get its own handler once understood.
        The response body deliberately omits exception details beyond
        the type name — leaking internal exception messages/tracebacks
        to API clients is an information disclosure risk, even for a
        local benchmarking tool that may later be exposed beyond
        localhost.
        """
        logger.exception(
            f"Unhandled exception on {request.method} {request.url.path}: "
            f"{exc}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": (
                    f"An unexpected {type(exc).__name__} occurred. "
                    f"Check server logs for details."
                ),
            },
        )
    
    # ADD new handler inside _register_exception_handlers():
    @app.exception_handler(LLMClientError)
    async def _handle_llm_client_error(
        request: Request, exc: LLMClientError
    ) -> JSONResponse:
        """
        LLM provider failures → 502 with provider-specific guidance.

        Distinct from the generic VectorStoreError/EmbeddingError 502s —
        LLMClientError carries provider + model_id which lets us give the
        user actionable information (which provider failed, what to check)
        rather than a generic "upstream service unavailable."
        """
        provider_status_urls = {
            "google":      "https://status.google.com",
            "groq":        "https://groqstatus.com",
            "openrouter":  "https://status.openrouter.ai",
        }
        status_url = provider_status_urls.get(exc.provider, "")

        logger.error(
            f"LLMClientError: provider={exc.provider}, "
            f"model={exc.model_id}, reason={exc.reason}"
        )

        detail = (
            f"LLM provider '{exc.provider}' failed for model '{exc.model_id}': "
            f"{exc.reason}"
        )
        if status_url:
            detail += f" Check provider status at {status_url}"

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "llm_provider_error",
                "provider": exc.provider,
                "model_id": exc.model_id,
                "detail": detail,
            },
        )    

# ===========================================================================
# ROUTER REGISTRATION
# ===========================================================================

def _register_routers(app: FastAPI) -> None:
    from src.api.routes import compare, datasets, evaluate, ingest, models

    # Auth dependency applied to every route in every router.
    # /health is defined directly on app (not via a router) so it
    # stays public — Docker/Render healthchecks don't send API keys.
    _auth = [Depends(verify_api_key)]

    app.include_router(
        ingest.router,
        prefix="/api/v1/ingest",
        tags=["ingestion"],
        dependencies=_auth,
    )
    app.include_router(
        datasets.router,
        prefix="/api/v1/datasets",
        tags=["datasets"],
        dependencies=_auth,
    )
    app.include_router(
        evaluate.router,
        prefix="/api/v1/evaluate",
        tags=["evaluation"],
        dependencies=_auth,
    )
    app.include_router(
        compare.router,
        prefix="/api/v1/compare",
        tags=["comparison"],
        dependencies=_auth,
    )
    app.include_router(
        models.router,
        prefix="/api/v1/models",
        tags=["models"],
        dependencies=_auth,
    )

    # /health stays public — no _auth dependency
    @app.get("/health", tags=["health"])
    async def health_check(request: Request) -> dict[str, Any]:
        from src.core.quota import get_quota_status
        from src.rag.clients import get_circuit_breaker_states
    
        resources = getattr(request.app.state, "resources", None)
        if resources is None:
            return {"status": "starting", "database": "unknown"}
    
        db_healthy = await resources.database.health_check()
        cb_states = get_circuit_breaker_states()
        quota = get_quota_status()
    
        # Determine overall status
        any_circuit_open = any(
            state == "open" for state in cb_states.values()
        )
        any_quota_critical = any(
            v["pct"] is not None and v["pct"] >= 1.0
            for v in quota.values()
        )
    
        if not db_healthy or any_quota_critical:
            overall = "degraded"
        elif any_circuit_open:
            overall = "degraded"
        else:
            overall = "healthy"
    
        return {
            "status":           overall,
            "database":         "connected" if db_healthy else "unreachable",
            "embedding_model":  resources.embedding_manager.model_name,
            "embedding_dim":    resources.embedding_manager.dimension,
            "evaluation_metrics": resources.evaluation_engine.metric_names,
            "circuit_breakers": cb_states,
            "quota":            quota,
        }
    # @app.get("/health", tags=["health"])
    # async def health_check(request: Request) -> dict[str, Any]:
    #     # ... existing health check code unchanged ...
    #     """
    #     Liveness/readiness probe.

    #     Checks the database connection is alive (the one
    #     application-scoped resource most likely to fail independently
    #     of the others, e.g. disk full, file permissions changed after
    #     startup) and reports which resources are attached to
    #     app.state. Does NOT make a live Gemini API call on every
    #     health check — that would burn API quota on infrastructure
    #     monitoring, which is exactly the kind of free-tier-hostile
    #     design this project has spent considerable effort avoiding
    #     elsewhere (see ComparisonRunner's max_total_runs cap,
    #     RAGPipeline.warm_up() skipping generation, etc.).
    #     """
    #     resources = getattr(request.app.state, "resources", None)
    #     if resources is None:
    #         return {
    #             "status": "starting",
    #             "database": "unknown",
    #         }

    #     db_healthy = await resources.database.health_check()

    #     return {
    #         "status": "healthy" if db_healthy else "degraded",
    #         "database": "connected" if db_healthy else "unreachable",
    #         "embedding_model": resources.embedding_manager.model_name,
    #         "embedding_dim": resources.embedding_manager.dimension,
    #         "evaluation_metrics": resources.evaluation_engine.metric_names,
    #     }

__all__ = ["create_app", "lifespan"]