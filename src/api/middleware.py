"""
src/api/middleware.py

Request ID middleware — assigns a unique ID to every incoming request
and propagates it through all log lines emitted during that request's
lifetime via loguru's contextvars-based bind().

Why contextvars, not threading.local():
    FastAPI is async — a single thread can interleave multiple requests.
    threading.local() would assign the same request_id to concurrent
    requests handled by the same thread. contextvars is coroutine-safe:
    each async task has its own context, so request_id is isolated per
    request even when multiple requests are in flight on one thread.

The request_id appears in:
    - Every loguru log line emitted during the request (via bind)
    - The X-Request-ID response header (for client-side correlation)
    - FastAPI's access log line (via the extra field)
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Per-coroutine context variable — safe for async FastAPI handlers
_request_id_var: ContextVar[str] = ContextVar(
    "request_id", default="no-request"
)


def get_current_request_id() -> str:
    """Return the request ID for the current coroutine context."""
    return _request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique UUID4 request ID to every incoming request.

    Priority order for the request ID:
        1. X-Request-ID header provided by the caller (useful when
           Streamlit's httpx client sends its own trace ID)
        2. Generated UUID4 (default for all other callers)

    The ID is:
        - Stored in a ContextVar (coroutine-safe, not thread-safe)
        - Bound into loguru's context so every log.info/warning/error
          during this request automatically includes request_id
        - Added to the response as X-Request-ID header for client-side
          correlation (the Streamlit UI can log this for debugging)
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        token = _request_id_var.set(request_id)

        # Bind request_id into loguru context for this coroutine.
        # All logger.* calls within call_next() will include request_id
        # in their record["extra"] dict, which _json_sink serialises
        # as a top-level JSON field.
        with logger.contextualize(request_id=request_id):
            logger.debug(
                f"→ {request.method} {request.url.path}"
            )
            response = await call_next(request)
            logger.debug(
                f"← {request.method} {request.url.path} "
                f"status={response.status_code}"
            )

        response.headers["X-Request-ID"] = request_id
        _request_id_var.reset(token)
        return response


__all__ = ["RequestIDMiddleware", "get_current_request_id"]