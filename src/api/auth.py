"""
src/api/auth.py

API key authentication dependency.

Uses X-API-Key header (industry standard for service-to-service auth)
rather than Authorization: Bearer because:
  1. The Streamlit UI is the only caller — it's not a user-facing OAuth flow
  2. X-API-Key is simpler to configure in Streamlit's st.secrets and
     Render/Streamlit Cloud environment variable dashboards
  3. FastAPI's built-in APIKeyHeader integrates cleanly with /docs

Security properties:
  - Constant-time comparison (secrets.compare_digest) prevents timing
    attacks that could leak key length/prefix via response time differences
  - Auth disabled in development when API_SECRET_KEY is empty — explicit
    opt-out, not implicit. Logs a loud WARNING so it's never forgotten.
  - /health and /docs excluded from auth — healthchecks must work without
    credentials for Docker/Render's own monitoring to function.

Public endpoints (no auth required):
    GET  /health
    GET  /docs
    GET  /openapi.json
    GET  /redoc

All /api/v1/* endpoints require X-API-Key.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger

from config.settings import Settings, get_settings

_API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,   # We raise our own exception for a better message
    description="API key for authentication. Required for all /api/v1/* endpoints.",
)

_auth_warning_logged = False


def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency — validates X-API-Key header on every request.

    Usage (applied globally in app.py, not per-route):
        app = FastAPI(dependencies=[Depends(verify_api_key)])

    Applied globally so every new route added to any router is
    automatically protected without the developer remembering to add
    the dependency manually — the failure mode for manual per-route
    auth is always "forgot to add it to the new endpoint."
    """
    global _auth_warning_logged

    # Development mode: auth disabled when secret key not configured
    if not settings.api.secret_key:
        if not _auth_warning_logged:
            logger.warning(
                "⚠️  API authentication is DISABLED. "
                "API_SECRET_KEY is not set in .env. "
                "Set it before any public deployment — without it, "
                "anyone who can reach this server can trigger LLM API "
                "calls against your Gemini/Groq/OpenRouter keys."
            )
            _auth_warning_logged = True
        return

    if not settings.api.auth_enabled:
        if not _auth_warning_logged:
            logger.warning(
                "⚠️  API authentication is DISABLED via auth_enabled=false. "
                "Only acceptable for local development."
            )
            _auth_warning_logged = True
        return

    # No key provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing API key. Include X-API-Key header with your request. "
                "Example: curl -H 'X-API-Key: your-key' http://localhost:8000/api/v1/..."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison — prevents timing attacks
    if not secrets.compare_digest(api_key, settings.api.secret_key):
        logger.warning(
            f"Auth failure: invalid API key provided "
            f"(first 4 chars: '{api_key[:4]}...'). "
            f"Check that API_SECRET_KEY matches on both client and server."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


__all__ = ["verify_api_key"]