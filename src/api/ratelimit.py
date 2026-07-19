"""
src/api/ratelimit.py

Per-IP rate limiting on expensive endpoints via slowapi.

Limits are intentionally asymmetric — read operations (list, get) are
generous; write/compute operations (ingest, generate, evaluate, compare)
are strict because they trigger LLM API calls that burn free-tier quota.

Why per-IP and not per-API-key:
  In the target deployment (Render free tier + Streamlit Cloud), there
  is exactly one API key shared by one Streamlit app — per-key limits
  would apply to the entire app as one user, not per individual browser
  session. Per-IP limits (X-Forwarded-For from Render's proxy) give
  better granularity when multiple people use the public Streamlit URL
  simultaneously.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],
)

# Endpoint-specific limits — applied via @limiter.limit decorator
# on individual route functions when needed.
INGEST_LIMIT = "5/minute"
GENERATE_LIMIT = "3/minute"
EVALUATE_LIMIT = "2/minute"
COMPARE_LIMIT = "1/minute"

__all__ = [
    "limiter",
    "INGEST_LIMIT",
    "GENERATE_LIMIT",
    "EVALUATE_LIMIT",
    "COMPARE_LIMIT",
]