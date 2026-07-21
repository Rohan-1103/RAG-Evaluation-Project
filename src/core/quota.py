"""
src/core/quota.py

Per-provider daily API call counter with 80% threshold warnings.

Tracks successful LLM generate() calls by provider and day.
Resets automatically at midnight UTC — no persistence needed since
free-tier quotas also reset daily.

Why not track token counts:
    Token counts would be more precise (Gemini's limit is 1M tokens/day,
    not just requests) but require per-model token pricing to be useful,
    and token counts aren't always available (some providers return 0).
    Request counts are always available and give a directionally-correct
    signal — "you've made 800 of your 1000 daily Groq requests" is
    actionable even if imprecise.

Integration:
    Called from get_llm_client()'s generate() result path — after a
    successful LLMGenerationResult is returned, the call site records
    the call. Failed calls (LLMClientError) are NOT counted — they don't
    consume quota.

Usage:
    from src.core.quota import record_llm_call, get_quota_status
    record_llm_call("groq")
    status = get_quota_status()
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from loguru import logger

# Known daily limits per provider (free tier, as of July 2026)
# Update these when providers change their limits.
_DAILY_LIMITS: dict[str, int] = {
    "google":      1000,    # Gemini free tier: 1000 RPD
    "groq":        1000,    # Groq free tier: 1000 RPD
    "openrouter":  50,      # OpenRouter free tier without credits: 50 RPD
}

_WARNING_THRESHOLD = 0.80   # Warn at 80% of daily limit

# Thread-safe counters: {provider: {date_str: count}}
_counters: dict[str, dict[str, int]] = defaultdict(
    lambda: defaultdict(int)
)
_lock = Lock()


def _today_utc() -> str:
    """Return today's UTC date as YYYY-MM-DD string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_llm_call(provider: str) -> None:
    """
    Record one successful LLM API call for a provider.

    Thread-safe. Logs a WARNING when the call count crosses 80% of the
    known daily limit, and an ERROR when it reaches 100% (quota likely
    exhausted — subsequent calls may fail with 429).

    Called after a successful generate() — failed calls don't consume
    quota and should not be recorded here.

    Args:
        provider: Provider name: 'google', 'groq', 'openrouter'.
    """
    today = _today_utc()
    with _lock:
        _counters[provider][today] += 1
        count = _counters[provider][today]

    limit = _DAILY_LIMITS.get(provider)
    if limit is None:
        return

    pct = count / limit
    if pct >= 1.0:
        logger.error(
            f"QUOTA [{provider}]: {count}/{limit} daily requests used "
            f"({pct:.0%}). Free-tier quota likely exhausted — "
            f"subsequent calls may fail with 429. "
            f"Quota resets at midnight UTC."
        )
    elif pct >= _WARNING_THRESHOLD:
        logger.warning(
            f"QUOTA [{provider}]: {count}/{limit} daily requests used "
            f"({pct:.0%}). Approaching free-tier daily limit. "
            f"Quota resets at midnight UTC."
        )


def get_quota_status() -> dict[str, Any]:
    """
    Return current quota usage for all providers.

    Used by the /health endpoint to expose quota status:
        {
            "google":     {"used": 45, "limit": 1000, "pct": 0.045},
            "groq":       {"used": 312, "limit": 1000, "pct": 0.312},
            "openrouter": {"used": 38, "limit": 50, "pct": 0.76},
        }

    Returns only providers that have had at least one call today.
    """
    today = _today_utc()
    status: dict[str, Any] = {}

    with _lock:
        for provider, daily in _counters.items():
            count = daily.get(today, 0)
            if count > 0:
                limit = _DAILY_LIMITS.get(provider)
                status[provider] = {
                    "used":  count,
                    "limit": limit,
                    "pct":   round(count / limit, 4) if limit else None,
                    "date":  today,
                }

    return status


def reset_quota_counters() -> None:
    """Reset all counters. Test-only — never call from application code."""
    with _lock:
        _counters.clear()


__all__ = ["record_llm_call", "get_quota_status", "reset_quota_counters"]