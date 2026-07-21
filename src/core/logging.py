"""
src/core/logging.py

Centralised logging configuration for both development and production.

Development (APP_ENV=development):
    Human-readable loguru output with colours — what you see locally.

Production (APP_ENV=production):
    JSON-formatted output, one object per line, compatible with every
    major log aggregator (Papertrail, Logtail, Datadog, CloudWatch).
    Each line contains: timestamp, level, message, module, function,
    line, and any extra fields bound via logger.bind().

Why loguru over stdlib logging:
    loguru's sink API is cleaner than logging.Handler subclassing,
    and its bind() context manager is what enables per-request
    request_id propagation (Item 2 below) without thread-locals.

Usage:
    from src.core.logging import setup_logging
    setup_logging()   # called once at app startup in app.py's lifespan
"""

from __future__ import annotations

import json
import sys
from typing import Any

from loguru import logger


def _json_sink(message: Any) -> None:
    """
    Loguru sink that writes one JSON object per log line to stdout.

    Fields included in every line:
        timestamp   ISO 8601 UTC timestamp
        level       DEBUG / INFO / WARNING / ERROR / CRITICAL
        message     the log message string
        module      Python module that emitted the log
        function    function name
        line        line number
        request_id  present when set via logger.bind(request_id=...)

    Extra fields bound via logger.bind() appear at the top level,
    making them directly filterable in log aggregators without
    JSON path expressions.
    """
    record = message.record
    log_entry: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    # Merge any extra fields from logger.bind()
    log_entry.update(record["extra"])
    print(json.dumps(log_entry, ensure_ascii=False), flush=True)


def setup_logging(app_env: str = "development") -> None:
    """
    Configure loguru for the current environment.

    Args:
        app_env: "development" or "production". Reads from
                 Settings.app_env — call after get_settings().

    Called once at the top of app.py's lifespan context manager,
    before any other startup work so every log line (including
    resource construction) uses the correct format.
    """
    # Remove loguru's default stderr sink
    logger.remove()

    if app_env == "production":
        logger.add(
            _json_sink,
            level="INFO",
            format="{message}",   # formatting is handled by _json_sink
            colorize=False,
            backtrace=False,      # don't leak stack traces to log aggregators
            diagnose=False,
        )
        logger.info(
            "Logging initialised in production mode (JSON output)."
        )
    else:
        # Development: coloured, human-readable, with full tracebacks
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        logger.info(
            "Logging initialised in development mode (coloured output)."
        )


__all__ = ["setup_logging"]