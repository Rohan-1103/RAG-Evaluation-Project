"""
src/storage/database.py

Async SQLAlchemy engine, session factory, and lifecycle management.

Responsibilities:
  - Construct the async engine from Settings.storage.database_url.
  - Apply SQLite-specific configuration (foreign key enforcement,
    WAL mode for concurrent reads during writes).
  - Expose a FastAPI-compatible async dependency (get_db_session).
  - Expose a plain async context manager for scripts/Streamlit/tests
    that are not running inside FastAPI's DI system.
  - init_db() creates all tables from Base.metadata — idempotent,
    safe to call on every app startup.
  - close_db() disposes the engine cleanly on shutdown.

Why this file imports Base from models.py, not the reverse:
  models.py owns the schema (table definitions). This file owns the
  connection (engine, sessions). Schema must exist independently of
  any particular connection — you can define ORM models without ever
  connecting to a database, but you cannot create a session without
  schema to bind it to. The dependency points one way: database.py
  depends on models.py, never the inverse. This is what makes
  swapping SQLite for PostgreSQL a one-line DATABASE_URL change rather
  than a model rewrite.

SQLite-specific concerns handled here:
  1. Foreign keys are OFF by default in SQLite — must be enabled per
     connection via PRAGMA foreign_keys=ON, or ondelete="CASCADE" in
     models.py silently does nothing.
  2. WAL (Write-Ahead Logging) mode allows concurrent readers while
     a write is in progress — without it, the Streamlit dashboard
     reading run history would block while an evaluation run is
     writing EvalResultRecords, and vice versa.
  3. SQLite has no real connection pool (file-based, single-writer)
     so pool_size tuning that matters for PostgreSQL is a no-op here
     — NullPool is used for SQLite, QueuePool for PostgreSQL.

Migration to PostgreSQL later requires only:
  DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
  No code in this file changes — the dialect-specific branches below
  already handle both cases.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from config.settings import Settings, get_settings
from src.storage.models import Base

# ===========================================================================
# ENGINE CONSTRUCTION
# ===========================================================================

def _is_sqlite(database_url: str) -> bool:
    """Return True if the database URL targets SQLite."""
    return database_url.startswith("sqlite")

def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """
    Ensure the parent directory of a SQLite file exists before connecting.

    SQLite fails with a cryptic 'unable to open database file' error if
    the parent directory doesn't exist — this is far more common on a
    fresh clone than on a developer's already-set-up machine, since
    Settings.create_required_directories() only creates the directories
    it explicitly knows about (data/raw_docs, data/datasets, etc.), not
    arbitrary nested paths a user might put in DATABASE_URL.
    """
    if not _is_sqlite(database_url):
        return

    # sqlite+aiosqlite:///./data/rag_eval.db -> ./data/rag_eval.db
    path_part = database_url.split("///", maxsplit=1)[-1]
    if path_part in (":memory:", ""):
        return

    db_path = Path(path_part)
    db_path.parent.mkdir(parents=True, exist_ok=True)

def _create_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """
    Construct the async engine with dialect-appropriate pooling.

    SQLite:     NullPool — no connection reuse needed; aiosqlite opens
                a fresh connection per session cheaply, and pooling a
                single-file, single-writer database provides no benefit
                while adding complexity around stale connections.
    PostgreSQL: QueuePool with sane defaults — reused across requests
                in a long-running FastAPI process.

    echo=True logs every SQL statement — useful for debugging, far too
    verbose for normal operation. Controlled by Settings.logging.level
    in init_db(), never hardcoded True here.
    """
    _ensure_sqlite_parent_dir(database_url)

    if _is_sqlite(database_url):
        engine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=NullPool,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    if _is_sqlite(database_url):
        _register_sqlite_pragmas(engine)

    return engine


def _register_sqlite_pragmas(engine: AsyncEngine) -> None:
    """
    Register a connection-level event listener to apply SQLite PRAGMAs.

    Must run on every new DBAPI connection (not once per engine) because
    PRAGMAs are connection-scoped in SQLite, not database-scoped. With
    NullPool, every session opens a genuinely new connection, so this
    fires on every checkout.

    PRAGMA foreign_keys=ON:
      Without this, ondelete="CASCADE" on every ForeignKey in models.py
      is silently ignored. Deleting a DatasetRecord would leave orphaned
      RunRecords pointing at a non-existent dataset_id — a correctness
      bug, not just a performance one.

    PRAGMA journal_mode=WAL:
      Allows the Streamlit dashboard to read run history concurrently
      while EvaluationEngine writes new EvalResultRecords during an
      active run. Without WAL, SQLite's default rollback-journal mode
      takes an exclusive lock during writes, and the dashboard would
      see "database is locked" errors during any active evaluation.

    PRAGMA synchronous=NORMAL:
      Safe with WAL mode (per SQLite docs) and meaningfully faster than
      the default FULL — appropriate for a local benchmarking tool
      where surviving an OS crash mid-write is not a requirement that
      justifies the latency cost on every commit.
    """
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

# ===========================================================================
# DATABASE MANAGER — owns engine + session factory lifecycle
# ===========================================================================

class Database:
    """
    Owns the async engine and session factory for the application lifetime.

    A single instance is created at process startup (via get_database())
    and reused for every request/script invocation. Constructing a new
    engine per request would exhaust SQLite file handles and defeat
    connection pooling entirely for PostgreSQL.

    Usage (FastAPI):
        db = get_database()
        async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
            async for session in db.session_dependency():
                yield session

    Usage (scripts / Streamlit / tests):
        db = get_database()
        async with db.session() as session:
            result = await session.execute(select(RunRecord))
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._echo = settings.logging.level == "DEBUG"
        self._engine: AsyncEngine = _create_engine(
            settings.storage.database_url,
            echo=self._echo,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = (
            async_sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autoflush=False,
                class_=AsyncSession,
            )
        )
        self._initialised = False

        logger.info(
            f"Database initialised. "
            f"url='{self._mask_credentials(settings.storage.database_url)}', "
            f"dialect={'sqlite' if _is_sqlite(settings.storage.database_url) else 'postgresql'}, "
            f"echo={self._echo}"
        )

    # ------------------------------------------------------------------
    # Schema lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """
        Create all tables defined in models.py if they don't exist.

        Idempotent — uses create_all() which only creates missing
        tables and never alters or drops existing ones. Safe to call
        on every application startup.

        For schema CHANGES after initial creation (adding a column,
        changing a type), use Alembic migrations — create_all() never
        modifies existing tables, by design, so schema evolution after
        the first run requires `alembic upgrade head`, not this method.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialised = True
        logger.info(
            f"Database.init_db: Schema ready. "
            f"tables={sorted(Base.metadata.tables.keys())}"
        )

    async def drop_all(self) -> None:
        """
        Drop all tables. DESTRUCTIVE — used only by test fixtures and
        the reset_database CLI script, never called from application
        startup paths.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("Database.drop_all: All tables dropped.")

    async def health_check(self) -> bool:
        """
        Verify the database connection is alive and schema exists.

        Used by FastAPI's startup event and the Streamlit sidebar
        connection indicator. Returns False rather than raising so
        callers can render a clear "database unavailable" state
        instead of crashing the whole app on a transient connection
        issue.
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Database.health_check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Session access patterns
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Async context manager yielding a session with commit/rollback
        handled automatically.

        Use for scripts, Streamlit callbacks, and tests — anywhere
        outside FastAPI's request-scoped dependency injection.

        On success: commits and closes.
        On exception: rolls back, closes, and re-raises — callers see
        the original exception, not a swallowed one.

        Usage:
            async with db.session() as session:
                session.add(RunRecord(...))
                # commit happens automatically on context exit
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def session_dependency(
        self,
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        FastAPI dependency-injection compatible session generator.

        Distinct from session() because FastAPI's Depends() machinery
        requires a plain async generator function (yield exactly once,
        no context-manager decoration) rather than an
        @asynccontextmanager-wrapped one. Shares the identical
        commit/rollback/close semantics with session() so behaviour is
        consistent regardless of which entrypoint is used.

        Wired into routes via:
            from src.api.dependencies import get_db
            @router.get("/runs")
            async def list_runs(db: AsyncSession = Depends(get_db)):
                ...
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Dispose the engine and release all underlying connections.

        Called on FastAPI shutdown event and at the end of script
        execution. For SQLite with NullPool this mainly closes any
        connections still checked out; for PostgreSQL it drains the
        connection pool cleanly so the process doesn't leave dangling
        TCP connections to the database server.
        """
        await self._engine.dispose()
        logger.info("Database: Engine disposed, all connections closed.")

    @staticmethod
    def _mask_credentials(database_url: str) -> str:
        """
        Redact password from a database URL before logging.

        postgresql+asyncpg://user:secret@host/db -> postgresql+asyncpg://user:***@host/db
        SQLite URLs have no credentials and pass through unchanged.
        """
        if "@" not in database_url or "://" not in database_url:
            return database_url
        scheme, rest = database_url.split("://", maxsplit=1)
        if "@" not in rest:
            return database_url
        creds, host_part = rest.split("@", maxsplit=1)
        if ":" in creds:
            user, _ = creds.split(":", maxsplit=1)
            return f"{scheme}://{user}:***@{host_part}"
        return database_url

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def is_sqlite(self) -> bool:
        return _is_sqlite(self._settings.storage.database_url)

    def __repr__(self) -> str:
        return (
            f"Database("
            f"dialect={'sqlite' if self.is_sqlite else 'postgresql'}, "
            f"initialised={self._initialised})"
        )

# ===========================================================================
# SINGLETON ACCESSOR
# ===========================================================================

_database_instance: Database | None = None
_database_lock = asyncio.Lock()


def get_database(settings: Settings | None = None) -> Database:
    """
    Return the process-level Database singleton.

    Unlike get_settings()/get_model_registry() (which use lru_cache
    since they are pure, synchronous, side-effect-free constructions),
    Database holds a live engine and connection state — constructing
    it twice would create two separate engines pointing at the same
    file/server, defeating pooling and risking SQLite file-lock
    contention between them. A plain module-level singleton with
    explicit reset_database() for tests is the correct pattern here,
    not lru_cache.

    Usage:
        from src.storage.database import get_database
        db = get_database()
        await db.init_db()
    """
    global _database_instance
    if _database_instance is None:
        _database_instance = Database(settings or get_settings())
    return _database_instance

async def reset_database_singleton() -> None:
    """
    Dispose and clear the singleton. Used by test fixtures to ensure
    each test gets a fresh engine bound to a fresh (often in-memory or
    temp-file) database, rather than reusing connections from a
    previous test's Database instance.

    Never called from application code paths — only from
    tests/conftest.py fixtures.
    """
    global _database_instance
    if _database_instance is not None:
        await _database_instance.close()
        _database_instance = None

# ===========================================================================
# CONVENIENCE STARTUP/SHUTDOWN HOOKS
# ===========================================================================

async def startup_db() -> Database:
    """
    Convenience function for FastAPI's lifespan/startup event and for
    scripts that just need "give me a ready database" in one call.

    Usage (FastAPI):
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from src.storage.database import startup_db, shutdown_db

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await startup_db()
            yield
            await shutdown_db()

        app = FastAPI(lifespan=lifespan)

    Usage (scripts):
        async def main():
            db = await startup_db()
            async with db.session() as session:
                ...
    """
    db = get_database()
    if not db.is_initialised:
        await db.init_db()
    healthy = await db.health_check()
    if not healthy:
        raise RuntimeError(
            "startup_db: Database health check failed immediately "
            "after initialisation. Check DATABASE_URL and file "
            "permissions."
        )
    return db

async def shutdown_db() -> None:
    """
    Convenience function for FastAPI's lifespan/shutdown event.
    Disposes the singleton engine cleanly.
    """
    global _database_instance
    if _database_instance is not None:
        await _database_instance.close()
        _database_instance = None

__all__ = [
    "Database",
    "get_database",
    "reset_database_singleton",
    "startup_db",
    "shutdown_db",
]