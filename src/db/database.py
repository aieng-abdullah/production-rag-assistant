"""SQLAlchemy engine/session wiring (PLAN.md PR-2a).

SQLite for local dev and tests, Postgres in production via DATABASE_URL.
`src/` stays framework-free: no FastAPI imports here.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import Config


class Base(DeclarativeBase):
    """Declarative base — all ORM models hang off this."""


def make_engine(url: str | None = None) -> Engine:
    """Create an engine for `url` (defaults to Config.DATABASE_URL).

    SQLite needs `check_same_thread=False`: FastAPI serves requests on
    worker threads while the engine pool is shared.
    """
    resolved = url or Config.DATABASE_URL
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    return create_engine(resolved, connect_args=connect_args)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Process-wide engine singleton (reset with `reset_engine()` in tests)."""
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def reset_engine() -> None:
    """Drop the engine/session singletons (tests, DATABASE_URL overrides)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error, always close."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
