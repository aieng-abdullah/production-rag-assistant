"""Quota enforcement (Phase 2: persisted quotas).

Free tier: 1x limits
Pro tier: 10x limits
"""

from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy.orm import sessionmaker, Session

from src.config import Config
from src.services.usage import get_daily_usage, get_document_count, record_usage
from src.db.database import get_session_factory as get_default_session_factory

__all__ = [
    "QuotaResult",
    "check_query_quota",
    "check_document_quota",
    "record_query_usage",
    "record_ingest_usage",
    "record_verify_usage",
    "TIER_MULTIPLIERS",
]


TIER_MULTIPLIERS = {
    "free": 1,
    "pro": 10,
}


# Test-overridable session factory
_session_factory_override: Optional[Callable[[], sessionmaker[Session]]] = None


def _get_session_factory() -> sessionmaker[Session]:
    if _session_factory_override:
        return _session_factory_override()
    return get_default_session_factory()


def set_session_factory_override(factory: Callable[[], sessionmaker[Session]]) -> None:
    """Override session factory for testing."""
    global _session_factory_override
    _session_factory_override = factory


def clear_session_factory_override() -> None:
    """Clear test session factory override."""
    global _session_factory_override
    _session_factory_override = None


@dataclass
class QuotaResult:
    allowed: bool
    limit: int
    used: int
    remaining: int
    tier: str


def _get_multiplier(tier: str) -> int:
    return TIER_MULTIPLIERS.get(tier, 1)


def check_query_quota(user_id: int, tier: str) -> QuotaResult:
    """
    Check if user can make a query.

    Args:
        user_id: User ID
        tier: "free" or "pro"

    Returns:
        QuotaResult with allowed status and details
    """
    session_factory = _get_session_factory()
    with session_factory() as session:
        used = get_daily_usage(session, user_id, kind="query")
        limit = Config.DAILY_QUERY_LIMIT * _get_multiplier(tier)
        remaining = max(limit - used, 0)
        return QuotaResult(
            allowed=used < limit,
            limit=limit,
            used=used,
            remaining=remaining,
            tier=tier,
        )


def check_document_quota(user_id: int, tier: str) -> QuotaResult:
    """
    Check if user can upload a document.

    Args:
        user_id: User ID
        tier: "free" or "pro"

    Returns:
        QuotaResult with allowed status and details
    """
    session_factory = _get_session_factory()
    with session_factory() as session:
        used = get_document_count(session, user_id)
        limit = Config.DOCUMENT_LIMIT * _get_multiplier(tier)
        remaining = max(limit - used, 0)
        return QuotaResult(
            allowed=used < limit,
            limit=limit,
            used=used,
            remaining=remaining,
            tier=tier,
        )


def record_query_usage(user_id: int) -> None:
    """Record a query usage event (2 units)."""
    session_factory = _get_session_factory()
    with session_factory() as session:
        record_usage(session, user_id, kind="query", units=2)
        session.commit()


def record_ingest_usage(user_id: int) -> None:
    """Record an ingest usage event (1 unit)."""
    session_factory = _get_session_factory()
    with session_factory() as session:
        record_usage(session, user_id, kind="ingest", units=1)
        session.commit()


def record_verify_usage(user_id: int) -> None:
    """Record a verify usage event (1 unit)."""
    session_factory = _get_session_factory()
    with session_factory() as session:
        record_usage(session, user_id, kind="verify", units=1)
        session.commit()