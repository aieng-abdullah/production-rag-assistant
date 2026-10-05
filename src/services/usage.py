"""Usage event tracking (Phase 2: persisted quotas).

Framework-free: session in, state out. Caller manages transaction.
"""

from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.db.models import UsageEvent

__all__ = ["record_usage", "get_daily_usage"]


def record_usage(
    session: Session,
    user_id: int,
    kind: str,
    units: int = 1,
) -> UsageEvent:
    """
    Record a usage event.

    Args:
        session: Open SQLAlchemy session
        user_id: User ID (tenant)
        kind: "query" | "ingest" | "verify"
        units: Number of units consumed (query=2, ingest=1, verify=1)

    Returns:
        The created UsageEvent
    """
    event = UsageEvent(
        user_id=user_id,
        kind=kind,
        units=units,
        created_at=datetime.now(timezone.utc),
    )
    session.add(event)
    session.flush()
    return event


def get_daily_usage(session: Session, user_id: int, kind: str = "query") -> int:
    """
    Get total units used today for a specific kind.

    Args:
        session: Open SQLAlchemy session
        user_id: User ID
        kind: Event kind to sum (default: "query")

    Returns:
        Total units used today
    """
    from sqlalchemy import func, select

    today = datetime.now(timezone.utc).date()
    stmt = (
        select(func.coalesce(func.sum(UsageEvent.units), 0))
        .where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == kind,
            func.date(UsageEvent.created_at) == today,
        )
    )
    return session.execute(stmt).scalar() or 0


def get_document_count(session: Session, user_id: int) -> int:
    """
    Get count of distinct documents for a user.

    Args:
        session: Open SQLAlchemy session
        user_id: User ID

    Returns:
        Number of documents
    """
    from sqlalchemy import func, select
    from src.db.models import Document

    stmt = select(func.count(Document.id)).where(Document.user_id == user_id)
    return session.execute(stmt).scalar() or 0