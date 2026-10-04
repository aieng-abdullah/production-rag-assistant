"""Usage quotas (PLAN.md PR-3): 20 queries/day, 5 docs, 100MB storage.

Framework-free: takes an open SQLAlchemy session, raises `QuotaExceeded`.
HTTP mapping (429) lives in `src/api/`.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.config import Config
from src.db.models import Document, UsageEvent

__all__ = [
    "QuotaExceeded",
    "DAILY_QUERY_LIMIT",
    "DOCUMENT_LIMIT",
    "STORAGE_LIMIT_BYTES",
    "queries_today",
    "check_query_quota",
    "check_document_quota",
    "record_usage",
]

DAILY_QUERY_LIMIT = 20
DOCUMENT_LIMIT = 5
STORAGE_LIMIT_BYTES = 100 * 1024 * 1024  # 100MB


class QuotaExceeded(Exception):
    """User-facing quota violation; `retry_after` = seconds to daily reset."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _midnight_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _seconds_to_reset() -> int:
    return int((_midnight_utc() + timedelta(days=1) - datetime.now(timezone.utc)).total_seconds())


def queries_today(session: Session, user_id: int) -> int:
    """Count this user's `query` events since 00:00 UTC."""
    count = (
        session.query(func.count(UsageEvent.id))
        .filter(
            UsageEvent.user_id == user_id,
            UsageEvent.kind == "query",
            UsageEvent.created_at >= _midnight_utc(),
        )
        .scalar()
    )
    return int(count or 0)


def check_query_quota(session: Session, user_id: int) -> None:
    used = queries_today(session, user_id)
    if used >= DAILY_QUERY_LIMIT:
        raise QuotaExceeded(
            f"Daily query quota reached ({DAILY_QUERY_LIMIT} queries/day). "
            "Resets at 00:00 UTC.",
            retry_after=_seconds_to_reset(),
        )


def storage_bytes(user_id: int) -> int:
    """Total bytes on disk for this tenant's uploads (0 if none)."""
    directory = Config.DATA_DIR / "raw" / str(user_id)
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def check_document_quota(session: Session, user_id: int, incoming_bytes: int) -> None:
    """Enforce 5-doc and 100MB limits BEFORE writing the new file."""
    docs = (
        session.query(func.count(Document.id))
        .filter(Document.user_id == user_id)
        .scalar()
    )
    if int(docs or 0) >= DOCUMENT_LIMIT:
        raise QuotaExceeded(
            f"Document quota reached ({DOCUMENT_LIMIT} documents). "
            "Delete a document to upload another."
        )
    if storage_bytes(user_id) + incoming_bytes > STORAGE_LIMIT_BYTES:
        raise QuotaExceeded(
            f"Storage quota reached ({STORAGE_LIMIT_BYTES // (1024 * 1024)}MB). "
            "Delete documents to free space."
        )


def record_usage(session: Session, user_id: int, kind: str, units: int = 1) -> None:
    """Persist one usage event (query | ingest | verify)."""
    session.add(UsageEvent(user_id=user_id, kind=kind, units=units))
    session.flush()
