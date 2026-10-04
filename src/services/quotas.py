"""Usage quotas (PLAN.md PR-3/PR-3b): queries/day, docs, storage.

Limits come from `Config` (env-tunable: `DAILY_QUERY_LIMIT`,
`DOCUMENT_LIMIT`, `STORAGE_LIMIT_MB`).

Framework-free: takes an open SQLAlchemy session, raises `QuotaExceeded`.
HTTP mapping (429) lives in `src/api/`.

Concurrency: check+insert happen in ONE statement (`INSERT ... SELECT ...
WHERE count < limit`). Under SQLite's single-writer lock that closes the
check-then-act race. Postgres multi-worker needs SERIALIZABLE/advisory
locks — tracked for PR-7 when the DB goes to Postgres.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, insert, literal, select
from sqlalchemy.orm import Session

from src.config import Config
from src.db.models import Document, UsageEvent

__all__ = [
    "QuotaExceeded",
    "DAILY_QUERY_LIMIT",
    "DOCUMENT_LIMIT",
    "STORAGE_LIMIT_BYTES",
    "queries_today",
    "reserve_query_slot",
    "reserve_verify_slot",
    "refund_usage",
    "check_document_quota",
    "reserve_document_slot",
    "enforce_storage_quota",
    "storage_bytes",
]

DAILY_QUERY_LIMIT = Config.DAILY_QUERY_LIMIT
DOCUMENT_LIMIT = Config.DOCUMENT_LIMIT
STORAGE_LIMIT_BYTES = Config.STORAGE_LIMIT_MB * 1024 * 1024


class QuotaExceeded(Exception):
    """User-facing quota violation; `retry_after` = seconds to daily reset."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _midnight_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _seconds_to_reset() -> int:
    tomorrow = _midnight_utc() + timedelta(days=1)
    return int((tomorrow - datetime.now(timezone.utc)).total_seconds())


def _query_quota_error() -> QuotaExceeded:
    return QuotaExceeded(
        f"Daily query quota reached ({DAILY_QUERY_LIMIT} queries/day). "
        "Resets at 00:00 UTC.",
        retry_after=_seconds_to_reset(),
    )


def queries_today(session: Session, user_id: int) -> int:
    """Count this user's quota units since 00:00 UTC.

    Counts `query` AND `verify` rows — verified answers cost 2 units
    (PLAN PR-4b), so `/usage` reports units consumed, not answers given.
    """
    count = (
        session.query(func.count(UsageEvent.id))
        .filter(
            UsageEvent.user_id == user_id,
            UsageEvent.kind.in_(["query", "verify"]),
            UsageEvent.created_at >= _midnight_utc(),
        )
        .scalar()
    )
    return int(count or 0)


def _reserve_unit(session: Session, user_id: int, kind: str) -> int:
    """Atomically count+insert one usage unit; returns its id.

    Single statement: the WHERE clause re-evaluates the combined
    query+verify count at insert time, so concurrent requests cannot both
    slip past the limit. Raises QuotaExceeded when no row was inserted.
    """
    today = (
        select(func.count(UsageEvent.id))
        .where(
            UsageEvent.user_id == user_id,
            UsageEvent.kind.in_(["query", "verify"]),
            UsageEvent.created_at >= _midnight_utc(),
        )
        .scalar_subquery()
    )
    stmt = (
        insert(UsageEvent)
        .from_select(
            [
                UsageEvent.user_id,
                UsageEvent.kind,
                UsageEvent.units,
                UsageEvent.created_at,
            ],
            select(
                literal(user_id),
                literal(kind),
                literal(1),
                literal(datetime.now(timezone.utc)),
            ).where(today < DAILY_QUERY_LIMIT),
        )
        .returning(UsageEvent.id)
    )
    row = session.execute(stmt).first()
    if row is None:
        raise _query_quota_error()
    return int(row[0])


def reserve_query_slot(session: Session, user_id: int) -> int:
    """Reserve the `query` unit of one chat request."""
    return _reserve_unit(session, user_id, "query")


def reserve_verify_slot(session: Session, user_id: int) -> int:
    """Reserve the `verify` unit (PLAN PR-4b: each verified answer costs
    query+verify = 2 units of the same daily pool)."""
    return _reserve_unit(session, user_id, "verify")


def refund_usage(session: Session, event_id: int) -> None:
    """Compensate a reserved slot when the work it paid for failed."""
    event = session.get(UsageEvent, event_id)
    if event is not None:
        session.delete(event)


def storage_bytes(user_id: int) -> int:
    """Total bytes on disk for this tenant's uploads (0 if none)."""
    directory = Config.DATA_DIR / "raw" / str(user_id)
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def enforce_storage_quota(user_id: int) -> None:
    """Post-write storage check — the shared filesystem closes the TOCTOU
    window that a pre-write count cannot."""
    if storage_bytes(user_id) > STORAGE_LIMIT_BYTES:
        raise QuotaExceeded(
            f"Storage quota reached ({STORAGE_LIMIT_BYTES // (1024 * 1024)}MB). "
            "Delete documents to free space."
        )


def check_document_quota(session: Session, user_id: int, incoming_bytes: int) -> None:
    """Pre-write fast fail (avoids stashing 100MB just to reject it).

    Best effort — enforcement is `reserve_document_slot` + `enforce_storage_quota`.
    """
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


def reserve_document_slot(session: Session, user_id: int, filename: str) -> int:
    """Atomically count+insert the Document row; returns its id.

    Same single-statement pattern as `reserve_query_slot` — the 5-doc
    limit holds under concurrent uploads.
    """
    doc_count = (
        select(func.count(Document.id))
        .where(Document.user_id == user_id)
        .scalar_subquery()
    )
    stmt = (
        insert(Document)
        .from_select(
            [
                Document.user_id,
                Document.filename,
                Document.status,
                Document.created_at,
            ],
            select(
                literal(user_id),
                literal(filename),
                literal("processing"),
                literal(datetime.now(timezone.utc)),
            ).where(doc_count < DOCUMENT_LIMIT),
        )
        .returning(Document.id)
    )
    row = session.execute(stmt).first()
    if row is None:
        raise QuotaExceeded(
            f"Document quota reached ({DOCUMENT_LIMIT} documents). "
            "Delete a document to upload another."
        )
    return int(row[0])
