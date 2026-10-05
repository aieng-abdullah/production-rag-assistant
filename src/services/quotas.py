"""Usage quotas (PLAN.md PR-3/PR-3b): queries/day, docs, storage.

Limits come from `Config` (env-tunable: `DAILY_QUERY_LIMIT`,
`DOCUMENT_LIMIT`, `STORAGE_LIMIT_MB`).

Framework-free: takes an open SQLAlchemy session, raises `QuotaExceeded`.
HTTP mapping (429) lives in `src/api/`.

Guest tier (PLAN PR-6): users provisioned by `POST /auth/anonymous` carry
an `anon-…@local` email and get the smaller guest limits (3 queries,
1 document); everyone else keeps the member limits.

Concurrency: check+insert happen in ONE statement (`INSERT ... SELECT ...
WHERE count < limit`). Under SQLite's single-writer lock that closes the
check-then-act race. Postgres multi-worker needs SERIALIZABLE/advisory
locks — tracked for PR-7 when the DB goes to Postgres.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, insert, literal, select
from sqlalchemy.orm import Session

from src.config import Config
from src.db.models import Document, UsageEvent, User

__all__ = [
    "ANON_EMAIL_PREFIX",
    "QuotaExceeded",
    "DAILY_QUERY_LIMIT",
    "DOCUMENT_LIMIT",
    "STORAGE_LIMIT_BYTES",
    "document_limit_for",
    "is_anonymous",
    "queries_today",
    "query_limit_for",
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
ANON_DOCUMENT_LIMIT = Config.ANON_DOCUMENT_LIMIT
# Guest query budget in usage UNITS: a verified answer burns query+verify
# = 2 units (PLAN PR-4b), so ANON_QUERY_LIMIT free questions = 6 units.
ANON_QUERY_LIMIT = Config.ANON_QUERY_LIMIT * 2
# Marker set by POST /auth/anonymous — the guest tier selector.
ANON_EMAIL_PREFIX = "anon-"


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


def is_anonymous(session: Session, user_id: int) -> bool:
    """True for guest-tier accounts minted by POST /auth/anonymous.

    Unknown user ids count as members (a valid JWT implies a row; this
    only guards malformed/stale tokens).
    """
    email = session.query(User.email).filter(User.id == user_id).scalar()
    return isinstance(email, str) and email.startswith(ANON_EMAIL_PREFIX)


def query_limit_for(session: Session, user_id: int) -> int:
    """Daily query-unit budget for this account's tier."""
    return ANON_QUERY_LIMIT if is_anonymous(session, user_id) else DAILY_QUERY_LIMIT


def document_limit_for(session: Session, user_id: int) -> int:
    """Document-count budget for this account's tier."""
    return (
        ANON_DOCUMENT_LIMIT if is_anonymous(session, user_id) else DOCUMENT_LIMIT
    )


def _query_quota_error(limit: int) -> QuotaExceeded:
    return QuotaExceeded(
        f"Daily query quota reached ({limit} queries/day). "
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
    The budget is the caller's tier limit (guest vs member).
    """
    limit = query_limit_for(session, user_id)
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
            ).where(today < limit),
        )
        .returning(UsageEvent.id)
    )
    row = session.execute(stmt).first()
    if row is None:
        raise _query_quota_error(limit)
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
    limit = document_limit_for(session, user_id)
    docs = (
        session.query(func.count(Document.id))
        .filter(Document.user_id == user_id)
        .scalar()
    )
    if int(docs or 0) >= limit:
        raise QuotaExceeded(
            f"Document quota reached ({limit} documents). "
            "Delete a document to upload another."
        )
    if storage_bytes(user_id) + incoming_bytes > STORAGE_LIMIT_BYTES:
        raise QuotaExceeded(
            f"Storage quota reached ({STORAGE_LIMIT_BYTES // (1024 * 1024)}MB). "
            "Delete documents to free space."
        )


def reserve_document_slot(session: Session, user_id: int, filename: str) -> int:
    """Atomically count+insert the Document row; returns its id.

    Same single-statement pattern as `reserve_query_slot` — the tier's
    document limit holds under concurrent uploads.
    """
    limit = document_limit_for(session, user_id)
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
            ).where(doc_count < limit),
        )
        .returning(Document.id)
    )
    row = session.execute(stmt).first()
    if row is None:
        raise QuotaExceeded(
            f"Document quota reached ({limit} documents). "
            "Delete a document to upload another."
        )
    return int(row[0])
