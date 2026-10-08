"""Usage quotas — two surfaces over one usage table (PLAN PR-3/3b/6).

**Session surface** (FastAPI, restored in PLAN-render-react Phase 3):
takes an open SQLAlchemy session, raises :class:`QuotaExceeded`.
Atomic check+insert (`INSERT ... SELECT ... WHERE count < limit`) closes
the check-then-act race under SQLite's single-writer lock. Guest tier
(PLAN PR-6): accounts minted by ``POST /auth/anonymous`` carry an
``anon-…@local`` email and get the smaller guest limits (3 queries,
1 document). Members get ``DAILY_QUERY_LIMIT``/``DOCUMENT_LIMIT``;
a ``Subscription.tier = "pro"`` row (admin endpoint or Stripe webhook)
multiplies both by ``TIER_MULTIPLIERS["pro"]`` (10x). Storage bytes
share one cap across tiers.

**Tier surface** (Streamlit single-process, PLAN PR-2a persistence):
``check_*_quota(user_id, tier) -> QuotaResult`` — free tier 1x, pro tier
10x (``TIER_MULTIPLIERS``) — plus ``record_*_usage`` writers.

HTTP mapping (429) lives in ``src/api/``; this module stays framework-free.

Naming: ``check_document_quota`` keeps the session (raising) signature —
the restored routers and their tests pin it. The tier variant that
returns ``QuotaResult`` is ``check_tier_document_quota``.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import func, insert, literal, select
from sqlalchemy.orm import Session, sessionmaker

from src.config import Config
from src.db.database import get_session_factory as get_default_session_factory
from src.db.models import Document, Subscription, UsageEvent, User
from src.services.usage import get_daily_usage, get_document_count, record_usage

__all__ = [
    "ANON_EMAIL_PREFIX",
    "DAILY_QUERY_LIMIT",
    "DOCUMENT_LIMIT",
    "QuotaExceeded",
    "QuotaResult",
    "STORAGE_LIMIT_BYTES",
    "TIER_MULTIPLIERS",
    "check_document_quota",
    "check_query_quota",
    "check_tier_document_quota",
    "clear_session_factory_override",
    "document_limit_for",
    "enforce_storage_quota",
    "is_anonymous",
    "queries_today",
    "query_limit_for",
    "record_ingest_usage",
    "record_query_usage",
    "record_verify_usage",
    "refund_usage",
    "reserve_document_slot",
    "reserve_query_slot",
    "reserve_verify_slot",
    "set_session_factory_override",
    "storage_bytes",
    "tier_for",
]

# --- Session surface constants (restored API) -------------------------------

DAILY_QUERY_LIMIT = Config.DAILY_QUERY_LIMIT
DOCUMENT_LIMIT = Config.DOCUMENT_LIMIT
STORAGE_LIMIT_BYTES = Config.STORAGE_LIMIT_MB * 1024 * 1024
ANON_DOCUMENT_LIMIT = Config.ANON_DOCUMENT_LIMIT
# Guest query budget in usage UNITS: a verified answer burns query+verify
# = 2 units (PLAN PR-4b), so ANON_QUERY_LIMIT free questions = 6 units.
ANON_QUERY_LIMIT = Config.ANON_QUERY_LIMIT * 2
# Marker set by POST /auth/anonymous — the guest tier selector.
ANON_EMAIL_PREFIX = "anon-"

# --- Tier surface constants (Streamlit) -------------------------------------

TIER_MULTIPLIERS = {
    "free": 1,
    "pro": 10,
}


# --- Session surface (FastAPI) ----------------------------------------------


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


def tier_for(session: Session, user_id: int) -> str:
    """Effective billing tier: ``"anonymous"`` (guest), ``"pro"`` or ``"free"``.

    Reads the ``Subscription`` row written by the admin tier endpoint or the
    Stripe webhook; guests never resolve to pro (their budget is the fixed
    guest limit, not a multiplier of the member limit).
    """
    if is_anonymous(session, user_id):
        return "anonymous"
    tier = (
        session.query(Subscription.tier).filter(Subscription.user_id == user_id).scalar()
    )
    return tier if tier in TIER_MULTIPLIERS else "free"


def query_limit_for(session: Session, user_id: int) -> int:
    """Daily query-unit budget for this account's tier.

    Pro multiplies the member limit by ``TIER_MULTIPLIERS`` (10x);
    guests keep the fixed ``ANON_QUERY_LIMIT``.
    """
    tier = tier_for(session, user_id)
    if tier == "anonymous":
        return ANON_QUERY_LIMIT
    return DAILY_QUERY_LIMIT * _get_multiplier(tier)


def document_limit_for(session: Session, user_id: int) -> int:
    """Document-count budget for this account's tier (pro = 10x member).

    Storage bytes are intentionally NOT multiplied — one shared cap.
    """
    tier = tier_for(session, user_id)
    if tier == "anonymous":
        return ANON_DOCUMENT_LIMIT
    return DOCUMENT_LIMIT * _get_multiplier(tier)


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
    The budget is the caller's tier limit (guest / member / pro).
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


# --- Tier surface (Streamlit, persisted) ------------------------------------


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


def check_tier_document_quota(user_id: int, tier: str) -> QuotaResult:
    """
    Check if user can upload a document (tier surface — returns a result,
    never raises).

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
