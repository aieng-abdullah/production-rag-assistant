"""Admin router — user management and aggregate stats.

Every route resolves the caller via require_user, then checks that their
email is in Config.ADMIN_EMAILS (403 otherwise). Router only mounts when
ADMIN_EMAILS is non-empty (fail-closed, see src/api/app.py).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func, select

from src.api.deps import require_user
from src.config import Config
from src.db.database import session_scope
from src.db.models import Answer, AnswerTrace, Document, Subscription, UsageEvent, User, Workspace
from src.services import RAGService

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(user_id: int) -> None:
    """Raise 404/403 unless the authenticated user is listed in ADMIN_EMAILS.

    Comparison is case-insensitive: Google emails may preserve local-part
    case and Config.ADMIN_EMAILS comes from operator input.
    """
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user.email.lower() not in {e.lower() for e in Config.ADMIN_EMAILS}:
            logger.warning(f"Admin denied user={user_id} email={user.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )


class TierUpdate(BaseModel):
    tier: str  # "free" | "pro"


@router.get("/users")
def list_users(user_id: int = Depends(require_user)) -> list[dict]:
    _require_admin(user_id)
    with session_scope() as session:
        rows = (
            session.query(User, Subscription.tier)
            .outerjoin(Subscription, Subscription.user_id == User.id)
            .order_by(User.created_at.desc())
            .all()
        )
        doc_counts = dict(
            session.query(Document.user_id, func.count(Document.id))
            .group_by(Document.user_id)
            .all()
        )
        return [
            {
                "id": u.id,
                "email": u.email,
                "name": u.name,
                "tier": tier or "free",
                "doc_count": doc_counts.get(u.id, 0),
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u, tier in rows
        ]


@router.put("/users/{target_user_id}/tier")
def update_user_tier(
    target_user_id: int,
    payload: TierUpdate,
    user_id: int = Depends(require_user),
) -> dict:
    """DB tier flip only — no Stripe sync. With billing live, the next
    webhook can revert this; upgrade real customers via Stripe instead."""
    _require_admin(user_id)
    if payload.tier not in ("free", "pro"):
        raise HTTPException(status_code=400, detail="tier must be 'free' or 'pro'")
    with session_scope() as session:
        target = session.get(User, target_user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Target user not found")
        sub = session.query(Subscription).filter_by(user_id=target_user_id).first()
        if sub is None:
            sub = Subscription(user_id=target_user_id, tier=payload.tier)
            session.add(sub)
        else:
            sub.tier = payload.tier
        logger.info(
            f"Admin tier change actor={user_id} target={target_user_id} "
            f"email={target.email} tier={payload.tier}"
        )
        return {"id": target.id, "email": target.email, "tier": payload.tier}


@router.delete("/users/{target_user_id}")
def delete_user(target_user_id: int, user_id: int = Depends(require_user)) -> dict:
    """Purge a user account: vectors/files first, then all DB rows.

    SQLite does not enforce FK cascades here (no PRAGMA foreign_keys), so
    child rows are deleted explicitly in dependency order.
    """
    _require_admin(user_id)
    if target_user_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    with session_scope() as session:
        target = session.get(User, target_user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="Target user not found")
        target_email = target.email

    # Vectors/files first (house pattern — retryable), DB rows last.
    try:
        RAGService().delete_tenant_data(str(target_user_id))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store purge failed",
        ) from exc

    with session_scope() as session:
        answer_id_subq = select(Answer.id).where(Answer.user_id == target_user_id)
        session.query(AnswerTrace).filter(
            AnswerTrace.answer_id.in_(answer_id_subq)
        ).delete(synchronize_session=False)
        for model in (Answer, UsageEvent, Document, Workspace, Subscription):
            session.query(model).filter(model.user_id == target_user_id).delete(
                synchronize_session=False
            )
        remaining = session.get(User, target_user_id)
        if remaining is not None:
            session.delete(remaining)
    logger.info(
        f"Admin delete actor={user_id} target={target_user_id} email={target_email}"
    )
    return {"deleted": True, "id": target_user_id}


@router.get("/stats")
def admin_stats(user_id: int = Depends(require_user)) -> dict:
    _require_admin(user_id)
    with session_scope() as session:
        total_users = session.query(func.count(User.id)).scalar() or 0
        total_docs = session.query(func.count(Document.id)).scalar() or 0
        total_answers = session.query(func.count(Answer.id)).scalar() or 0
        midnight = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        queries_today = (
            session.query(func.count(UsageEvent.id))
            .filter(
                # Same rows /usage counts (query + verify) so the admin
                # number matches the per-user "Queries today" meter.
                UsageEvent.kind.in_(["query", "verify"]),
                UsageEvent.created_at >= midnight,
            )
            .scalar()
            or 0
        )
        return {
            "total_users": total_users,
            "total_documents": total_docs,
            "total_answers": total_answers,
            "queries_today": queries_today,
        }
