"""Usage stats endpoint (PLAN.md PR-3): quota state for the UI."""

from fastapi import APIRouter, Depends
from sqlalchemy import func

from src.api.deps import require_user
from src.db.database import session_scope
from src.db.models import Document
from src.services.quotas import (
    STORAGE_LIMIT_BYTES,
    document_limit_for,
    is_anonymous,
    queries_today,
    query_limit_for,
    storage_bytes,
)

__all__ = ["router"]

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("")
def usage(user_id: int = Depends(require_user)) -> dict:
    """Current quota consumption — drives sidebar meters and 429 previews.

    Limits are tier-aware (guest vs member, PLAN PR-6); `tier` lets the
    UI show guest copy without re-deriving it from the email.
    """
    with session_scope() as session:
        used_queries = queries_today(session, user_id)
        used_docs = int(
            session.query(func.count(Document.id))
            .filter(Document.user_id == user_id)
            .scalar()
            or 0
        )
        tier = "anonymous" if is_anonymous(session, user_id) else "member"
        query_limit = query_limit_for(session, user_id)
        document_limit = document_limit_for(session, user_id)
    return {
        "tier": tier,
        "queries": {"used": used_queries, "limit": query_limit},
        "documents": {"used": used_docs, "limit": document_limit},
        "storage": {
            "used_bytes": storage_bytes(user_id),
            "limit_bytes": STORAGE_LIMIT_BYTES,
        },
    }
