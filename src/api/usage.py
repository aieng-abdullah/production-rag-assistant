"""Usage stats endpoint (PLAN.md PR-3): quota state for the UI."""

from fastapi import APIRouter, Depends
from sqlalchemy import func

from src.api.deps import require_user
from src.db.database import session_scope
from src.db.models import Document
from src.services.quotas import (
    DAILY_QUERY_LIMIT,
    DOCUMENT_LIMIT,
    STORAGE_LIMIT_BYTES,
    queries_today,
    storage_bytes,
)

__all__ = ["router"]

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("")
def usage(user_id: int = Depends(require_user)) -> dict:
    """Current quota consumption — drives sidebar meters and 429 previews."""
    with session_scope() as session:
        used_queries = queries_today(session, user_id)
        used_docs = int(
            session.query(func.count(Document.id))
            .filter(Document.user_id == user_id)
            .scalar()
            or 0
        )
    return {
        "queries": {"used": used_queries, "limit": DAILY_QUERY_LIMIT},
        "documents": {"used": used_docs, "limit": DOCUMENT_LIMIT},
        "storage": {
            "used_bytes": storage_bytes(user_id),
            "limit_bytes": STORAGE_LIMIT_BYTES,
        },
    }
