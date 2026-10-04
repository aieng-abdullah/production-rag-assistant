"""Chat endpoint (PLAN.md PR-3): quota → retrieve → generate → usage.

Response shape matches today's Streamlit app exactly: `answer` +
`sources[{doc_id, page_num, text}]` (PR-6 swaps the client, not the shape).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from src.api.deps import quota_to_http, require_user
from src.db.database import session_scope
from src.services import RAGService
from src.services.bm25_cache import get_bm25
from src.services.quotas import QuotaExceeded, check_query_quota, record_usage

__all__ = ["router"]

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@router.post("")
def chat(body: ChatRequest, user_id: int = Depends(require_user)) -> dict:
    """Tenant-scoped answer. 429 on exhausted daily quota; usage recorded
    only on success (failed generations are free)."""
    tenant = str(user_id)
    try:
        with session_scope() as session:
            check_query_quota(session, user_id)
    except QuotaExceeded as exc:
        raise quota_to_http(exc) from exc

    try:
        cited = RAGService().generate_answer(
            tenant, body.query, bm25_index=get_bm25(tenant)
        )
    except Exception as exc:
        logger.error(f"Generation failed tenant={tenant}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Answer generation failed",
        ) from exc

    with session_scope() as session:
        record_usage(session, user_id, "query")

    return {
        "answer": cited.answer,
        "sources": [source.model_dump() for source in cited.sources],
    }
