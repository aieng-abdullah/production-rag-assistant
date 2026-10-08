"""Chat endpoint (PLAN.md PR-3): quota → retrieve → generate → usage.

Response shape: `answer` + `sources[{doc_id, page_num, text}]` plus the
PLAN PR-4b `verification{status, per_claim[]}` badge (additive).

Quota: a verified answer costs 2 units of the daily pool — one `query`
row + one `verify` row (PLAN PR-4b). Both reserved atomically before
generation; both refunded on failure (failed generations are free).
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from src.api.deps import quota_to_http, require_user
from src.config import Config
from src.db.database import session_scope
from src.generation.sanitize import sanitize_untrusted
from src.services import RAGService
from src.services.bm25_cache import get_bm25
from src.services.quotas import (
    QuotaExceeded,
    refund_usage,
    reserve_query_slot,
    reserve_verify_slot,
)
from src.services.trace_store import persist_answer

__all__ = ["router"]

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatTurn(BaseModel):
    """One prior turn of client-held conversation history.

    Content mirrors the query constraints and is re-sanitized at the
    boundary — history enters prompts and traces exactly like the query.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    workspace: Literal["legal", "academic"] = Config.DEFAULT_WORKSPACE
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


def _sanitize_history(turns: list[ChatTurn]) -> list[dict]:
    """Sanitize every turn before quota reservation and before persistence.

    A turn that is empty after sanitization is rejected with 422 — the same
    rule as the query — so the client never believes context was accepted
    when it was dropped. Returns plain dicts for the service layer.
    """
    cleaned: list[dict] = []
    for index, turn in enumerate(turns):
        content = sanitize_untrusted(turn.content)
        if not content.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"History turn {index} has no usable content",
            )
        cleaned.append({"role": turn.role, "content": content})
    return cleaned


@router.post("")
def chat(body: ChatRequest, user_id: int = Depends(require_user)) -> dict:
    """Tenant-scoped answer. 429 on exhausted daily quota; the quota slots
    (query + verify = 2 units) are reserved atomically BEFORE generation
    and refunded on failure. `workspace` picks the niche: prompt
    profile + retrieval filter (PLAN PR-4).

    The query is sanitized at the API boundary — before any quota is
    reserved and before anything is persisted — so stored traces and the
    prompt builder see the same delimiter-free, control-char-free text.
    `history` follows the same rule and costs nothing extra: one request
    is always 2 units regardless of history length.
    """
    query = sanitize_untrusted(body.query)
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query has no usable content",
        )
    history = _sanitize_history(body.history)
    tenant = str(user_id)
    try:
        with session_scope() as session:
            usage_id = reserve_query_slot(session, user_id)
            verify_id = reserve_verify_slot(session, user_id)
    except QuotaExceeded as exc:
        # Session rolled back — both units released, nothing burned.
        raise quota_to_http(exc) from exc

    try:
        cited = RAGService().generate_answer(
            tenant,
            query,
            bm25_index=get_bm25(tenant, body.workspace),
            workspace=body.workspace,
            history=history,
        )
    except Exception as exc:
        try:
            with session_scope() as session:
                refund_usage(session, usage_id)
                refund_usage(session, verify_id)
        except Exception as refund_exc:
            # Slots stay burned (quota leak, not auth issue) — never mask the 502.
            logger.error(
                f"Quota refund failed usage={usage_id}/{verify_id}: {refund_exc}"
            )
        logger.error(f"Generation failed tenant={tenant}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Retrieval or generation failed",
        ) from exc

    answer_id: int | None = None
    try:
        with session_scope() as session:
            answer_id = persist_answer(session, user_id, query, cited)
    except Exception as exc:
        # Provenance is non-critical — the answer ships with answer_id: null.
        logger.warning(f"Answer trace persist failed tenant={tenant}: {exc}")

    return {
        "answer_id": answer_id,
        "answer": cited.answer,
        "sources": [source.model_dump() for source in cited.sources],
        "verification": cited.verification,
    }
