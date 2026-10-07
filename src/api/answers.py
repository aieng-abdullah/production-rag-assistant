"""Answer trace endpoint (PLAN PR-4b-iii): GET /answers/{id}/trace.

Ownership enforced server-side: another tenant's answer id is a 404
(no existence leak).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from src.api.deps import require_user
from src.db.database import session_scope
from src.db.models import Answer, AnswerTrace

__all__ = ["router"]

router = APIRouter(prefix="/answers", tags=["answers"])


@router.get("/{answer_id}/trace")
def answer_trace(answer_id: int, user_id: int = Depends(require_user)) -> dict:
    """Provenance payload: chunks retrieved/cited, prompt version, model
    output (claims + quotes), verification verdicts, token usage."""
    with session_scope() as session:
        answer = session.get(Answer, answer_id)
        if answer is None or answer.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Answer not found"
            )
        traces = session.scalars(
            select(AnswerTrace)
            .where(AnswerTrace.answer_id == answer_id)
            .order_by(AnswerTrace.id)
        ).all()
        return {
            "answer": {
                "id": answer.id,
                "query": answer.query,
                "answer": answer.answer,
                "created_at": answer.created_at.isoformat(),
            },
            "trace": [trace.payload for trace in traces],
        }
