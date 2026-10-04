"""Answer + trace persistence (PLAN PR-4b-iii).

Framework-free: an open SQLAlchemy session goes in, the new answer id
comes out. Trace failures are handled by the caller — an un-persisted
trace degrades to `answer_id: null`, never blocks the answer.
"""

from sqlalchemy.orm import Session

from src.db.models import Answer, AnswerTrace

__all__ = ["persist_answer"]


def persist_answer(session: Session, user_id: int, query: str, cited) -> int:
    """Insert the answer row + its provenance trace payload; returns id."""
    answer = Answer(user_id=user_id, query=query, answer=cited.answer)
    session.add(answer)
    session.flush()

    payload = dict(cited.trace or {})
    payload.setdefault("verification", cited.verification)
    session.add(AnswerTrace(answer_id=answer.id, payload=payload))
    return int(answer.id)
