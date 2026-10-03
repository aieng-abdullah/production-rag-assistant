"""Tests for SQLAlchemy models (PLAN.md PR-2a)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.models import Answer, Document, Subscription, User, Workspace


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()
    engine.dispose()


def _user(session, email="a@example.com"):
    user = User(email=email, name="A")
    session.add(user)
    session.flush()
    return user


def test_user_roundtrip(session):
    user = _user(session)
    session.commit()

    loaded = session.query(User).one()
    assert loaded.id == user.id
    assert loaded.email == "a@example.com"
    assert loaded.created_at is not None


def test_duplicate_email_rejected(session):
    _user(session, email="dup@example.com")
    session.commit()

    session.add(User(email="dup@example.com"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_workspace_unique_per_user(session):
    user = _user(session)
    session.commit()

    session.add(Workspace(user_id=user.id, name="legal"))
    session.commit()

    session.add(Workspace(user_id=user.id, name="legal"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    # Same name for a different user is fine.
    other = User(email="b@example.com")
    session.add(other)
    session.flush()
    session.add(Workspace(user_id=other.id, name="legal"))
    session.commit()


def test_document_status_default(session):
    user = _user(session)
    session.commit()

    doc = Document(user_id=user.id, filename="paper.pdf")
    session.add(doc)
    session.commit()

    assert session.query(Document).one().status == "processing"


def test_subscription_one_per_user(session):
    user = _user(session)
    session.commit()

    session.add(Subscription(user_id=user.id, tier="free"))
    session.commit()

    session.add(Subscription(user_id=user.id, tier="pro"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_answer_and_usage_event_link_user(session):
    user = _user(session)
    session.commit()

    answer = Answer(user_id=user.id, query="q", answer="a")
    session.add(answer)
    session.flush()
    assert answer.id is not None

    from src.db.models import UsageEvent

    session.add(UsageEvent(user_id=user.id, kind="query", units=1))
    session.commit()
    assert session.query(UsageEvent).count() == 1
