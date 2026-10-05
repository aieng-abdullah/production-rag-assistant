"""Guest tier tests (PLAN PR-6): anonymous sessions, tier-aware quotas.

ChatGPT-style free try-out — 3 questions + 1 upload, then the login wall.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import Document, UsageEvent, User
from src.services.quotas import (
    QuotaExceeded,
    check_document_quota,
    document_limit_for,
    is_anonymous,
    query_limit_for,
    reserve_query_slot,
)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'guest.db'}")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield create_app()
    reset_engine()


@pytest.fixture()
def client(api):
    return TestClient(api)


def _guest_id() -> int:
    with session_scope() as session:
        user = User(email="anon-dev1ce-0001@local", name="Guest")
        session.add(user)
        session.flush()
        return int(user.id)


# --- POST /auth/anonymous ---


def test_anonymous_issues_working_token(client):
    response = client.post("/auth/anonymous", json={"device_id": "device-0001-abc"})

    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "anonymous"
    assert body["email"] == "Guest"

    whoami = client.get("/usage", headers={"Authorization": f"Bearer {body['token']}"})
    assert whoami.status_code == 200
    assert whoami.json()["tier"] == "anonymous"


def test_anonymous_upserts_one_account_per_device(client):
    first = client.post("/auth/anonymous", json={"device_id": "device-0001-abc"})
    second = client.post("/auth/anonymous", json={"device_id": "device-0001-abc"})

    assert first.json()["user_id"] == second.json()["user_id"]
    with session_scope() as session:
        rows = session.query(User).filter(User.email.like("anon-%")).all()
        assert len(rows) == 1


def test_anonymous_rejects_malformed_device_id(client):
    response = client.post("/auth/anonymous", json={"device_id": "DROP TABLE"})

    assert response.status_code == 422


def test_anonymous_500_without_jwt_secret(client, monkeypatch):
    monkeypatch.setattr(Config, "JWT_SECRET", "")

    response = client.post("/auth/anonymous", json={"device_id": "device-0001-abc"})

    assert response.status_code == 500


# --- Tier-aware quota rules ---


def test_is_anonymous_detects_guest_email(api):
    guest_id = _guest_id()
    with session_scope() as session:
        assert is_anonymous(session, guest_id) is True
        assert is_anonymous(session, 999_999) is False


def test_guest_query_budget_is_three_questions_in_units(api):
    """3 questions × (query+verify) = 6 units; the 7th unit raises."""
    guest_id = _guest_id()
    with session_scope() as session:
        assert query_limit_for(session, guest_id) == Config.ANON_QUERY_LIMIT * 2
        for _ in range(6):
            reserve_query_slot(session, guest_id)
        with pytest.raises(QuotaExceeded, match="6 queries/day"):
            reserve_query_slot(session, guest_id)


def test_member_keeps_daily_query_limit(api):
    member_id = _guest_id() + 1
    with session_scope() as session:
        session.add(User(email="member@example.com", name="Member"))
    with session_scope() as session:
        assert query_limit_for(session, member_id) == Config.DAILY_QUERY_LIMIT


def test_guest_document_limit_is_one(api):
    guest_id = _guest_id()
    with session_scope() as session:
        assert document_limit_for(session, guest_id) == 1
        session.add(
            Document(user_id=guest_id, filename="a.pdf", status="ready")
        )
    with session_scope() as session:
        with pytest.raises(QuotaExceeded, match="1 documents"):
            check_document_quota(session, guest_id, 10)


def test_member_document_limit_unchanged(api):
    member_id = _guest_id() + 1
    with session_scope() as session:
        session.add(User(email="member2@example.com", name="Member"))
    with session_scope() as session:
        assert document_limit_for(session, member_id) == Config.DOCUMENT_LIMIT


# --- End-to-end through the API ---


def test_guest_chat_429_after_free_questions(client):
    guest_id = _guest_id()
    token = create_token(guest_id)
    headers = {"Authorization": f"Bearer {token}"}
    with session_scope() as session:
        for _ in range(6):
            session.add(UsageEvent(user_id=guest_id, kind="query", units=1))

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 429
    assert "6 queries/day" in response.json()["detail"]


def test_usage_reports_guest_tier_and_limits(client):
    guest_id = _guest_id()
    token = create_token(guest_id)

    body = client.get(
        "/usage", headers={"Authorization": f"Bearer {token}"}
    ).json()

    assert body["tier"] == "anonymous"
    assert body["queries"]["limit"] == Config.ANON_QUERY_LIMIT * 2
    assert body["documents"]["limit"] == 1
