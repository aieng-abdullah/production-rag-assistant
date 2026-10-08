"""Admin router: allow-list gate, user CRUD, stats (fail-closed mounting)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import Answer, AnswerTrace, Document, Subscription, UsageEvent, User, Workspace


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'admin.db'}")
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    monkeypatch.setattr(Config, "ADMIN_EMAILS", ["admin@example.com"])
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield create_app()
    reset_engine()


@pytest.fixture()
def client(api):
    return TestClient(api)


def _seed_user(email: str) -> int:
    with session_scope() as session:
        user = User(email=email, name=email.split("@")[0])
        session.add(user)
        session.flush()
        return user.id


def _headers(user_id: int) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


@pytest.fixture()
def admin_id():
    return _seed_user("admin@example.com")


@pytest.fixture()
def normal_id():
    return _seed_user("someone@example.com")


class TestGate:
    def test_no_token_401(self, client):
        assert client.get("/admin/stats").status_code == 401

    def test_non_admin_403(self, client, normal_id):
        assert client.get("/admin/stats", headers=_headers(normal_id)).status_code == 403

    def test_unknown_user_404(self, client, admin_id):
        assert client.get("/admin/stats", headers=_headers(999)).status_code == 404

    def test_admin_passes(self, client, admin_id):
        assert client.get("/admin/stats", headers=_headers(admin_id)).status_code == 200

    def test_email_match_case_insensitive(self, client, monkeypatch):
        uid = _seed_user("Admin@Example.COM")
        monkeypatch.setattr(Config, "ADMIN_EMAILS", ["ADMIN@example.com"])
        assert client.get("/admin/stats", headers=_headers(uid)).status_code == 200

    def test_router_unmounted_without_admin_emails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'x.db'}")
        monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(
            Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
        )
        monkeypatch.setattr(Config, "ADMIN_EMAILS", [])
        reset_engine()
        Base.metadata.create_all(get_engine())
        try:
            app = create_app()
            paths = [getattr(r, "path", "") for r in app.routes]
            assert not any(p.startswith("/admin") for p in paths)
        finally:
            reset_engine()


class TestUsers:
    def test_list_users(self, client, admin_id, normal_id):
        with session_scope() as session:
            session.add(Document(user_id=normal_id, filename="a.pdf", status="ready"))
        res = client.get("/admin/users", headers=_headers(admin_id))
        assert res.status_code == 200
        rows = res.json()
        assert {r["email"] for r in rows} == {"admin@example.com", "someone@example.com"}
        target = next(r for r in rows if r["id"] == normal_id)
        assert target["tier"] == "free"
        assert target["doc_count"] == 1

    def test_change_tier_creates_subscription(self, client, admin_id, normal_id):
        res = client.put(
            f"/admin/users/{normal_id}/tier",
            json={"tier": "pro"},
            headers=_headers(admin_id),
        )
        assert res.status_code == 200
        assert res.json()["tier"] == "pro"
        with session_scope() as session:
            sub = session.query(Subscription).filter_by(user_id=normal_id).one()
            assert sub.tier == "pro"

    def test_change_tier_updates_existing(self, client, admin_id, normal_id):
        with session_scope() as session:
            session.add(Subscription(user_id=normal_id, tier="pro"))
        res = client.put(
            f"/admin/users/{normal_id}/tier",
            json={"tier": "free"},
            headers=_headers(admin_id),
        )
        assert res.status_code == 200
        with session_scope() as session:
            sub = session.query(Subscription).filter_by(user_id=normal_id).one()
            assert sub.tier == "free"

    def test_invalid_tier_400(self, client, admin_id, normal_id):
        res = client.put(
            f"/admin/users/{normal_id}/tier",
            json={"tier": "gold"},
            headers=_headers(admin_id),
        )
        assert res.status_code == 400

    def test_tier_unknown_user_404(self, client, admin_id):
        res = client.put(
            "/admin/users/999/tier",
            json={"tier": "pro"},
            headers=_headers(admin_id),
        )
        assert res.status_code == 404


class TestDelete:
    def test_delete_purges_rows(self, client, admin_id, normal_id):
        with session_scope() as session:
            session.add(Document(user_id=normal_id, filename="a.pdf", status="ready"))
            answer = Answer(user_id=normal_id, query="q", answer="a")
            session.add(answer)
            session.flush()
            session.add(AnswerTrace(answer_id=answer.id, payload={"chunks": []}))
            session.add(UsageEvent(user_id=normal_id, kind="query", units=1))
            session.add(Workspace(user_id=normal_id, name="legal"))

        with patch("src.api.admin.RAGService") as rag:
            res = client.delete(
                f"/admin/users/{normal_id}", headers=_headers(admin_id)
            )
        assert res.status_code == 200
        rag.return_value.delete_tenant_data.assert_called_once_with(str(normal_id))

        with session_scope() as session:
            assert session.query(User).filter_by(id=normal_id).count() == 0
            assert session.query(Document).filter_by(user_id=normal_id).count() == 0
            assert session.query(Answer).filter_by(user_id=normal_id).count() == 0
            assert session.query(AnswerTrace).count() == 0
            assert session.query(UsageEvent).filter_by(user_id=normal_id).count() == 0
            assert session.query(Workspace).filter_by(user_id=normal_id).count() == 0

    def test_delete_self_400(self, client, admin_id):
        res = client.delete(f"/admin/users/{admin_id}", headers=_headers(admin_id))
        assert res.status_code == 400

    def test_delete_unknown_404(self, client, admin_id):
        with patch("src.api.admin.RAGService") as rag:
            res = client.delete("/admin/users/999", headers=_headers(admin_id))
        assert res.status_code == 404
        rag.assert_not_called()

    def test_vector_store_failure_502(self, client, admin_id, normal_id):
        with patch("src.api.admin.RAGService") as rag:
            rag.return_value.delete_tenant_data.side_effect = RuntimeError("down")
            res = client.delete(f"/admin/users/{normal_id}", headers=_headers(admin_id))
        assert res.status_code == 502
        with session_scope() as session:
            assert session.query(User).filter_by(id=normal_id).count() == 1


class TestStats:
    def test_stats_shape(self, client, admin_id, normal_id):
        with session_scope() as session:
            session.add(Document(user_id=normal_id, filename="a.pdf", status="ready"))
            session.add(Answer(user_id=normal_id, query="q", answer="a"))
            session.add(UsageEvent(user_id=normal_id, kind="query", units=1))
            session.add(UsageEvent(user_id=normal_id, kind="verify", units=1))
        res = client.get("/admin/stats", headers=_headers(admin_id))
        assert res.status_code == 200
        body = res.json()
        assert body == {
            "total_users": 2,
            "total_documents": 1,
            "total_answers": 1,
            # query + verify rows — same count /usage reports.
            "queries_today": 2,
        }
