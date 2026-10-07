"""Auth endpoint tests (PLAN.md PR-2b).

Covers: 501 when Google creds absent, OAuth state CSRF, callback → user
upsert + JWT redirect, `require_user` 401/500 paths, and the security-gate-3
promise that issued tokens never reach the logs.
"""

from urllib.parse import parse_qs, urlparse

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from loguru import logger as loguru_logger

from src.api.app import create_app
from src.api.deps import require_user
from src.api.security import TokenError, create_token, decode_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import User

FAKE_PROFILE = {
    "sub": "google-sub-1",
    "email": "dev@example.com",
    "name": "Dev Example",
    "picture": "https://example.com/pic.png",
}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """Fresh SQLite DB + app with Google creds and JWT secret configured."""
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'auth.db'}")
    monkeypatch.setattr(Config, "GOOGLE_CLIENT_ID", "cid-test")
    monkeypatch.setattr(Config, "GOOGLE_CLIENT_SECRET", "csec-test")
    monkeypatch.setattr(Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef")
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield create_app()
    reset_engine()


@pytest.fixture()
def client(api):
    return TestClient(api)


def _oauth_state(client: TestClient) -> str:
    """Kick off OAuth, return the state value Google would echo back."""
    response = client.get("/auth/google", follow_redirects=False)
    assert response.status_code == 302
    location = urlparse(response.headers["location"])
    assert location.netloc == "accounts.google.com"
    state = parse_qs(location.query)["state"][0]
    assert state, "state param missing from authorize URL"
    # TestClient persists the state cookie for the follow-up callback.
    return state


def _redirect_token(location: str) -> str:
    """JWT rides the redirect fragment (`#token=...`), never the query string."""
    parsed = urlparse(location)
    assert not parsed.query, f"token leaked into query string: {parsed.query}"
    return parse_qs(parsed.fragment)["token"][0]


def test_google_route_501_without_credentials(api, monkeypatch):
    monkeypatch.setattr(Config, "GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr(Config, "GOOGLE_CLIENT_SECRET", "")
    client = TestClient(api)

    response = client.get("/auth/google", follow_redirects=False)

    assert response.status_code == 501
    assert "GOOGLE_CLIENT_ID" in response.json()["detail"]


def test_callback_501_without_credentials(api, monkeypatch):
    monkeypatch.setattr(Config, "GOOGLE_CLIENT_SECRET", "")
    client = TestClient(api)

    response = client.get("/auth/google/callback", params={"code": "x", "state": "y"})

    assert response.status_code == 501


def test_authorize_sets_state_cookie(client):
    state = _oauth_state(client)

    assert "oauth_state" in client.cookies
    assert client.cookies["oauth_state"] == state


def test_callback_rejects_missing_state(client):
    response = client.get(
        "/auth/google/callback", params={"code": "abc"}, follow_redirects=False
    )

    assert response.status_code == 400


def test_callback_rejects_state_mismatch(client):
    _oauth_state(client)

    response = client.get(
        "/auth/google/callback",
        params={"code": "abc", "state": "forged-state"},
        follow_redirects=False,
    )

    assert response.status_code == 400


def test_callback_creates_user_and_redirects_with_jwt(api, monkeypatch):
    monkeypatch.setattr("src.api.auth._exchange_code", lambda code: FAKE_PROFILE)
    client = TestClient(api)
    state = _oauth_state(client)

    response = client.get(
        "/auth/google/callback",
        params={"code": "valid-code", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 302
    token = _redirect_token(response.headers["location"])
    with session_scope() as session:
        user = session.query(User).one()
    assert decode_token(token) == user.id
    assert user.email == "dev@example.com"
    assert user.google_sub == "google-sub-1"


def test_callback_upserts_existing_user(api, monkeypatch):
    monkeypatch.setattr("src.api.auth._exchange_code", lambda code: FAKE_PROFILE)
    client = TestClient(api)

    state = _oauth_state(client)
    client.get(
        "/auth/google/callback",
        params={"code": "c1", "state": state},
        follow_redirects=False,
    )
    state = _oauth_state(client)
    response = client.get(
        "/auth/google/callback",
        params={"code": "c2", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 302
    with session_scope() as session:
        assert session.query(User).count() == 1


def test_callback_maps_httpx_failure_to_502(api, monkeypatch):
    import httpx

    def httpx_boom(code):
        raise httpx.HTTPError("connection refused")

    monkeypatch.setattr("src.api.auth._exchange_code", httpx_boom)
    client = TestClient(api)
    state = _oauth_state(client)

    response = client.get(
        "/auth/google/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 502


def test_callback_maps_oauth_error_to_502(api, monkeypatch):
    """Expired/replayed code → authlib OAuthError (invalid_grant), not 500."""
    from authlib.integrations.base_client.errors import OAuthError

    def oauth_boom(code):
        raise OAuthError(error="invalid_grant")

    monkeypatch.setattr("src.api.auth._exchange_code", oauth_boom)
    client = TestClient(api)
    state = _oauth_state(client)

    response = client.get(
        "/auth/google/callback",
        params={"code": "c", "state": state},
        follow_redirects=False,
    )

    assert response.status_code == 502


def test_require_user_401_without_token():
    client = TestClient(_protected_app())

    assert client.get("/whoami").status_code == 401


def test_require_user_401_on_garbage_token(api):
    client = TestClient(_protected_app())

    response = client.get("/whoami", headers={"Authorization": "Bearer not.a.jwt"})

    assert response.status_code == 401


def test_require_user_401_on_expired_token(api):
    expired = create_token(7, ttl_days=-1)
    client = TestClient(_protected_app())

    response = client.get(
        "/whoami", headers={"Authorization": f"Bearer {expired}"}
    )

    assert response.status_code == 401


def test_require_user_401_on_tampered_signature(api):
    token = create_token(42)
    tampered = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
    client = TestClient(_protected_app())

    response = client.get(
        "/whoami", headers={"Authorization": f"Bearer {tampered}"}
    )

    assert response.status_code == 401


def test_require_user_returns_user_id(api):
    token = create_token(42)
    client = TestClient(_protected_app())

    response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"user_id": 42}


def test_require_user_500_when_jwt_secret_missing(api, monkeypatch):
    monkeypatch.setattr(Config, "JWT_SECRET", "")
    client = TestClient(_protected_app())

    response = client.get(
        "/whoami", headers={"Authorization": "Bearer whatever.token.here"}
    )

    assert response.status_code == 500


def test_create_token_without_secret_raises(monkeypatch):
    monkeypatch.setattr(Config, "JWT_SECRET", "")

    with pytest.raises(EnvironmentError):
        create_token(1)


def test_decode_token_rejects_wrong_secret(api):
    forged = pyjwt.encode(
        {"sub": "1"}, "wrong-secret-0123456789abcdef0123456789ab", algorithm="HS256"
    )

    with pytest.raises(TokenError):
        decode_token(forged)


def test_issued_token_never_logged(api, monkeypatch):
    """Security gate 3: callback path emits no token material into logs."""
    records: list[str] = []
    sink_id = loguru_logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        monkeypatch.setattr("src.api.auth._exchange_code", lambda code: FAKE_PROFILE)
        client = TestClient(api)
        state = _oauth_state(client)
        response = client.get(
            "/auth/google/callback",
            params={"code": "c", "state": state},
            follow_redirects=False,
        )
        assert response.status_code == 302
        token = _redirect_token(response.headers["location"])
    finally:
        loguru_logger.remove(sink_id)

    assert token
    assert token not in "\n".join(records)


def _protected_app() -> FastAPI:
    """Minimal app exposing /whoami behind require_user."""
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user_id: int = Depends(require_user)):
        return {"user_id": user_id}

    return app


# --- Demo login (PLAN PR-6) ---


@pytest.fixture()
def demo_api(tmp_path, monkeypatch):
    """App with demo mode on: local JWT secret, fresh DB, no Google needed."""
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'demo.db'}")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    monkeypatch.setattr(Config, "ENABLE_DEMO_LOGIN", "on")
    # Chroma is out of scope for auth tests — keep the suite fast/hermetic.
    monkeypatch.setattr("src.db.qdrant_client.has_chunks", lambda tenant: False)
    monkeypatch.setattr(
        "src.db.qdrant_client.reassign_tenant", lambda old, new: 0
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield create_app()
    reset_engine()


def test_demo_login_issues_valid_token(demo_api):
    client = TestClient(demo_api)

    response = client.post("/auth/demo")

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "demo@local"
    assert body["user_id"] >= 1
    assert decode_token(body["token"]) == body["user_id"]


def test_demo_login_upserts_single_user(demo_api):
    client = TestClient(demo_api)

    first = client.post("/auth/demo").json()
    second = client.post("/auth/demo").json()

    assert first["user_id"] == second["user_id"]
    with session_scope() as session:
        demo_rows = session.query(User).filter_by(email="demo@local").all()
        assert len(demo_rows) == 1


def test_demo_token_roundtrips_require_user(demo_api):
    token = client_post_demo_token(demo_api)
    app = _protected_app()

    response = TestClient(app).get(
        "/whoami", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200


def test_demo_route_absent_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'off.db'}")
    monkeypatch.setattr(Config, "ENABLE_DEMO_LOGIN", "off")
    reset_engine()
    Base.metadata.create_all(get_engine())
    app = create_app()
    reset_engine()

    response = TestClient(app).post("/auth/demo")

    assert response.status_code == 404


def test_demo_adopts_legacy_default_tenant(tmp_path, monkeypatch):
    """First demo sign-in moves the pre-API `default` corpus to the user."""
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'mig.db'}")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    monkeypatch.setattr(Config, "ENABLE_DEMO_LOGIN", "on")
    monkeypatch.setattr("src.db.qdrant_client.has_chunks", lambda tenant: False)
    moved: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.db.qdrant_client.reassign_tenant", lambda old, new: moved.append((old, new)) or 5
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    client = TestClient(create_app())
    reset_engine()

    body = client.post("/auth/demo").json()

    assert moved == [("default", str(body["user_id"]))]


def test_demo_token_never_logged(demo_api):
    """Security gate 3 extends to demo tokens."""
    records: list[str] = []
    sink_id = loguru_logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        token = client_post_demo_token(demo_api)
    finally:
        loguru_logger.remove(sink_id)

    assert token not in "\n".join(records)


def client_post_demo_token(app) -> str:
    """POST /auth/demo on a fresh client, return the issued token."""
    return TestClient(app).post("/auth/demo").json()["token"]
