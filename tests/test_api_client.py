"""API client tests (PLAN PR-6): request shaping, auth header, error paths.

All HTTP goes through httpx.MockTransport — no network, no FastAPI app.
"""

from unittest.mock import patch

import httpx
import pytest

import api_client
from api_client import APIError


def _mount(handler, monkeypatch, token: str | None = None) -> None:
    monkeypatch.setattr(api_client, "_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(api_client, "_token", lambda: token)


def test_list_documents_sends_bearer_token(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[{"id": 1, "status": "ready"}])

    _mount(handler, monkeypatch, token="tok-123")

    docs = api_client.list_documents()

    assert docs == [{"id": 1, "status": "ready"}]
    assert seen["auth"] == "Bearer tok-123"


def test_no_token_omits_authorization_header(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[])

    _mount(handler, monkeypatch, token=None)

    api_client.list_documents()

    assert seen["auth"] is None


def test_401_clears_session(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid credentials"})

    _mount(handler, monkeypatch, token="stale")
    with patch.object(api_client, "_session_clear") as clear:
        with pytest.raises(APIError) as exc:
            api_client.usage()

    assert exc.value.status == 401
    clear.assert_called_once()


def test_429_surfaces_quota_detail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Daily query limit reached"})

    _mount(handler, monkeypatch, token="tok")

    with pytest.raises(APIError) as exc:
        api_client.ask("hi", "legal")

    assert exc.value.status == 429
    assert "Daily query limit" in exc.value.detail


def test_502_detail_passthrough(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "Retrieval or generation failed"})

    _mount(handler, monkeypatch, token="tok")

    with pytest.raises(APIError) as exc:
        api_client.ask("hi", "academic")

    assert exc.value.status == 502


def test_unreachable_backend_reports_status_zero(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _mount(handler, monkeypatch, token="tok")

    with pytest.raises(APIError) as exc:
        api_client.usage()

    assert exc.value.status == 0
    assert "uvicorn" in exc.value.detail


def test_ask_posts_chat_payload(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.read()
        return httpx.Response(
            200, json={"answer_id": 7, "answer": "ok", "sources": [], "verification": {}}
        )

    _mount(handler, monkeypatch, token="tok")

    result = api_client.ask("what is force majeure?", "legal")

    assert captured["path"] == "/chat"
    assert b'"workspace":"legal"' in captured["body"]
    assert result["answer_id"] == 7


def test_upload_sends_multipart_and_provenance(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers.get("content-type", "")
        body = request.read()
        captured["body"] = body
        return httpx.Response(202, json={"id": 3, "filename": "a.pdf", "status": "processing"})

    _mount(handler, monkeypatch, token="tok")

    result = api_client.upload_document(
        b"%PDF-1.4 fake",
        "contract.pdf",
        "legal",
        doc_date="2024-01-01",
        jurisdiction="NY",
    )

    assert captured["path"] == "/documents"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b"contract.pdf" in captured["body"]
    assert b"2024-01-01" in captured["body"]
    assert result["status"] == "processing"


def test_delete_document_hits_document_path(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"deleted": 5})

    _mount(handler, monkeypatch, token="tok")

    result = api_client.delete_document(5)

    assert captured == {"method": "DELETE", "path": "/documents/5"}
    assert result == {"deleted": 5}


def test_demo_login_posts_demo_route(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200, json={"token": "new", "user_id": 1, "email": "demo@local"}
        )

    _mount(handler, monkeypatch, token=None)

    result = api_client.demo_login()

    assert captured["path"] == "/auth/demo"
    assert result["email"] == "demo@local"


def test_non_json_error_body_uses_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    _mount(handler, monkeypatch, token="tok")

    with pytest.raises(APIError) as exc:
        api_client.usage()

    assert exc.value.status == 500
    assert "Internal Server Error" in exc.value.detail


# --- Session helpers (guest + demo) ---


class _FakeSession(dict):
    """SessionState stand-in: dict API plus attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _mount_session(monkeypatch) -> _FakeSession:
    session = _FakeSession()
    monkeypatch.setattr(api_client, "st", type("NS", (), {"session_state": session})())
    return session


def test_anonymous_login_posts_device_id(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        return httpx.Response(
            200, json={"token": "g", "user_id": 4, "email": "Guest", "tier": "anonymous"}
        )

    _mount(handler, monkeypatch, token=None)

    body = api_client.anonymous_login("abcd1234")

    assert b'"device_id":"abcd1234"' in captured["body"]
    assert body["tier"] == "anonymous"


def test_ensure_guest_session_provisions_once(monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200, json={"token": "g", "user_id": 4, "email": "Guest", "tier": "anonymous"}
        )

    _mount(handler, monkeypatch, token=None)
    session = _mount_session(monkeypatch)

    api_client.ensure_guest_session()
    api_client.ensure_guest_session()

    assert calls == ["/auth/anonymous"]
    assert session["jwt"] == "g"
    assert session["anon_tier"] is True
    assert session["user_email"] == "Guest"
    assert len(session["device_id"]) == 32


def test_ensure_guest_session_keeps_existing_jwt(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no HTTP expected")

    _mount(handler, monkeypatch, token=None)
    session = _mount_session(monkeypatch)
    session["jwt"] = "member-token"

    api_client.ensure_guest_session()

    assert session["jwt"] == "member-token"


def test_complete_demo_login_sets_session(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/auth/demo"
        return httpx.Response(
            200, json={"token": "d", "user_id": 1, "email": "demo@local"}
        )

    _mount(handler, monkeypatch, token=None)
    session = _mount_session(monkeypatch)
    session["anon_tier"] = True

    api_client.complete_demo_login()

    assert session["jwt"] == "d"
    assert session["user_email"] == "demo@local"
    assert session["anon_tier"] is False
    assert session["show_pricing_modal"] is True
