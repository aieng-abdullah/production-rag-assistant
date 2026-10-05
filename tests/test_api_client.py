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
