"""HTTP client for the FastAPI backend (PLAN PR-6).

Streamlit pages talk to the API through this module instead of importing
`src.*` service code — one process boundary, real auth, real quotas.
The session JWT lives in `st.session_state["jwt"]`; a 401 clears it so the
next render shows the login wall, a 429 surfaces the server's quota detail
as `APIError.status`. `_transport` / `_token` / `_session_clear` are the
test seams (httpx.MockTransport + monkeypatch).
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st
from loguru import logger

from src.config import Config

__all__ = [
    "APIError",
    "anonymous_login",
    "ask",
    "complete_demo_login",
    "delete_document",
    "demo_login",
    "ensure_guest_session",
    "get_document",
    "list_documents",
    "usage",
    "upload_document",
]

_TIMEOUT = 60.0  # cross-encoder + Groq latency headroom (PR-1 measured ~14s)


class APIError(Exception):
    """Backend returned non-2xx, or is unreachable (`status == 0`)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"API {status}: {detail}")
        self.status = status
        self.detail = detail


# Test seam: swap for httpx.MockTransport(handler).
_transport: httpx.BaseTransport | None = None


def _token() -> str | None:
    return st.session_state.get("jwt")


def _session_clear() -> None:
    for key in ("jwt", "user_id", "user_email", "avatar", "show_pricing_modal", "anon_tier"):
        st.session_state.pop(key, None)


def _request(method: str, path: str, **kwargs: Any) -> Any:
    headers = {}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(
            base_url=Config.API_BASE_URL,
            transport=_transport,
            timeout=_TIMEOUT,
        ) as client:
            response = client.request(method, path, headers=headers, **kwargs)
    except httpx.HTTPError as exc:
        logger.error(f"API unreachable method={method} path={path}: {exc}")
        raise APIError(
            0,
            "Backend unreachable — start it with "
            "`uvicorn src.api.app:app --port 8001`.",
        ) from exc

    if response.status_code == 401:
        _session_clear()
        raise APIError(401, "Session expired — please sign in again.")
    if response.status_code >= 400:
        try:
            body: Any = response.json()
            detail = str(body.get("detail", body) if isinstance(body, dict) else body)
        except ValueError:
            detail = response.text or f"HTTP {response.status_code}"
        logger.warning(f"API error status={response.status_code} path={path}: {detail}")
        raise APIError(response.status_code, detail)

    if not response.content:
        return None
    return response.json()


def demo_login() -> dict:
    """POST /auth/demo — issue a token for the local demo account."""
    return _request("POST", "/auth/demo")


def anonymous_login(device_id: str) -> dict:
    """POST /auth/anonymous — per-device guest session (free tier)."""
    return _request("POST", "/auth/anonymous", json={"device_id": device_id})


def complete_demo_login() -> None:
    """Demo sign-in + session keys — shared by login page and login wall.

    Raises `APIError` on failure; callers surface `detail` to the UI.
    """
    body = demo_login()
    st.session_state.jwt = body["token"]
    st.session_state.user_id = str(body["user_id"])
    st.session_state.user_email = body["email"]
    st.session_state.anon_tier = False
    st.session_state.show_pricing_modal = True


def ensure_guest_session() -> None:
    """Provision the per-device guest session on first use (free tier).

    Device id is generated once per browser session and reused, so guest
    quotas follow the device across Streamlit reruns. Raises `APIError`
    when the backend is down or rejects the id.
    """
    if st.session_state.get("jwt"):
        return
    device_id = st.session_state.get("device_id")
    if not device_id:
        device_id = uuid.uuid4().hex
        st.session_state.device_id = device_id
    body = anonymous_login(device_id)
    st.session_state.jwt = body["token"]
    st.session_state.user_id = str(body["user_id"])
    st.session_state.user_email = "Guest"
    st.session_state.anon_tier = True
    st.session_state.anon_queries = 0
    logger.info("Guest session provisioned user_id={uid}", uid=body["user_id"])


def ask(query: str, workspace: str) -> dict:
    """POST /chat — retrieval + generation. 429 = daily quota exhausted."""
    return _request("POST", "/chat", json={"query": query, "workspace": workspace})


def list_documents() -> list[dict]:
    """GET /documents — newest first, status per document."""
    return _request("GET", "/documents")


def get_document(document_id: int) -> dict:
    """GET /documents/{id} — poll until status is ready or failed."""
    return _request("GET", f"/documents/{document_id}")


def upload_document(
    payload: bytes,
    filename: str,
    workspace: str,
    doc_date: str | None = None,
    doc_version: str | None = None,
    jurisdiction: str | None = None,
) -> dict:
    """POST /documents — multipart upload; 202 with `processing` status."""
    data: dict[str, str] = {"workspace": workspace}
    if doc_date:
        data["doc_date"] = doc_date
    if doc_version:
        data["doc_version"] = doc_version
    if jurisdiction:
        data["jurisdiction"] = jurisdiction
    return _request(
        "POST",
        "/documents",
        files={"file": (filename, payload, "application/pdf")},
        data=data,
    )


def delete_document(document_id: int) -> dict:
    """DELETE /documents/{id} — chunks + file + row, tenant-scoped."""
    return _request("DELETE", f"/documents/{document_id}")


def usage() -> dict:
    """GET /usage — quota meters for sidebar and dashboard."""
    return _request("GET", "/usage")
