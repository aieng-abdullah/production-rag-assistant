"""Documents page — upload, status, delete (docs/UI_DESIGN.md §5 — 3_Documents).

PR-6: multipart upload → `POST /documents` (202), status polled through
`GET /documents/{id}` until the background ingest flips to ready/failed,
delete via `DELETE /documents/{id}`. Guests provision their per-device
session on first upload — the tier allows one free document.
"""

import time

import streamlit as st
from loguru import logger

import api_client
from api_client import APIError
from menu import menu_with_redirect, show_login_wall
from ui_core import (
    apply_workspace_accent,
    init_session_state,
    load_css,
    lottie,
    page_config,
)

_POLL_INTERVAL_SECONDS = 2.0
_POLL_DEADLINE_SECONDS = 300.0

page_config("Documents — RAG Research Assistant")
init_session_state()
load_css()
apply_workspace_accent()
menu_with_redirect()


def _upload(uploaded_file) -> dict:
    """Send the PDF bytes to the API; provisions the guest session first."""
    if not st.session_state.get("jwt"):
        api_client.ensure_guest_session()
    workspace = st.session_state.workspace
    payload = uploaded_file.getvalue()
    return api_client.upload_document(payload, uploaded_file.name, workspace)


def _await_ingest(document_id: int, filename: str) -> str:
    """Poll the API until the background ingest finishes; returns status."""
    deadline = time.monotonic() + _POLL_DEADLINE_SECONDS
    status = "processing"
    while time.monotonic() < deadline:
        doc = api_client.get_document(document_id)
        status = doc.get("status", "processing")
        if status in ("ready", "failed"):
            return status
        time.sleep(_POLL_INTERVAL_SECONDS)
    logger.warning(f"Ingest poll timed out doc={filename} id={document_id}")
    return status


def render_uploader() -> None:
    st.markdown("### Upload")
    uploaded_file = st.file_uploader(
        "Drag and drop a PDF",
        type=["pdf"],
        help="Upload a research paper or document to analyze",
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            try:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    result = _upload(uploaded_file)
                with st.spinner(
                    f"Processing {result['filename']} into "
                    f"'{st.session_state.workspace}'..."
                ):
                    status = _await_ingest(result["id"], result["filename"])
            except APIError as exc:
                logger.error(f"Upload failed: status={exc.status} {exc.detail}")
                if exc.status == 429 and st.session_state.get("anon_tier"):
                    st.error(exc.detail)
                    show_login_wall()
                else:
                    st.error(f"Upload failed: {exc.detail}")
                return

            if status == "ready":
                st.toast(f"{result['filename']} indexed", icon=":material/check_circle:")
                logger.info(f"PDF processed: {result['filename']}")
                st.session_state.success_flash = (
                    f"{result['filename']} indexed and ready."
                )
                st.rerun()
            elif status == "failed":
                st.error(f"Ingest failed for {result['filename']} — check the API logs.")
                st.toast("Ingest failed", icon=":material/error:")
            else:
                st.warning(
                    "Still processing — refresh in a moment to see the status."
                )


_STATUS_ICONS = {
    "ready": (":material/check_circle:", "indexed"),
    "processing": (":material/hourglass_top:", "processing"),
    "failed": (":material/error:", "failed"),
}


def render_documents() -> None:
    """List documents with status + delete (API-backed)."""
    st.markdown("### Documents")
    try:
        docs = api_client.list_documents()
    except APIError as exc:
        if exc.status == 401:
            st.info("Sign in to see your documents.")
        else:
            logger.warning(f"Document list unavailable: {exc.detail}")
            st.warning(f"Document list unavailable: {exc.detail}")
        return

    if not docs:
        _e_left, _e_anim, _e_right = st.columns([1, 2, 1])
        with _e_anim:
            lottie("empty", height=170)
        st.info("No documents yet. Upload a PDF above.")
        return

    for doc in docs:
        document_id = doc["id"]
        filename = doc["filename"]
        status = doc.get("status", "ready")
        icon, label = _STATUS_ICONS.get(status, (":material/help:", status))
        row = st.container(border=True)
        c1, c2, c3 = row.columns([4, 2, 1])
        c1.markdown(f"`{filename}`")
        c2.markdown(f"{icon} {label}")
        if c3.button(
            "Delete",
            key=f"del_{document_id}",
            icon=":material/delete:",
        ):
            try:
                api_client.delete_document(document_id)
                st.toast(f"{filename} deleted", icon=":material/delete:")
                logger.info(f"Document deleted via UI: {filename}")
                st.session_state.success_flash = f"{filename} deleted."
                st.rerun()
            except APIError as exc:
                st.error(f"Delete failed: {exc.detail}")
                logger.error(f"Delete failed doc={filename}: {exc.detail}")


def _render_success_flash() -> None:
    """One-shot banner: ingest ready / document deleted (consumes the flag)."""
    message = st.session_state.pop("success_flash", None)
    if not message:
        return
    _s_left, _s_anim, _s_right = st.columns([1, 2, 1])
    with _s_anim:
        lottie("success", height=130, loop=False)
    st.success(message)


def main() -> None:
    _render_success_flash()
    render_uploader()
    st.divider()
    render_documents()


if __name__ == "__main__":
    main()
