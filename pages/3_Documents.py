"""Documents page — upload, status, delete (docs/UI_DESIGN.md §5 — 3_Documents).

Single-process mode (PLAN PR-6.1): ingest runs in-process with a progress
bar; BM25 index invalidated after every upload/delete.
"""

from pathlib import Path

import streamlit as st
from loguru import logger

from menu import menu_with_redirect
from src.auth.dependencies import is_guest, get_current_tier
from src.config import Config
from src.services import RAGService, check_tier_document_quota, record_ingest_usage, TIER_MULTIPLIERS
from src.services.bm25_cache import invalidate
from ui_core import (
    apply_workspace_accent,
    init_session_state,
    load_css,
    lottie,
    page_config,
)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

rag_service = RAGService()

page_config("Documents — RAG Research Assistant")
init_session_state()
load_css()
apply_workspace_accent()
menu_with_redirect()


def _get_tenant_id() -> str:
    """Get tenant_id from current session."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return "default"
    return str(user_id)


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded PDF to data/raw/."""
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf(file_path: Path) -> None:
    """Ingest PDF into the active workspace, refresh the BM25 cache."""
    tenant_id = _get_tenant_id()
    workspace = st.session_state.workspace
    tier = get_current_tier()

    # Quota check
    if is_guest():
        if st.session_state.guest_docs >= 1:
            st.error("Guest limit: 1 document. Sign in to upload more.")
            return
        st.session_state.guest_docs += 1
    else:
        quota = check_tier_document_quota(int(tenant_id), tier)
        if not quota.allowed:
            st.error(
                f"Document limit reached ({quota.used}/{quota.limit}). "
                f"Upgrade to Pro for {TIER_MULTIPLIERS['pro']}x quota."
            )
            return
        record_ingest_usage(int(tenant_id))

    _i_l, _i_anim, _i_r = st.columns([1, 2, 1])
    with _i_anim:
        lottie("indexing", height=140)
    with st.spinner(f"Processing {file_path.name} into '{workspace}'..."):
        progress_bar = st.progress(0)
        try:
            result = rag_service.ingest(
                tenant_id, file_path, workspace=workspace
            )
            progress_bar.progress(75)
            invalidate(tenant_id)
            progress_bar.progress(100)
            # Chunks exist now; the raw PDF was only an ingest input (PR-3b).
            # Purge failure must not fail the page — chunks are already live.
            try:
                file_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Raw purge failed {file_path}: {exc}")

            if file_path.name not in st.session_state.ingested_docs:
                st.session_state.ingested_docs.append(file_path.name)

            st.session_state.success_flash = (
                f"Processed {result['pages']} pages, "
                f"{result['chunks']} chunks into '{workspace}'"
            )
            st.toast(
                f"{file_path.name} indexed",
                icon=":material/check_circle:",
            )
            logger.info(f"PDF processed: {file_path.name} workspace={workspace} tenant={tenant_id}")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.toast("Ingest failed", icon=":material/error:")
            logger.error(f"PDF processing failed: {e}")
            raise


def render_uploader() -> None:
    st.markdown("### Upload")

    tier = get_current_tier()
    tenant_id = _get_tenant_id()

    # Show quota info
    if is_guest():
        st.caption("Guest limit: 1 document")
    else:
        quota = check_tier_document_quota(int(tenant_id), tier)
        st.caption(f"Documents: {quota.used}/{quota.limit} ({tier})")
        if quota.remaining == 0:
            st.warning("Document limit reached. Upgrade to Pro for more.")

    uploaded_file = st.file_uploader(
        "Drag and drop a PDF",
        type=["pdf"],
        help="Upload a research paper or document to analyze",
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            try:
                file_path = save_uploaded_file(uploaded_file)
                process_pdf(file_path)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")


def render_documents() -> None:
    """List persisted documents with status + delete."""
    st.markdown("### Documents")
    tenant_id = _get_tenant_id()
    doc_ids = rag_service.list_documents(tenant_id)

    if not doc_ids:
        _e_left, _e_anim, _e_right = st.columns([1, 2, 1])
        with _e_anim:
            lottie("empty", height=170)
        st.info("No documents yet. Upload a PDF above.")
        return

    for doc_id in doc_ids:
        row = st.container(border=True)
        c1, c2, c3 = row.columns([4, 2, 1])
        c1.markdown(f"`{doc_id}`")
        c2.markdown(":material/check_circle: indexed")
        if c3.button(
            "Delete",
            key=f"del_{doc_id}",
            icon=":material/delete:",
        ):
            try:
                rag_service.delete_document(tenant_id, doc_id)
                # All workspace BM25 indexes must drop the deleted chunks.
                invalidate(tenant_id)
                st.toast(
                    f"{doc_id} deleted",
                    icon=":material/delete:",
                )
                st.session_state.success_flash = f"{doc_id} deleted."
                logger.info(f"Document deleted via UI: {doc_id} tenant={tenant_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
                logger.error(f"Delete failed doc={doc_id}: {e}")


def _render_success_flash() -> None:
    """One-shot banner: ingest done / document deleted (consumes the flag)."""
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