"""Documents page — upload, status, delete (docs/UI_DESIGN.md §5 — 3_Documents).

Single-process mode (PLAN PR-6.1): ingest runs in-process with a progress
bar; BM25 index invalidated after every upload/delete.
"""

from pathlib import Path

import streamlit as st
from loguru import logger

from menu import menu_with_redirect
from src.services import DEFAULT_TENANT, RAGService
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


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded PDF to data/raw/."""
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf(file_path: Path) -> None:
    """Ingest PDF into the active workspace, refresh the BM25 cache."""
    workspace = st.session_state.workspace
    with st.spinner(f"Processing {file_path.name} into '{workspace}'..."):
        progress_bar = st.progress(0)
        try:
            result = rag_service.ingest(
                DEFAULT_TENANT, file_path, workspace=workspace
            )
            progress_bar.progress(75)
            invalidate(DEFAULT_TENANT)
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
            logger.info(f"PDF processed: {file_path.name} workspace={workspace}")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.toast("Ingest failed", icon=":material/error:")
            logger.error(f"PDF processing failed: {e}")
            raise


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
                file_path = save_uploaded_file(uploaded_file)
                process_pdf(file_path)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")


def render_documents() -> None:
    """List persisted documents with status + delete."""
    st.markdown("### Documents")
    doc_ids = rag_service.list_documents(DEFAULT_TENANT)

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
                rag_service.delete_document(DEFAULT_TENANT, doc_id)
                # All workspace BM25 indexes must drop the deleted chunks.
                invalidate(DEFAULT_TENANT)
                st.toast(
                    f"{doc_id} deleted",
                    icon=":material/delete:",
                )
                st.session_state.success_flash = f"{doc_id} deleted."
                logger.info(f"Document deleted via UI: {doc_id}")
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
