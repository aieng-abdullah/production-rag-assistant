"""Documents page — upload, status, delete (docs/UI_DESIGN.md §5 — 3_Documents).

Interim: direct service calls; httpx + JWT in PLAN PR-6.
"""

from pathlib import Path

import streamlit as st
from loguru import logger

from menu import menu_with_redirect
from src.db.chroma_client import load_all_chunks
from src.retrieval.bm25_index import build_bm25_index
from src.services import DEFAULT_TENANT, RAGService
from ui_core import apply_workspace_accent, init_session_state, load_css, page_config

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
    """Ingest PDF, rebuild BM25, announce completion with a toast."""
    with st.spinner(f"Processing {file_path.name}..."):
        progress_bar = st.progress(0)
        try:
            result = rag_service.ingest(DEFAULT_TENANT, file_path)
            progress_bar.progress(50)

            chunks = load_all_chunks()
            progress_bar.progress(75)

            if chunks:
                st.session_state.bm25_index = build_bm25_index(chunks)
            progress_bar.progress(100)

            if file_path.name not in st.session_state.ingested_docs:
                st.session_state.ingested_docs.append(file_path.name)

            st.success(
                f"Processed {result['pages']} pages, "
                f"{result['chunks']} chunks"
            )
            st.toast(
                f"{file_path.name} indexed",
                icon=":material/check_circle:",
            )
            logger.info(f"PDF processed: {file_path.name}")

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
                # BM25 must drop the deleted chunks too.
                chunks = load_all_chunks()
                st.session_state.bm25_index = (
                    build_bm25_index(chunks) if chunks else None
                )
                st.toast(
                    f"{doc_id} deleted",
                    icon=":material/delete:",
                )
                logger.info(f"Document deleted via UI: {doc_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
                logger.error(f"Delete failed doc={doc_id}: {e}")


def main() -> None:
    render_uploader()
    st.divider()
    render_documents()


if __name__ == "__main__":
    main()
