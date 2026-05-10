"""Streamlit UI for RAG Research Assistant.

Implements the specification from streamlit_spec.md:
- Left sidebar: app title, PDF uploader, document list
- Main area: chat history, citations as expanders, chat input
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from loguru import logger

from src.config import Config
from src.db.chroma_client import load_all_chunks
from src.generation.chain import generate
from src.ingestion.pipeline import ingest
from src.retrieval.bm25_index import build_bm25_index

# Constants
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded PDF to data/raw/ directory."""
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf(file_path: Path):
    """Process PDF through ingestion pipeline."""
    with st.spinner(f"Processing {file_path.name}..."):
        progress_bar = st.progress(0)

        try:
            # Step 1: Ingest
            result = ingest(str(file_path))
            progress_bar.progress(50)

            # Step 2: Reload chunks
            st.session_state.chunks = load_all_chunks()
            progress_bar.progress(75)

            # Step 3: Rebuild BM25 index
            if st.session_state.chunks:
                st.session_state.bm25_index = build_bm25_index(st.session_state.chunks)
            progress_bar.progress(100)

            # Add to ingested docs
            if file_path.name not in st.session_state.ingested_docs:
                st.session_state.ingested_docs.append(file_path.name)

            st.success(f"✅ Processed {result['pages']} pages, {result['chunks']} chunks")
            logger.info(f"PDF processed: {file_path.name}")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            logger.error(f"PDF processing failed: {e}")
            raise


def display_cited_answer(cited_answer):
    """Display answer with expandable citations."""
    # Display the answer text
    st.markdown(cited_answer.answer)

    # Show sources as expanders
    if cited_answer.sources:
        st.markdown("---")
        st.markdown("**Sources:**")

        for i, source in enumerate(cited_answer.sources, 1):
            with st.expander(f"[{i}] {source.doc_id} - Page {source.page_num}"):
                st.markdown(f"**Document:** `{source.doc_id}`")
                st.markdown(f"**Page:** {source.page_num}")
                st.markdown(f"**Text:**")
                st.text(source.text[:500] + "..." if len(source.text) > 500 else source.text)


def handle_query(query: str):
    """Handle user query and generate response."""
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload a PDF first!")
        return

    if st.session_state.bm25_index is None:
        st.warning("⚠️ BM25 index not ready. Please process a PDF first.")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Generate answer
                cited_answer = generate(
                    query,
                    st.session_state.chunks,
                    st.session_state.bm25_index,
                )

                # Display answer with citations
                display_cited_answer(cited_answer)

                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cited_answer.answer,
                    "sources": [
                        {
                            "doc_id": s.doc_id,
                            "page_num": s.page_num,
                            "text": s.text,
                        }
                        for s in cited_answer.sources
                    ],
                })

            except Exception as e:
                error_msg = f"❌ Error generating answer: {e}"
                st.error(error_msg)
                logger.error(f"Generation failed: {e}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error while generating the answer. Please try again.",
                })


def render_sidebar():
    """Render left sidebar with upload and document list."""
    st.sidebar.title("📚 RAG Research Assistant")
    st.sidebar.markdown("---")

    # PDF Uploader
    st.sidebar.markdown("### Upload PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Drag and drop a PDF",
        type=["pdf"],
        help="Upload a research paper or document to analyze",
    )

    if uploaded_file is not None:
        if st.sidebar.button("Process Document", type="primary"):
            try:
                file_path = save_uploaded_file(uploaded_file)
                process_pdf(file_path)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

    st.sidebar.markdown("---")

    # Document List
    st.sidebar.markdown("### Uploaded Documents")
    if st.session_state.ingested_docs:
        for doc in st.session_state.ingested_docs:
            st.sidebar.markdown(f"- ✅ {doc}")
    else:
        st.sidebar.info("No documents uploaded yet")

    # Stats
    if st.session_state.chunks:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Stats")
        st.sidebar.text(f"Total chunks: {len(st.session_state.chunks)}")
        if st.session_state.bm25_index:
            st.sidebar.text("BM25 index: ✅ Ready")


def render_chat():
    """Render main chat area."""
    st.markdown("### Chat")

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show citations for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                st.markdown("---")
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    with st.expander(f"[{i}] {source['doc_id']} - Page {source['page_num']}"):
                        st.markdown(f"**Document:** `{source['doc_id']}`")
                        st.markdown(f"**Page:** {source['page_num']}")
                        st.markdown(f"**Text:**")
                        st.text(source["text"][:500] + "..." if len(source["text"]) > 500 else source["text"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_query(prompt)


def main():
    """Main app entry point."""
    init_session_state()

    # Validate config
    try:
        Config.validate()
    except EnvironmentError as e:
        st.error(f"⚠️ Configuration Error: {e}")
        st.stop()

    # Render UI
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
