"""Streamlit UI for RAG Research Assistant.

Implements the specification from streamlit_spec.md:
- Left sidebar: app title, PDF uploader, document list
- Main area: chat history, citations as expanders, chat input

Uses core classes for better architecture.
"""

import os
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from loguru import logger

from src.config import Config
from src.core.vector_store import VectorStore
from src.core.bm25_retriever import BM25Retriever
from src.core.cross_encoder import CrossEncoderReranker
from src.core.retrieval_pipeline import RetrievalPipeline
from src.core.rag_generator import RAGGenerator
from src.ingestion.pipeline import ingest

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
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "bm25_retriever" not in st.session_state:
        st.session_state.bm25_retriever = None
    if "retrieval_pipeline" not in st.session_state:
        st.session_state.retrieval_pipeline = None
    if "rag_generator" not in st.session_state:
        st.session_state.rag_generator = None
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded PDF to data/raw/ directory."""
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf(file_path: Path):
    """Process PDF through ingestion pipeline using core classes."""
    with st.spinner(f"Processing {file_path.name}..."):
        progress_bar = st.progress(0)

        try:
            # Step 1: Ingest
            result = ingest(str(file_path))
            progress_bar.progress(50)

            # Step 2: Initialize core classes
            st.session_state.vector_store = VectorStore()
            st.session_state.bm25_retriever = BM25Retriever()
            st.session_state.cross_encoder = CrossEncoderReranker()
            progress_bar.progress(75)

            # Step 3: Load chunks
            st.session_state.chunks = st.session_state.vector_store.load_all()
            
            # Step 4: Build BM25 index
            if st.session_state.chunks:
                st.session_state.bm25_retriever.build_index(st.session_state.chunks)
            
            # Step 5: Create retrieval pipeline
            st.session_state.retrieval_pipeline = RetrievalPipeline(
                vector_store=st.session_state.vector_store,
                bm25_retriever=st.session_state.bm25_retriever,
                cross_encoder=st.session_state.cross_encoder,
                top_k=5,
            )
            
            # Step 6: Create RAG generator
            llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL)
            st.session_state.rag_generator = RAGGenerator(
                llm=llm,
                retrieval_pipeline=st.session_state.retrieval_pipeline,
            )
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
    """Handle user query and generate response using core classes."""
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload a PDF first!")
        return

    if st.session_state.rag_generator is None:
        st.warning("⚠️ RAG generator not ready. Please process a PDF first.")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Generate answer using RAGGenerator
                cited_answer = st.session_state.rag_generator.generate(
                    query,
                    st.session_state.chunks,
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
        if st.session_state.rag_generator:
            st.sidebar.text("RAG Pipeline: ✅ Ready")


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
