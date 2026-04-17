"""Streamlit main app - RAG Research Assistant."""

import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📚 RAG Research Assistant")
st.markdown("Upload PDFs and ask questions with cited answers.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📤 Upload", "💬 Chat", "ℹ️ About"]
)

# Route to pages
if page == "🏠 Home":
    st.markdown("""
    ## Welcome!
    
    This app demonstrates production-grade RAG with:
    - **Hybrid Retrieval**: BM25 + Vector search with RRF fusion
    - **Cross-Encoder Reranking**: ms-marco MiniLM for precision
    - **Citation Enforcement**: Every claim validated against sources
    - **CI-Gated Eval**: GitHub Actions runs Ragas on every PR
    
    ### Get Started
    1. Go to **Upload** to add PDF documents
    2. Go to **Chat** to ask questions
    
    ### Links
    - [LangSmith Traces](https://smith.langchain.com)
    - [GitHub Repo](https://github.com)
    """)

elif page == "📤 Upload":
    # Import and run upload page
    from pages.upload import render_upload
    render_upload()

elif page == "💬 Chat":
    # Import and run chat page
    from pages.chat import render_chat
    render_chat()

elif page == "ℹ️ About":
    from pages.about import render_about
    render_about()
