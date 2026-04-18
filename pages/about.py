"""About page - project info and architecture."""

import streamlit as st


def render_about():
    """Render the about page UI."""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## RAG Research Assistant
    
    A production-grade Retrieval-Augmented Generation system for research documents.
    
    ### Architecture
    
    ```
    PDF → PyMuPDF → Chunker → Sentence Transformers → ChromaDB
                                              ↓
    Query → BM25 Search ─┬─→ RRF Fusion → Cross-Encoder Rerank → Groq LLM → Cited Answer
             Vector Search ─┘
    ```
    
    ### Key Features
    
    1. **Hybrid Retrieval (BM25 + Vector)**
       - BM25 catches exact keyword matches
       - Vector search captures semantic similarity
       - RRF (Reciprocal Rank Fusion) combines both
    
    2. **Cross-Encoder Reranking**
       - ms-marco-MiniLM scores query-chunk relevance
       - Significantly improves precision over first-stage retrieval
    
    3. **Citation Enforcement**
       - LLM must cite [SOURCE N] for every claim
       - Validation ensures citations reference valid chunks
       - Hallucination-resistant design
    
    4. **CI-Gated Evaluation**
       - GitHub Actions runs Ragas on every PR
       - Fails build if faithfulness < 0.80
    
    ### Tech Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python + FastAPI (optional)
    - **Vector DB**: ChromaDB
    - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local, free)
    - **LLM**: Groq GPT-OSS-120B / Llama-3.3-70B (fast, free tier)
    - **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
    - **Evaluation**: Ragas + Pytest
    - **Observability**: Langfuse + Loguru
    
    ### Links
    
    - [GitHub Repository](https://github.com/yourname/rag-research-assistant)
    - [LangSmith Project](https://smith.langchain.com)
    - [Live Demo](https://yourapp.streamlit.app)
    
    ### License
    
    MIT License - See LICENSE file
    """)


if __name__ == "__main__":
    render_about()
