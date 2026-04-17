"""Chat page - Q&A with streaming and citations."""

import streamlit as st
from src.retrieval.pipeline import hybrid_search
from src.generation.chain import generate
from src.monitoring.logger import get_logger

logger = get_logger("chat_page")


def render_chat():
    """Render the chat page UI."""
    st.header("💬 Ask Questions")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Document selector
    st.sidebar.subheader("Document Filter")
    doc_filter = st.sidebar.text_input(
        "Filter by Doc ID (optional)",
        placeholder="Leave empty to search all"
    )
    
    # Debug mode
    show_debug = st.sidebar.checkbox("Show retrieval debug", value=False)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show citations if assistant message
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f"**[{i}]** {src['filename']} - Page {src['page_num']}")
                        st.caption(src['text'][:200] + "...")
    
    # Chat input
    if question := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Retrieve chunks
                chunks, debug = hybrid_search(
                    question,
                    doc_id=doc_filter if doc_filter else None,
                    include_debug=show_debug
                )
                
                if not chunks:
                    st.warning("No relevant documents found. Try uploading a document first.")
                    return
                
                st.info(f"Retrieved {len(chunks)} relevant chunks")
            
            with st.spinner("Generating answer..."):
                # Generate answer
                result = generate(question, chunks)
                
                # Display answer
                st.markdown(result["answer"])
                
                # Sources expander
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(result["sources"]):
                            st.markdown(f"**[{i}]** {src['filename']} - Page {src['page_num']}")
                            st.caption(src['text'][:200] + "...")
                            if "rerank_score" in src:
                                st.caption(f"Rerank score: {src['rerank_score']:.3f}")
                
                # Debug expander
                if show_debug and debug:
                    with st.expander("🔍 Debug Info"):
                        st.json(debug)
                
                # Usage info
                if result.get("usage"):
                    st.caption(f"Tokens: {result['usage'].get('total_tokens', 'N/A')}")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", [])
        })
        
        logger.info(f"Chat exchange: Q='{question[:50]}...'")


if __name__ == "__main__":
    render_chat()
