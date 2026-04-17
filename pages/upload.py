"""Upload page - PDF ingestion with progress tracking."""

import streamlit as st
import asyncio
from pathlib import Path

from src.ingestion.pipeline import ingest
from src.monitoring.logger import get_logger

logger = get_logger("upload_page")


def render_upload():
    """Render the upload page UI."""
    st.header("📤 Upload Documents")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload research papers or documents to query"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_path = Path(f"data/temp/{uploaded_file.name}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File saved: {uploaded_file.name}")
        
        # Ingest button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                
                try:
                    # Run ingestion
                    result = asyncio.run(ingest(temp_path))
                    
                    # Show success
                    progress_bar.progress(100)
                    
                    st.success("✅ Document processed successfully!")
                    
                    # Results card
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks", result["chunk_count"])
                    col2.metric("Pages", result["page_count"])
                    col3.metric("Time", f"{result['embed_time_ms']}ms")
                    
                    # Document ID for reference
                    st.code(f"Doc ID: {result['doc_id']}", language=None)
                    
                    logger.info(f"Upload complete: {result}")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    logger.error(f"Upload failed: {e}")


if __name__ == "__main__":
    render_upload()
