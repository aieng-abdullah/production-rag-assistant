"""Text chunking with recursive character splitter."""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import Config
from src.monitoring.logger import get_logger

logger = get_logger("chunker")


def chunk_pages(pages: List[Dict], doc_id: str, filename: str) -> List[Dict]:
    """Split pages into chunks with metadata preservation."""
    # Initialize splitter with config settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_index = 0
    
    for page in pages:
        # Split page text into chunks
        page_chunks = splitter.split_text(page["text"])
        
        for chunk_text in page_chunks:
            # Skip tiny chunks
            if len(chunk_text) < 50:
                continue
            
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "filename": filename,
                "page_num": page["page_num"],
                "chunk_index": chunk_index,
                "char_count": len(chunk_text)
            })
            chunk_index += 1
    
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks
