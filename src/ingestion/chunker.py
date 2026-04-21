"""
Document chunking with metadata preservation.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """Split pages into chunks while preserving metadata."""

    chunks = []
    global_chunk_index = 0

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_num", -1)
        doc_id = page.get("doc_id", "unknown_doc")
        filename = page.get("source", "unknown_source")

        # Skip empty pages
        if not text:
            continue

        page_chunks = _text_splitter.split_text(text)

        for chunk_index_in_page, chunk_text in enumerate(page_chunks):
            # Skip very small chunks
            if len(chunk_text) < 50:
                continue

            chunks.append(
                {
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{global_chunk_index}",
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_index": global_chunk_index,
                    "chunk_index_in_page": chunk_index_in_page,
                    "char_count": len(chunk_text),
                }
            )

            global_chunk_index += 1

    # Add total_chunks metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total_chunks

    return chunks
