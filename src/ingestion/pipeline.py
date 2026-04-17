"""End-to-end ingestion pipeline orchestrator."""

import uuid
import time
from pathlib import Path
from typing import Dict

from src.db.chroma_client import get_chroma_client, get_or_create_collection
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.monitoring.logger import get_logger

logger = get_logger("ingestion")


async def ingest(file_path: str | Path) -> Dict:
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    file_path = Path(file_path)
    start_time = time.time()
    
    # Generate document ID
    doc_id = str(uuid.uuid4())
    logger.info(f"Starting ingestion: {file_path.name}, doc_id: {doc_id}")
    
    # Step 1: Parse PDF
    pages = parse_pdf(file_path)
    
    # Step 2: Chunk pages
    chunks = chunk_pages(pages, doc_id, file_path.name)
    chunk_count = len(chunks)
    
    if chunk_count == 0:
        raise ValueError("No valid chunks extracted from PDF")
    
    # Step 3: Generate embeddings
    chunks = await embed_chunks(chunks)
    
    # Step 4: Upsert to ChromaDB
    collection = get_or_create_collection(get_chroma_client())
    
    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(chunk_count)],
        documents=[c["text"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[{
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "char_count": c["char_count"]
        } for c in chunks]
    )
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    result = {
        "doc_id": doc_id,
        "filename": file_path.name,
        "chunk_count": chunk_count,
        "embed_time_ms": elapsed_ms,
        "page_count": len(pages)
    }
    
    logger.info(f"Ingestion complete: {result}")
    return result
