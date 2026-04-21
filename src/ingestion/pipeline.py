"""
ingestion pipeline to run tha full process of tha ingestion prasing,chunking,embedding.
"""

from typing import Optional
from src.ingestion.parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.db.chroma_client import get_collection, reset_client, upsert_chunks


def ingest(pdf_path: str) -> dict:
    pages = extract_pages(pdf_path)
    chunking = chunk_pages(pages)
    embedding = embed_chunks(chunking)
    save = upsert_chunks(embedding)

    return {"pages": len(pages), "chunks": save}
