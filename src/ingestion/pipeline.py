"""
Ingestion pipeline: PDF parsing → chunking → embedding → storage.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import Config
from src.db.chroma_client import DEFAULT_TENANT, upsert_chunks
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.ingestion.parser import extract_pages


def _document_provenance(
    pdf_path: str, provenance: dict | None
) -> dict:
    """Doc-level metadata stamped on every chunk (PLAN PR-4b-iv):
    file hash + ingest time (always), user-supplied date/version/
    jurisdiction (optional upload inputs).

    The hash guard only fires when the file is missing — step 1
    (`extract_pages`) already hard-fails on unreadable files in production,
    so the guard covers extraction-stubbed callers, never a silent prod gap."""
    path = Path(pdf_path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        doc_hash = digest.hexdigest()
    else:
        doc_hash = ""
    supplied = provenance or {}
    return {
        "doc_hash": doc_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "doc_date": str(supplied.get("date") or ""),
        "doc_version": str(supplied.get("version") or ""),
        "jurisdiction": str(supplied.get("jurisdiction") or ""),
    }


def ingest(
    pdf_path: str,
    tenant_id: str = DEFAULT_TENANT,
    workspace: str = Config.DEFAULT_WORKSPACE,
    provenance: dict | None = None,
) -> dict:
    """Process a PDF file through the full ingestion pipeline.

    Steps:
        1. Extract text from PDF pages
        2. Split pages into chunks
        3. Generate embeddings for chunks
        4. Store chunks in vector database

    `workspace` is stamped on every chunk metadata (legal | academic) so
    retrieval can filter by niche (PLAN PR-4).
    """
    # Step 1: Extract pages from PDF
    try:
        pages = extract_pages(pdf_path)
        logger.info(f"Extracted {len(pages)} pages from PDF")
    except FileNotFoundError:
        logger.error(f"PDF not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise RuntimeError(f"Failed to extract pages: {e}")

    if not pages:
        raise ValueError("No pages extracted from PDF")

    # Step 2: Chunk pages
    try:
        chunks = chunk_pages(pages)
        doc_meta = _document_provenance(pdf_path, provenance)
        for chunk in chunks:
            chunk["workspace"] = workspace
            chunk.update(doc_meta)
        logger.info(
            f"Created {len(chunks)} chunks (workspace={workspace}, "
            f"doc_hash={doc_meta['doc_hash'][:12]})"
        )
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise RuntimeError(f"Failed to chunk pages: {e}")

    if not chunks:
        raise ValueError("No chunks created from pages")

    # Step 3: Generate embeddings
    try:
        embedded_chunks = embed_chunks(chunks)
        logger.info("Embeddings generated")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {e}")

    # Step 4: Store in vector database (tenant-partitioned)
    try:
        chunk_count = upsert_chunks(embedded_chunks, tenant_id=tenant_id)
        logger.info(f"Stored {chunk_count} chunks in vector store")
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        raise RuntimeError(f"Failed to store chunks: {e}")

    return {"pages": len(pages), "chunks": chunk_count}
