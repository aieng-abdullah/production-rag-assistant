"""
Ingestion pipeline: PDF parsing → chunking → embedding → storage.
"""

from loguru import logger

from src.db.chroma_client import upsert_chunks
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.ingestion.parser import extract_pages


def ingest(pdf_path: str) -> dict:
    """Process a PDF file through the full ingestion pipeline.

    Steps:
        1. Extract text from PDF pages
        2. Split pages into chunks
        3. Generate embeddings for chunks
        4. Store chunks in vector database

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with page count and chunk count.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        ValueError: If extraction or chunking produces no results.
        RuntimeError: If any pipeline step fails.
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
        logger.info(f"Created {len(chunks)} chunks")
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

    # Step 4: Store in vector database
    try:
        chunk_count = upsert_chunks(embedded_chunks)
        logger.info(f"Stored {chunk_count} chunks in vector store")
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        raise RuntimeError(f"Failed to store chunks: {e}")

    return {"pages": len(pages), "chunks": chunk_count}
