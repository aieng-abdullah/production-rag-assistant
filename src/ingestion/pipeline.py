"""
ingestion pipeline to run tha full process of tha ingestion prasing,chunking,embedding.
"""

from loguru import logger
from src.ingestion.parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.db.chroma_client import upsert_chunks


def ingest(pdf_path: str) -> dict:
    try:

        # pasing pdf
        pages = extract_pages(pdf_path)
        logger.info("Pdf extracting....")

    except FileNotFoundError:

        logger.info("Pdf loading is failed")
        raise FileNotFoundError(f"Error: PDF file not found at path: {pdf_path}")

    except Exception as e:
        logger.info(f"Error during PDF extraction: {e}")
        raise RuntimeError(f"Failed to extract pages: {e}")

    if not pages:
        raise ValueError("Extraction failed: No pages were extracted.")

    try:
        # chunking
        chunking = chunk_pages(pages)
        if len(chunking) == 0:
            raise ValueError("Not enough chunk found")
    except Exception as e:
        raise RuntimeError(f"Failed to chunk pages: {e}.")
    try:
        # vector embedding
        embedding = embed_chunks(chunking)
        logger.info("embedding is  done")

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise ValueError(f"Embedding failed: {e}")

    try:

        # saiving tha emabadding
        save = upsert_chunks(embedding)
        logger.info("succesfully save  embeeding chunks iin vector store")

    except Exception as e:
        raise RuntimeError("error while saivinf embed chunks")

    return {"pages": len(pages), "chunks": save}
