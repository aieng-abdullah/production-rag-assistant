"""Embedding generation using LangChain HuggingFaceEmbeddings."""

from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import Config


# Lazy-loaded model instance
_model: HuggingFaceEmbeddings | None = None


def _get_model() -> HuggingFaceEmbeddings:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")
    return _model


def embed_query(text: str) -> List[float]:
    """Generate embedding for a single query text.

    Args:
        text: The text to embed.

    Returns:
        List of float embedding values.
    """
    model = _get_model()
    embedding = model.embed_query(text)
    return embedding


def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Generate embeddings for chunks in batches.

    Args:
        chunks: List of chunk dictionaries with 'text' field.
        batch_size: Number of texts to embed at once (used for logging only
                   as HuggingFaceEmbeddings handles batching internally).

    Returns:
        Chunks with 'embedding' field added.
    """
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]

    # HuggingFaceEmbeddings handles batching internally
    logger.debug(f"Embedding {len(texts)} chunks (batch_size hint: {batch_size})")

    embeddings = model.embed_documents(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.debug(f"Successfully embedded {len(chunks)} chunks")
    return chunks
