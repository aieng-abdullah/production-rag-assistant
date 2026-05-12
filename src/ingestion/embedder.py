"""Embedding generation using LangChain HuggingFaceEmbeddings.

Backward compatibility wrapper around EmbeddingModel class.
"""

from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import Config
from src.core.embedding_model import EmbeddingModel


# Lazy-loaded model instance (for backward compatibility)
_model_instance: Optional[EmbeddingModel] = None


def _get_model() -> HuggingFaceEmbeddings:
    """Get or initialize the embedding model (backward compatibility wrapper)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance.model


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
        chunks: List of chunk dictionaries.
        batch_size: Batch size hint (handled internally by model).

    Returns:
        List of chunk dictionaries with 'embedding' field added.
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
