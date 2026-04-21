"""Embedding generation with local Sentence Transformers."""

from typing import List, Dict
from sentence_transformers import SentenceTransformer

from src.config import Config


# Lazy-loaded model instance
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(Config.EMBEDDING_MODEL, device="cpu")
    return _model


def embed_query(text: str) -> List[float]:
    """Generate embedding for a single query text.

    Args:
        text: Query string to embed

    Returns:
        Embedding vector as list of floats
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Generate embeddings for chunks in batches."""
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True)

        for j, embedding in enumerate(embeddings):
            chunks[i + j]["embedding"] = embedding.tolist()

    return chunks
