"""Cross-encoder reranking for second-pass scoring."""

from typing import List, Dict
from sentence_transformers import CrossEncoder

from src.config import Config
from src.monitoring.logger import get_logger

logger = get_logger("cross_encoder")

# Load cross-encoder model (lazy initialization)
_model = None


def get_model():
    """Get or initialize cross-encoder model."""
    global _model
    if _model is None:
        _model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Loaded cross-encoder model")
    return _model


def rerank(query: str, chunks: List[Dict], top_n: int = None) -> List[Dict]:
    """Rerank chunks using cross-encoder."""
    top_n = top_n or Config.TOP_K_RERANK
    
    if not chunks:
        return []
    
    # Prepare pairs for cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Get scores
    model = get_model()
    scores = model.predict(pairs)
    
    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    # Sort by rerank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    logger.info(f"Reranked {len(chunks)} chunks, top score: {reranked[0]['rerank_score']:.3f}")
    return reranked[:top_n]
