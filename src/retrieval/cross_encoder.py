"""Cross-encoder reranking for result refinement.

Backward compatibility wrapper around CrossEncoderReranker class.
"""

from typing import Optional

from sentence_transformers import CrossEncoder
from loguru import logger
from src.core.cross_encoder import CrossEncoderReranker

# Lazy-loaded model instance (for backward compatibility)
_reranker_instance: Optional[CrossEncoderReranker] = None


def _get_model() -> CrossEncoder:
    """Get or initialize the cross-encoder model (backward compatibility wrapper)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance.model


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Rerank chunks against a query using a cross-encoder model.
    
    Args:
        query: The query string.
        chunks: List of chunk dictionaries to rerank.
        top_k: Number of top results to return.
        score_threshold: Optional minimum score threshold.
                       Chunks below this threshold are filtered out.
        
    Returns:
        List of reranked chunk dictionaries with 'rerank_score' added.
        
    Raises:
        ValueError: If top_k is not a positive integer.
    """
    reranker = CrossEncoderReranker()
    return reranker.rerank(query, chunks, top_k, score_threshold)