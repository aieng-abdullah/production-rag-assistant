"""Reciprocal Rank Fusion (RRF) for merging search results."""

from typing import List, Tuple, Dict
from collections import defaultdict

from src.config import Config
from src.monitoring.logger import get_logger

logger = get_logger("hybrid_fusion")


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = None
) -> List[Tuple[str, float]]:
    """Fuse vector and BM25 results using Reciprocal Rank Fusion."""
    k = k or Config.RRF_K
    
    # Track fused scores
    rrf_scores = defaultdict(float)
    
    # Add vector rankings
    for rank, (chunk_id, _) in enumerate(vector_results):
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
    
    # Add BM25 rankings
    for rank, (chunk_id, _) in enumerate(bm25_results):
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.debug(f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 = {len(fused)} fused")
    return fused
