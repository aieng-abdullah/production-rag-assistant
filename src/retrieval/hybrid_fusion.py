""" 
Description: 
        hybrid fushion method for combining multiple ranked lists of search results (BM25 + Vecto_Search)
        into a single, optimized ranking,  used to improve Retrieval-Augmented Generation (RAG) pipelines
"""
from src.retrieval.bm25_index import bm25_search , build_bm25_index
from src.retrieval.chroma_search import vector_search
from loguru import logger
from typing import List, Dict
from src.config import Config


def rrf_fusion(bm25_results: List[Dict],vector_results: List[Dict],top_k: int) -> List[Dict]:
    """
    Combine BM25 and vector search results using Reciprocal Rank Fusion (RRF).

    """

    scores: Dict[str, float] = {}
    all_chunks: Dict[str, Dict] = {}

    # Step 1 — BM25 scores
    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)
        all_chunks[chunk_id] = chunk

    # Step 2 — Vector search scores
    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)

        # Keep chunk if not already added (avoid overwrite issues)
        if chunk_id not in all_chunks:
            all_chunks[chunk_id] = chunk

    # Step 3 — Sort by fused score (descending)
    ranked_chunks = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Step 4 — Build final result list
    fused_results = []
    for chunk_id, score in ranked_chunks[:top_k]:
        chunk = all_chunks[chunk_id].copy()
        chunk["rrf_score"] = score  # optional but useful for debugging
        fused_results.append(chunk)

    logger.debug(f"RRF fused {len(fused_results)} results")

    return fused_results

