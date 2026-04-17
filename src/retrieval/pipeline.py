"""Full retrieval pipeline: hybrid search → reranking."""

import time
from typing import List, Dict, Optional

from src.config import Config
from src.db.chroma_client import get_chroma_client, get_or_create_collection
from src.retrieval.chroma_search import vector_search
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.cross_encoder import rerank
from src.monitoring.logger import get_logger
from src.monitoring.rag_metrics import metrics_collector

logger = get_logger("retrieval_pipeline")


def fetch_chunks(chunk_ids: List[str]) -> List[Dict]:
    """Fetch full chunk data from ChromaDB by IDs."""
    collection = get_or_create_collection(get_chroma_client())
    results = collection.get(ids=chunk_ids)
    
    chunks = []
    for i, chunk_id in enumerate(results["ids"]):
        chunks.append({
            "id": chunk_id,
            "text": results["documents"][i],
            **results["metadatas"][i]
        })
    
    return chunks


def hybrid_search(
    query: str,
    top_k: int = None,
    doc_id: str = None,
    include_debug: bool = False,
    collect_metrics: bool = True
) -> tuple:
    """Full hybrid retrieval: BM25 + Vector + RRF + Cross-encoder rerank."""
    top_k = top_k or Config.TOP_K_RERANK
    start_time = time.time()
    
    logger.info(f"Hybrid search: '{query[:50]}...', doc_filter: {doc_id}")
    
    # Step 1: Vector search
    vector_results = vector_search(
        query, 
        top_k=Config.TOP_K_VECTOR, 
        doc_id=doc_id
    )
    
    # Step 2: BM25 search
    bm25_index = BM25Index()
    bm25_index.build(doc_id=doc_id)
    bm25_results = bm25_index.search(query, top_k=Config.TOP_K_BM25)
    
    # Step 3: RRF fusion
    fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
    
    # Take top fused results for reranking
    fused_top_ids = [chunk_id for chunk_id, _ in fused_results[:Config.TOP_K_VECTOR]]
    
    # Step 4: Fetch chunk contents
    chunks = fetch_chunks(fused_top_ids)
    
    # Attach fusion scores for debug
    fusion_score_map = dict(fused_results)
    for chunk in chunks:
        chunk["rrf_score"] = fusion_score_map.get(chunk["id"], 0)
    
    # Step 5: Cross-encoder reranking
    reranked = rerank(query, chunks, top_n=top_k)
    
    # Collect metrics
    retrieval_metrics = None
    if collect_metrics:
        retrieval_metrics = metrics_collector.collect_retrieval(
            query=query,
            start_time=start_time,
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            fused_count=len(fused_results),
            reranked_count=len(reranked),
            doc_filter=doc_id
        )
    
    # Build debug info if requested
    debug = None
    if include_debug:
        debug = {
            "vector_hits": len(vector_results),
            "bm25_hits": len(bm25_results),
            "fused_count": len(fused_results),
            "rrf_scores": {cid: score for cid, score in fused_results[:10]},
            "vector_scores": {cid: score for cid, score in vector_results[:10]},
            "bm25_scores": {cid: score for cid, score in bm25_results[:10]},
            "retrieval_metrics": retrieval_metrics.__dict__ if retrieval_metrics else None
        }
    
    logger.info(f"Retrieval complete: {len(reranked)} final chunks")
    return reranked, debug
