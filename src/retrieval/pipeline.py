"""
pipeline.py:
            pipeline: retrieves relevant text chunks from a vector database using embeddings,
            using traditional BM25 search and vector search then uses a reranker to combine them.
"""
from loguru import logger

from src.retrieval.bm25_index import bm25_search
from src.retrieval.chroma_search import vector_search
from src.retrieval.hybrid_fusion import rrf_fusion
from src.retrieval.cross_encoder import rerank


def retrieval(query: str, chunks: list[dict], bm25_index, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant text chunks from a vector database using embeddings,
    using traditional BM25 search and vector search then uses a reranker to combine them.
    """
    # BM25 search
    try:
        bm25_results = bm25_search(bm25_index, query, chunks, top_k=20)
        logger.info(f"BM25 search returned {len(bm25_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while BM25 search: {e}")

    # Vector search
    try:
        vector_results = vector_search(query, top_k=20)
        logger.info(f"Vector search returned {len(vector_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while vector search: {e}")

    # RRF fusion
    try:
        rrf_results = rrf_fusion(bm25_results, vector_results, top_k=20)
        logger.info(f"RRF fusion returned {len(rrf_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while RRF fusion: {e}")

    # Rerank
    try:
        rerank_results = rerank( query,rrf_results, top_k)  
        logger.info(f"Rerank returned {len(rerank_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while reranking: {e}")

    return rerank_results