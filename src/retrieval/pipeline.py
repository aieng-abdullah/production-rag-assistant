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


def retrieval(
    query: str,
    bm25_index,
    top_k: int = 5,
    lf_retrieval_parent=None,
) -> list[dict]:
    """
    Retrieve relevant text chunks from a vector database using embeddings,
    using traditional BM25 search and vector search then uses a reranker to combine them.

    lf_retrieval_parent: optional Langfuse retriever span; when set, records bm25 / vector /
    rrf / rerank as child retriever observations.
    """
    def _traced_step(name: str, fn):
        if lf_retrieval_parent is None:
            return fn()
        with lf_retrieval_parent.start_as_current_observation(
            name=name,
            as_type="retriever",
            input={"query": query},
        ) as obs:
            try:
                out = fn()
                obs.update(output={"result_count": len(out)})
                return out
            except Exception as e:
                obs.update(level="ERROR", status_message=str(e))
                raise

    # BM25 search
    try:
        bm25_results = _traced_step(
            "bm25-search",
            lambda: bm25_search(bm25_index, query, top_k=20),
        )
        logger.info(f"BM25 search returned {len(bm25_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while BM25 search: {e}")

    # Vector search
    try:
        vector_results = _traced_step(
            "vector-search",
            lambda: vector_search(query, top_k=20),
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while vector search: {e}")

    # RRF fusion
    try:
        rrf_results = _traced_step(
            "rrf-fusion",
            lambda: rrf_fusion(bm25_results, vector_results, top_k=20),
        )
        logger.info(f"RRF fusion returned {len(rrf_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while RRF fusion: {e}")

    # Rerank
    try:
        rerank_results = _traced_step(
            "rerank",
            lambda: rerank(query, rrf_results, top_k),
        )
        logger.info(f"Rerank returned {len(rerank_results)} results")

    except Exception as e:
        raise RuntimeError(f"Error while reranking: {e}")

    return rerank_results