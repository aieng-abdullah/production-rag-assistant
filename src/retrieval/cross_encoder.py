from src.config import Config
from sentence_transformers import CrossEncoder
from loguru import logger

_model = None

def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(Config.RERANKER_MODEL,device="cpu") 
    
        logger.info(f"Loaded cross encoder model: {Config.RERANKER_MODEL}")
    return _model


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Rerank chunks against a query using a cross-encoder model.
    """
    if  top_k <1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
    if not chunks:
        logger.warning("rerank() called with empty chunks list — returning []")
        return []

    model = _get_model()

    try:
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"Cross-encoder inference failed: {e}")
        raise

    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    if score_threshold is not None:
        before = len(reranked)
        reranked = [(c, s) for c, s in reranked if s >= score_threshold]
        logger.debug(f"Score threshold {score_threshold} filtered {before - len(reranked)} chunks")

    top = reranked[:top_k]
    logger.debug(f"Reranked {len(chunks)} chunks → returning {len(top)} (top score: {top[0][1]:.4f})" if top else "No chunks passed reranking")

    return [
    {**chunk, "rerank_score": float(score)}
    for chunk, score in reranked[:top_k]
]