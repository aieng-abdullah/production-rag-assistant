"""Voyage AI reranking (``rerank-3-lite``) — API-only, no local model.

Replaces ``src/retrieval/cross_encoder.py`` (PLAN-render-react Phase 1).

Score scale: Voyage returns ``relevance_score`` in ``[0, 1]`` (higher =
more relevant). The local cross-encoder this replaced returned raw
logits, so ``score_threshold`` compares on the ``[0, 1]`` scale — no
logit/softmax conversion is applied.

There is deliberately no local fallback: an API failure surfaces as an
exception and ``src/retrieval/pipeline.py`` wraps it into
``RuntimeError("Error while reranking: …")``.
"""

import logging

import httpx
from loguru import logger
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Config
from src.voyage_usage import record as record_usage


# Voyage trial accounts (no payment method): 3 RPM. Waits below outlast one
# rate-limit window (20s) without hammering the API.
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=45),
    retry=retry_if_exception_type(httpx.HTTPError),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def _post_rerank(query: str, documents: list[str]) -> list[dict]:
    """POST ``/rerank`` with retry + exponential backoff.

    Retries transport errors, timeouts and 429/5xx (waits outlast the trial
    3-RPM window). Client errors (bad key, bad model) raise immediately —
    retrying cannot fix them.
    """
    if not Config.VOYAGE_API_KEY:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set — rerank is API-only (no local "
            "fallback). Add it to .env."
        )

    response = httpx.post(
        f"{Config.VOYAGE_BASE_URL}/rerank",
        json={
            "model": Config.VOYAGE_RERANKER_MODEL,
            "query": query,
            "documents": documents,
            "top_k": len(documents),
        },
        headers={"Authorization": f"Bearer {Config.VOYAGE_API_KEY}"},
        timeout=Config.RERANK_TIMEOUT_S,
    )
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    if response.status_code >= 400:
        raise RuntimeError(
            f"Voyage rerank rejected request (HTTP {response.status_code}): "
            f"{response.text[:300]}"
        )

    data = response.json()
    usage = data.get("usage") or {}
    record_usage("rerank", int(usage.get("total_tokens") or 0))
    # Voyage /rerank responds with `data` ([{"index", "relevance_score"}]).
    return data.get("data") or []


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int,
    score_threshold: float | None = None,
) -> list[dict]:
    """Rerank chunks against a query via the Voyage rerank API.

    Returns up to ``top_k`` chunks sorted by descending relevance score,
    each carrying a ``rerank_score`` in ``[0, 1]``.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")

    if not chunks:
        logger.warning("rerank() called with empty chunks list — returning []")
        return []

    results = _post_rerank(query, [chunk["text"] for chunk in chunks])

    scored: list[tuple[dict, float]] = []
    for item in results:
        try:
            index = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Skipping malformed Voyage rerank result: {exc}")
            continue
        if 0 <= index < len(chunks):
            scored.append((chunks[index], score))
        else:
            logger.warning(f"Voyage rerank index out of range: {index} (n={len(chunks)})")

    scored.sort(key=lambda pair: pair[1], reverse=True)

    if score_threshold is not None:
        before = len(scored)
        scored = [(c, s) for c, s in scored if s >= score_threshold]
        logger.debug(
            f"Score threshold {score_threshold} filtered {before - len(scored)} chunks"
        )

    top = scored[:top_k]
    logger.debug(
        f"Reranked {len(chunks)} chunks → returning {len(top)} "
        f"(top score: {top[0][1]:.4f})"
        if top
        else "No chunks passed reranking"
    )

    return [{**chunk, "rerank_score": float(score)} for chunk, score in top]
