"""Embedding generation via the Voyage AI API (``voyage-4-lite``).

Replaces ``HuggingFaceEmbeddings`` (PLAN-render-react Phase 1) — torch and
the ~90MB model download leave the container; embeddings are API-only.

``VoyageAIEmbeddings`` implements LangChain's ``Embeddings`` interface so
``src/db/qdrant_client.py`` keeps passing it straight to
``Chroma(embedding_function=…)``. Module functions ``embed_query`` /
``embed_chunks`` keep their names and signatures — every consumer
(``qdrant_search``, ingestion pipeline, eval) stays untouched.

Vectors are L2-normalized, matching the old local embedder
(``normalize_embeddings=True``): dot product == cosine similarity.
"""

import logging
import time
from threading import Lock
from typing import Dict, List

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from loguru import logger
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import Config
from src.ingestion.embed_cache import cached_embed_query
from src.voyage_pacer import (
    EMBEDDINGS,
    estimate_tokens,
    observe,
    reserve as pacer_reserve,
)
from src.voyage_usage import record as record_usage

# Voyage trial accounts (no payment method): 3 RPM / 10K TPM. Waits below
# outlast one rate-limit window (20s) without hammering the API. Randomised
# so a burst of concurrent retries does not re-synchronise into the next
# window and collide again.
_RETRY_POLICY = dict(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=2, min=5, max=45),
    retry=retry_if_exception_type(httpx.HTTPError),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)

# Single choke point for pacing. The limit is per organization, so one
# process-wide gate has to cover embeddings *and* rerank, not just the
# embedding path — otherwise concurrent callers race past the window.
_rate_lock = Lock()
_last_call = 0.0


def _reset_pacing() -> None:
    """Clear the pacing window (tests)."""
    global _last_call
    with _rate_lock:
        _last_call = 0.0


def _pace(estimated_tokens: int) -> None:
    """Block until `estimated_tokens` fit in the Voyage embedding budget.

    Delegates to `src.voyage_pacer`, which paces by token count rather
    than by call count: a query embedding costs ~10 tokens and a 64-input
    batch costs ~3,000, so any single fixed interval is simultaneously
    too slow for one and too loose for the other. Rerank keeps its own
    window — Voyage counts limits per model, so coupling them would halve
    throughput for nothing.

    `VOYAGE_EMBED_PACE_S` is kept as an additional floor for callers that
    want a guaranteed minimum gap regardless of token accounting.
    """
    pacer_reserve(estimated_tokens, EMBEDDINGS)
    pace_s = Config.VOYAGE_EMBED_PACE_S
    if not pace_s:
        return
    global _last_call
    with _rate_lock:
        elapsed = time.monotonic() - _last_call
        if _last_call and elapsed < pace_s:
            time.sleep(pace_s - elapsed)
        _last_call = time.monotonic()



def _normalize(vector: List[float]) -> List[float]:
    """L2-normalize one embedding (parity with the local model's output)."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        logger.warning("Voyage returned a zero vector — passing through unnormalized")
        return [float(x) for x in arr]
    return (arr / norm).tolist()


@retry(**_RETRY_POLICY)
def _post_embeddings(texts: List[str], input_type: str) -> List[List[float]]:
    """POST ``/embeddings`` with retry + exponential backoff.

    Retries transport errors, timeouts, 429 and 5xx (waits outlast the
    trial 3-RPM window); client errors (bad key, bad model) raise
    immediately. ``input_type`` is ``"query"`` or ``"document"`` — Voyage
    optimizes the embedding space per side.
    """
    if not Config.VOYAGE_API_KEY:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set — embeddings are API-only (no local "
            "fallback). Add it to .env."
        )

    estimated = sum(estimate_tokens(text) for text in texts)
    _pace(estimated)

    response = httpx.post(
        f"{Config.VOYAGE_BASE_URL}/embeddings",
        json={
            "model": Config.VOYAGE_EMBEDDING_MODEL,
            "input": texts,
            "input_type": input_type,
            "truncation": True,
        },
        headers={"Authorization": f"Bearer {Config.VOYAGE_API_KEY}"},
        timeout=Config.EMBED_TIMEOUT_S,
    )
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    if response.status_code >= 400:
        raise RuntimeError(
            f"Voyage embeddings rejected request (HTTP {response.status_code}): "
            f"{response.text[:300]}"
        )

    data = response.json()
    usage = data.get("usage") or {}
    actual = int(usage.get("total_tokens") or 0)
    record_usage("embeddings", actual)
    observe(actual, estimated, EMBEDDINGS)

    items = sorted(data.get("data") or [], key=lambda item: item["index"])
    if len(items) != len(texts):
        raise RuntimeError(
            f"Voyage returned {len(items)} embeddings for {len(texts)} inputs"
        )
    return [_normalize(item["embedding"]) for item in items]


class VoyageAIEmbeddings(LangChainEmbeddings):
    """LangChain ``Embeddings`` implementation backed by Voyage."""

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = batch_size or Config.VOYAGE_EMBED_BATCH_SIZE

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(_post_embeddings(batch, input_type="document"))
            logger.debug(f"Embedded batch {start // self.batch_size + 1} ({len(batch)} texts)")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return _post_embeddings([text], input_type="query")[0]


# Lazy-loaded client (lock: page render + background warm-up race)
_model: VoyageAIEmbeddings | None = None
_model_lock = Lock()


def _get_model() -> VoyageAIEmbeddings:
    """Get or initialize the embedding client (singleton — no model to load)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = VoyageAIEmbeddings()
                logger.info(
                    f"Voyage embeddings client ready: {Config.VOYAGE_EMBEDDING_MODEL}"
                )
    return _model


def embed_query(text: str) -> List[float]:
    """Generate embedding for a single query text.

    Args:
        text: The text to embed.

    Returns:
        List of float embedding values.
    """
    return cached_embed_query(_get_model().embed_query, text)


def embed_chunks(chunks: List[Dict], batch_size: int | None = None) -> List[Dict]:
    """Generate embeddings for chunks in batches.

    Args:
        chunks: Chunk dicts with a ``text`` key (metadata passes through).
        batch_size: Inputs per API call; defaults to the configured
            ``VOYAGE_EMBED_BATCH_SIZE``.
    """
    if not chunks:
        logger.warning("embed_chunks() called with no chunks — returning []")
        return chunks

    texts = [chunk["text"] for chunk in chunks]
    size = batch_size or Config.VOYAGE_EMBED_BATCH_SIZE

    logger.debug(f"Embedding {len(texts)} chunks (batch_size: {size})")
    vectors: List[List[float]] = []
    for start in range(0, len(texts), size):
        vectors.extend(_post_embeddings(texts[start : start + size], input_type="document"))

    for chunk, embedding in zip(chunks, vectors):
        chunk["embedding"] = embedding

    logger.debug(f"Successfully embedded {len(chunks)} chunks")
    return chunks
