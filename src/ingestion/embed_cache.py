"""In-process cache for Voyage query embeddings.

A query embedding is a pure function of (model, input type, text). It
does not depend on the tenant, the corpus, or anything that changes
under it, so a cached entry can never go stale and needs no
invalidation.

**Tenant isolation.** The key is derived from the query text only. Two
users asking the same question produce the same vector because the
embedding is a function of the string, not of who asked — nothing
tenant-scoped enters the key, and the cached value carries no document
content. Document embeddings are not cached here: they belong to a
specific tenant's corpus and are invalidated by ingest and delete.

This exists for the evaluation loop, not for live traffic. Live queries
are unique, so the hit rate there is near zero; `eval/retrieval_precision.py`
asks the same 37 fixed questions and spends ~13 minutes in Voyage pacing,
which drops to near zero on a re-run.
"""

import hashlib
from typing import List

from cachetools import TTLCache
from loguru import logger

from src.config import Config

__all__ = ["cached_embed_query", "invalidate_all", "snapshot"]

_cache: TTLCache = TTLCache(maxsize=4096, ttl=Config.VOYAGE_CACHE_TTL_S)

_hits = 0
_misses = 0


def _key(text: str, input_type: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{Config.VOYAGE_EMBEDDING_MODEL}:{input_type}:{digest}"


def cached_embed_query(embed, text: str) -> List[float]:
    """Return the embedding for `text`, calling `embed` only on a miss.

    `embed` is the uncached callable, injected so this module does not
    import the embedder and create a cycle.
    """
    global _hits, _misses
    key = _key(text, "query")
    if key in _cache:
        _hits += 1
        return _cache[key]

    _misses += 1
    vector = embed(text)
    _cache[key] = vector
    logger.debug("Voyage embed cache miss (ttl={}s)", Config.VOYAGE_CACHE_TTL_S)
    return vector


def invalidate_all() -> None:
    """Drop every cached embedding (tests)."""
    global _hits, _misses
    _cache.clear()
    _hits = 0
    _misses = 0


def snapshot() -> dict:
    """Hit/miss counters — diagnostics and tests."""
    total = _hits + _misses
    return {
        "hits": _hits,
        "misses": _misses,
        "size": len(_cache),
        "hit_rate": round(_hits / total, 4) if total else None,
    }
