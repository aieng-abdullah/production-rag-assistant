"""Per-tenant BM25 TTL cache (PLAN.md PR-3, moved from PR-1).

Built lazily from ChromaDB chunks on first query per tenant; invalidated
on ingest/delete. TTL bounds staleness when a worker misses an invalidation.
"""

from cachetools import TTLCache
from loguru import logger

from src.db.chroma_client import load_all_chunks
from src.retrieval.bm25_index import build_bm25_index

__all__ = ["get_bm25", "invalidate", "clear_all"]

TTL_SECONDS = 300
MAX_TENANTS = 128

_cache: TTLCache = TTLCache(maxsize=MAX_TENANTS, ttl=TTL_SECONDS)


def get_bm25(tenant_id: str):
    """Return the tenant's BM25 index, building it on cache miss.

    Caches `None` for empty tenants too (membership check, not None check).
    """
    if tenant_id in _cache:
        return _cache[tenant_id]
    chunks = load_all_chunks(tenant_id)
    index = build_bm25_index(chunks) if chunks else None
    _cache[tenant_id] = index
    logger.debug("BM25 built tenant={} chunks={}", tenant_id, len(chunks))
    return index


def invalidate(tenant_id: str) -> None:
    """Drop one tenant's index — call after ingest or delete."""
    _cache.pop(tenant_id, None)
    logger.debug("BM25 invalidated tenant={}", tenant_id)


def clear_all() -> None:
    """Test/reset hook."""
    _cache.clear()
