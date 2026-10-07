"""Per-tenant BM25 TTL cache (PLAN.md PR-3, moved from PR-1).

Keyed by (tenant_id, workspace) — each niche keeps its own index so the
legal corpus never leaks into academic retrieval (PLAN PR-4). Built lazily
from ChromaDB chunks on first use; invalidated on ingest/delete for every
workspace of that tenant. TTL bounds staleness on missed invalidations.
"""

from cachetools import TTLCache
from loguru import logger

from src.db.qdrant_client import load_all_chunks
from src.retrieval.bm25_index import build_bm25_index

__all__ = ["get_bm25", "invalidate", "clear_all"]

TTL_SECONDS = 300
MAX_TENANTS = 128

_cache: TTLCache = TTLCache(maxsize=MAX_TENANTS, ttl=TTL_SECONDS)


def get_bm25(tenant_id: str, workspace: str | None = None):
    """Return the tenant's BM25 index for `workspace`, building on cache miss.

    Caches `None` for empty tenants too (membership check, not None check).
    """
    key = (tenant_id, workspace)
    if key in _cache:
        return _cache[key]
    chunks = load_all_chunks(tenant_id, workspace=workspace)
    index = build_bm25_index(chunks) if chunks else None
    _cache[key] = index
    logger.debug(
        "BM25 built tenant={} workspace={} chunks={}", tenant_id, workspace, len(chunks)
    )
    return index


def invalidate(tenant_id: str) -> None:
    """Drop every workspace index of one tenant — call after ingest or delete."""
    for key in [k for k in _cache if k[0] == tenant_id]:
        _cache.pop(key, None)
    logger.debug("BM25 invalidated tenant={}", tenant_id)


def clear_all() -> None:
    """Test/reset hook."""
    _cache.clear()
