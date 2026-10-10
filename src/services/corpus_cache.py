"""Per-tenant corpus chunk cache, invalidated alongside the BM25 index.

The BM25 index is built from `load_all_chunks`, which reads every chunk
for a workspace out of Qdrant. Answer-time context widening needs the
same chunk list to find a chunk's section and neighbours. Caching the
chunks separately means that read happens once per tenant instead of on
every query — otherwise enabling widening in #125 turns every user query
into a full corpus read (1,213 chunks for the academic workspace alone).

Invalidation is shared with `bm25_cache`: both caches key on
`(tenant_id, workspace)` and are dropped by the same `invalidate()`
calls already wired into ingest and delete.

Chunks are the tenant's own data, so unlike the query-embedding cache
this one is strictly tenant-scoped and is never shared across tenants.
"""

from typing import List

from cachetools import TTLCache
from loguru import logger

from src.db.qdrant_client import load_all_chunks

__all__ = ["get_chunks", "invalidate_chunks", "clear_chunks"]

TTL_SECONDS = 900
MAX_TENANTS = 128

_chunks: TTLCache = TTLCache(maxsize=MAX_TENANTS, ttl=TTL_SECONDS)


def get_chunks(tenant_id: str, workspace: str | None = None) -> List[dict] | None:
    """Return the workspace's chunk list, reading from Qdrant on a miss.

    Caches `None` for an empty workspace — membership, not truthiness —
    so a tenant with no documents does not re-query on every request.
    """
    key = (tenant_id, workspace)
    if key in _chunks:
        return _chunks[key]

    chunks = load_all_chunks(tenant_id=tenant_id, workspace=workspace)
    _chunks[key] = chunks
    logger.debug(
        "Corpus chunks cached tenant={} workspace={} chunks={}",
        tenant_id,
        workspace,
        len(chunks),
    )
    return chunks


def invalidate_chunks(tenant_id: str) -> None:
    """Drop every workspace's chunks for one tenant — call after ingest/delete."""
    dropped = sum(1 for key in list(_chunks) if key[0] == tenant_id)
    for key in [k for k in _chunks if k[0] == tenant_id]:
        _chunks.pop(key, None)
    logger.debug("Corpus chunks invalidated tenant={} entries={}", tenant_id, dropped)


def clear_chunks() -> None:
    """Test/reset hook."""
    _chunks.clear()
