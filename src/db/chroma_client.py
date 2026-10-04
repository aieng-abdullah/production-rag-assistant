"""
ChromaDB client using pure LangChain Chroma integration.
"""

from typing import Optional, List, Dict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from src.config import Config
from src.ingestion.embedder import _get_model as get_embedding_model

# Lazy-loaded vectorstore instance
_vectorstore: Optional[Chroma] = None

# Partition key for chunks written before multi-tenancy; also the
# Streamlit/eval single-user tenant. Re-exported by src.services.
DEFAULT_TENANT = "default"


def metadata_where(tenant_id: str, workspace: str | None = None) -> dict:
    """Chroma `where` predicate: tenant partition + optional workspace niche.

    `workspace=None` keeps the legacy tenant-only predicate (tests, eval,
    default flow).
    """
    if workspace is None:
        return {"tenant_id": tenant_id}
    return {
        "$and": [
            {"tenant_id": {"$eq": tenant_id}},
            {"workspace": {"$eq": workspace}},
        ]
    }


def _backfill_metadata(collection) -> None:
    """Idempotent metadata normalization (PLAN.md PR-1 + PR-4).

    Legacy chunks lack `tenant_id` (would vanish from tenant predicates)
    or `workspace` (would vanish from workspace predicates). Tags them with
    DEFAULT_TENANT / DEFAULT_WORKSPACE on client init.
    """
    results = collection.get(include=["metadatas"])
    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []
    stale_ids: list = []
    stale_metas: list = []
    for chunk_id, meta in zip(ids, metadatas):
        if meta is None:
            continue
        patch = {}
        if "tenant_id" not in meta:
            patch["tenant_id"] = DEFAULT_TENANT
        if "workspace" not in meta:
            patch["workspace"] = Config.DEFAULT_WORKSPACE
        if patch:
            stale_ids.append(chunk_id)
            stale_metas.append({**meta, **patch})
    if stale_ids:
        collection.update(ids=stale_ids, metadatas=stale_metas)
        logger.info(
            f"Backfilled tenant/workspace keys on {len(stale_ids)} legacy chunks"
        )


def _get_vectorstore() -> Chroma:
    """Get or initialize the LangChain Chroma vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        embedding_model = get_embedding_model()
        _vectorstore = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(Config.CHROMA_DIR),
        )
        _backfill_metadata(_vectorstore._collection)
        logger.info(f"LangChain Chroma initialized: {Config.COLLECTION_NAME}")
    return _vectorstore


def get_collection():
    """
    Returns the underlying Chroma collection for compatibility.
    Accesses the internal _collection from LangChain Chroma.
    """
    vectorstore = _get_vectorstore()
    # Access internal collection for backward compatibility
    return vectorstore._collection


def upsert_chunks(chunks: List[Dict], tenant_id: str = DEFAULT_TENANT) -> int:
    """
    Add or update chunks in the vectorstore using LangChain add_documents.
    Stamps `tenant_id` as the metadata partition key on every chunk.
    """
    vectorstore = _get_vectorstore()

    # Convert chunks to LangChain Documents.
    # Full metadata passes through (provenance keys — PR-4b-iv); only the
    # partition key `tenant_id` is authoritative server-side, and
    # `workspace` falls back to the configured default when absent.
    documents = []
    ids = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                **{
                    key: value
                    for key, value in chunk.items()
                    if key not in ("text", "embedding")
                },
                "workspace": chunk.get("workspace", Config.DEFAULT_WORKSPACE),
                "tenant_id": tenant_id,
            }
        )
        documents.append(doc)
        doc_id = chunk.get("doc_id", "unknown")
        chunk_index = chunk.get("chunk_index", len(ids))
        base_id = chunk.get("chunk_id", f"{doc_id}_chunk_{chunk_index}")
        # Chroma ids are collection-global: tenants uploading the same
        # filename would collide without the partition prefix.
        ids.append(f"{tenant_id}::{base_id}")

    vectorstore.add_documents(documents=documents, ids=ids)
    logger.info(f"Upserted {len(chunks)} chunks to vectorstore")
    return len(chunks)


def load_all_chunks(
    tenant_id: str = DEFAULT_TENANT, workspace: str | None = None
) -> List[Dict]:
    """
    Load chunks for one tenant (optionally one workspace) from the vectorstore.
    Predicate is enforced server-side via `metadata_where`.
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection

    results = collection.get(where=metadata_where(tenant_id, workspace))
    chunks = []
    for text, metadata in zip(results["documents"], results["metadatas"]):
        chunks.append({
        "text": text,
        "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
        **metadata
    })
    return chunks


def has_chunks(tenant_id: str = DEFAULT_TENANT) -> bool:
    """Check if any chunks exist for a tenant without loading them."""
    return count_chunks(tenant_id) > 0


def purge_tenant(tenant_id: str) -> int:
    """Delete every chunk for one tenant across all workspaces (PR-3b).

    Returns the number of chunk ids removed. Predicate is tenant-only —
    an account purge must not leave workspace-partitioned leftovers.
    """
    collection = get_collection()
    ids = collection.get(where=metadata_where(tenant_id), include=[]).get("ids") or []
    if not ids:
        return 0
    collection.delete(ids=ids)
    logger.info(f"Purged {len(ids)} chunks tenant={tenant_id}")
    return len(ids)


def count_chunks(
    tenant_id: str = DEFAULT_TENANT, workspace: str | None = None
) -> int:
    """Return the tenant's (optionally workspace's) chunk count.

    chromadb >=1.x `Collection.count()` dropped the `where` parameter, so
    the predicate runs through `get(include=[])` (ids only).
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(where=metadata_where(tenant_id, workspace), include=[])
    return len(results.get("ids") or [])


def reset_client():
    """
    Resets the singleton vectorstore (useful for tests).
    """
    global _vectorstore
    _vectorstore = None
    logger.debug("Vectorstore reset")


def get_vectorstore() -> Chroma:
    """Get the LangChain Chroma vectorstore for use in chains.

    Returns:
        Chroma vectorstore instance.
    """
    return _get_vectorstore()