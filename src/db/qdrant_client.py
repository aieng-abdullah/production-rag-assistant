"""Qdrant vector store client — 1:1 port of the old Chroma interface."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import Config
from src.ingestion.embedder import _get_model as get_embedding_model

_vectorstore: Optional["QdrantStore"] = None

DEFAULT_TENANT = "default"
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


def _point_id(tenant_id: str, chunk_key: str) -> str:
    """Deterministic Qdrant point id for a tenant-scoped chunk."""
    return str(uuid.uuid5(_NAMESPACE, f"{tenant_id}::{chunk_key}"))


def metadata_where(tenant_id: str, workspace: str | None = None) -> Filter:
    """Qdrant filter: tenant partition + optional workspace niche."""
    must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    if workspace is not None:
        must.append(FieldCondition(key="workspace", match=MatchValue(value=workspace)))
    return Filter(must=must)


def _where_to_filter(where: Any) -> Filter | None:
    if where is None:
        return None
    if isinstance(where, Filter):
        return where
    if not isinstance(where, dict):
        raise TypeError(f"Unsupported where: {type(where)}")
    if "$and" in where:
        must: list[FieldCondition] = []
        for part in where["$and"]:
            sub = _where_to_filter(part)
            if sub and sub.must:
                must.extend(sub.must)
        return Filter(must=must)

    must: list[FieldCondition] = []
    for key, value in where.items():
        if isinstance(value, dict) and "$eq" in value:
            value = value["$eq"]
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must)


class QdrantStore:
    """Small Chroma-like adapter over QdrantClient."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection {self.collection_name} size={vector_size}")
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        for field in ("tenant_id", "workspace", "doc_id"):
            try:
                self.client.create_payload_index(
                    self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug(f"Payload index {field} not created: {exc}")

    def _scroll(self, scroll_filter: Filter | None = None, include: list[str] | None = None):
        include = include or ["documents", "metadatas"]
        points = []
        offset: str | int | None = None
        while True:
            batch, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                with_payload=("documents" in include or "metadatas" in include),
                with_vectors=("embeddings" in include),
                offset=offset,
            )
            points.extend(batch)
            if offset is None:
                break
        return points

    def get(self, where: Any = None, include: list[str] | None = None):
        if not self.client.collection_exists(self.collection_name):
            return {"ids": [], "documents": [], "metadatas": []}
        qfilter = _where_to_filter(where)
        include = include or ["documents", "metadatas"]
        points = self._scroll(scroll_filter=qfilter, include=include)
        ids = [str(p.id) for p in points]
        documents = [p.payload.get("text") for p in points] if "documents" in include else []
        metadatas = [dict(p.payload or {}) for p in points] if "metadatas" in include else []
        if metadatas:
            for meta in metadatas:
                meta.pop("text", None)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def query(self, query_embeddings: list[list[float]], n_results: int, where: Any = None):
        if not self.client.collection_exists(self.collection_name):
            return {"documents": [[]], "metadatas": [[]]}
        qfilter = _where_to_filter(where)
        results = self.client.query_points(
            self.collection_name,
            query=query_embeddings[0],
            limit=n_results,
            query_filter=qfilter,
            with_payload=True,
        )
        docs = []
        metas = []
        for point in results.points:
            payload = dict(point.payload or {})
            docs.append(payload.pop("text", None))
            metas.append(payload)
        return {"documents": [docs], "metadatas": [metas]}

    def upsert(self, embeddings: list[list[float]], documents: list[str], metadatas: list[dict], ids: list[str]):
        if not embeddings:
            return
        self._ensure_collection(len(embeddings[0]))
        points = []
        for vector, text, meta, pid in zip(embeddings, documents, metadatas, ids):
            tenant_id = meta.get("tenant_id", DEFAULT_TENANT)
            point_id = _point_id(str(tenant_id), str(pid))
            payload = {**meta, "text": text}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        self.client.upsert(self.collection_name, points=points)

    def delete(self, where: Any = None, ids: list[str] | None = None):
        if not self.client.collection_exists(self.collection_name):
            return
        if ids is not None:
            # Convert Chroma-style ids back to deterministic point ids.
            point_ids = []
            for id_str in ids:
                if "::" in id_str:
                    tenant, chunk_key = id_str.split("::", 1)
                    point_ids.append(_point_id(tenant, chunk_key))
                else:
                    point_ids.append(_point_id(DEFAULT_TENANT, id_str))
            self.client.delete(self.collection_name, points_selector=point_ids)
            return
        qfilter = _where_to_filter(where)
        if qfilter is None:
            return
        self.client.delete(self.collection_name, points_selector=qfilter)

    def update(self, ids: list[str], metadatas: list[dict]):
        if not self.client.collection_exists(self.collection_name):
            return
        existing = self._scroll(include=["metadatas"])
        existing_by_chroma_id: dict[str, Any] = {}
        for point in existing:
            payload = point.payload or {}
            # Reconstruct the Chroma-style id used by callers.
            # New upserts use f"{tenant_id}::{base_id}" for deletion only; for
            # metadata updates we match on tenant_id + base id embedded in payload.
            doc_id = payload.get("doc_id")
            chunk_index = payload.get("chunk_index")
            tenant_id = payload.get("tenant_id", DEFAULT_TENANT)
            if doc_id is not None and chunk_index is not None:
                existing_by_chroma_id[f"{tenant_id}::{doc_id}_chunk_{chunk_index}"] = point
        for chroma_id, meta_patch in zip(ids, metadatas):
            point = existing_by_chroma_id.get(chroma_id)
            if point is None:
                # Support raw deterministic tenant::chunk_id strings.
                if "::" in chroma_id:
                    tenant, chunk_key = chroma_id.split("::", 1)
                    pid = _point_id(tenant, chunk_key)
                    self.client.set_payload(self.collection_name, payload=meta_patch, points=[pid])
                continue
            self.client.set_payload(self.collection_name, payload=meta_patch, points=[point.id])


def _make_client() -> QdrantClient:
    url = getattr(Config, "QDRANT_URL", "") or os.getenv("QDRANT_URL", "")
    api_key = getattr(Config, "QDRANT_API_KEY", "") or os.getenv("QDRANT_API_KEY", "")
    if not url or url == ":memory:":
        return QdrantClient(":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=api_key or None)
    # Treat anything else as a filesystem path.
    return QdrantClient(path=url)


def _backfill_metadata(store: QdrantStore) -> None:
    """Post-v1 chunks may miss partition keys; tag them idempotently."""
    try:
        points = store._scroll(include=["metadatas"])
        for point in points:
            payload = dict(point.payload or {})
            patch = {}
            if "tenant_id" not in payload:
                patch["tenant_id"] = DEFAULT_TENANT
            if "workspace" not in payload:
                patch["workspace"] = Config.DEFAULT_WORKSPACE
            if patch:
                store.client.set_payload(store.collection_name, payload=patch, points=[point.id])
                logger.info(f"Backfixed {len(patch)} keys on legacy chunk {point.id}")
    except Exception as exc:
        logger.debug(f"Metadata backfill skipped: {exc}")


def _get_vectorstore() -> QdrantStore:
    global _vectorstore
    if _vectorstore is None:
        client = _make_client()
        _vectorstore = QdrantStore(client, Config.COLLECTION_NAME)
        try:
            _vectorstore._ensure_payload_indexes()
        except Exception as exc:
            logger.debug(f"Payload index bootstrap skipped: {exc}")
        _backfill_metadata(_vectorstore)
        logger.info(f"Qdrant vectorstore initialized: {Config.COLLECTION_NAME}")
    return _vectorstore


def get_collection() -> QdrantStore:
    return _get_vectorstore()


def upsert_chunks(chunks: List[Dict], tenant_id: str = DEFAULT_TENANT) -> int:
    vectorstore = _get_vectorstore()

    documents: list[str] = []
    ids: list[str] = []
    embeddings: list[list[float] | None] = []
    metadatas: list[dict] = []

    precomputed = bool(chunks) and all(chunk.get("embedding") for chunk in chunks)
    for chunk in chunks:
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in ("text", "embedding")
        }
        metadata["workspace"] = chunk.get("workspace", Config.DEFAULT_WORKSPACE)
        metadata["tenant_id"] = tenant_id

        doc_id = chunk.get("doc_id", "unknown")
        chunk_index = chunk.get("chunk_index", len(ids))
        base_id = chunk.get("chunk_id", f"{doc_id}_chunk_{chunk_index}")
        ids.append(f"{tenant_id}::{base_id}")
        documents.append(chunk["text"])
        embeddings.append(chunk.get("embedding"))
        metadatas.append(metadata)

    if precomputed:
        vectorstore.upsert(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)  # type: ignore[arg-type]
    else:
        model = get_embedding_model()
        embedded = model.embed_documents(documents)
        vectorstore.upsert(embeddings=embedded, documents=documents, metadatas=metadatas, ids=ids)

    logger.info(f"Upserted {len(chunks)} chunks to Qdrant")
    return len(chunks)


def load_all_chunks(
    tenant_id: str = DEFAULT_TENANT, workspace: str | None = None
) -> List[Dict]:
    vectorstore = _get_vectorstore()
    results = vectorstore.get(where=metadata_where(tenant_id, workspace))
    chunks = []
    for text, metadata in zip(results.get("documents", []), results.get("metadatas", [])):
        chunks.append({
            "text": text,
            "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
            **metadata,
        })
    return chunks


def has_chunks(tenant_id: str = DEFAULT_TENANT, workspace: str | None = None) -> bool:
    return count_chunks(tenant_id, workspace) > 0


def reassign_tenant(old_tenant: str, new_tenant: str) -> int:
    collection = get_collection()
    results = collection.get(where=metadata_where(old_tenant))
    ids = results.get("ids", [])
    if not ids:
        return 0
    metas = []
    for meta in results.get("metadatas", []):
        metas.append({**meta, "tenant_id": new_tenant})
    # Chroma-style ids embed tenant; rebuild ids for Qdrant set_payload matching.
    chroma_ids = []
    for meta in results.get("metadatas", []):
        doc_id = meta.get("doc_id")
        chunk_index = meta.get("chunk_index")
        if doc_id is not None and chunk_index is not None:
            chroma_ids.append(f"{old_tenant}::{doc_id}_chunk_{chunk_index}")
    if chroma_ids:
        collection.update(ids=chroma_ids, metadatas=metas)
    logger.info(f"Reassigned {len(ids)} chunks tenant={old_tenant} -> {new_tenant}")
    return len(ids)


def purge_tenant(tenant_id: str) -> int:
    collection = get_collection()
    ids = collection.get(where=metadata_where(tenant_id), include=[]).get("ids") or []
    if not ids:
        return 0
    collection.delete(where=metadata_where(tenant_id))
    logger.info(f"Purged {len(ids)} chunks tenant={tenant_id}")
    return len(ids)


def count_chunks(
    tenant_id: str = DEFAULT_TENANT, workspace: str | None = None
) -> int:
    vectorstore = _get_vectorstore()
    results = vectorstore.get(where=metadata_where(tenant_id, workspace), include=[])
    return len(results.get("ids") or [])


def reset_client():
    global _vectorstore
    if _vectorstore is not None:
        try:
            _vectorstore.client.close()
        except Exception:
            pass
    _vectorstore = None
    logger.debug("Qdrant vectorstore reset")


def get_vectorstore() -> QdrantStore:
    return _get_vectorstore()
