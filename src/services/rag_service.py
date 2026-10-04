"""Application service — single entry point for RAG operations.

PLAN.md PR-1: every method takes an explicit `tenant_id` context argument
(function-scoped propagation, no ambient global state). The Chroma layer
enforces the matching `where={"tenant_id": ...}` predicate on all reads
and writes. Legacy chunks without the partition key are backfilled to
`DEFAULT_TENANT` at vectorstore init.
"""

from pathlib import Path

from loguru import logger

from src.config import Config
from src.db.chroma_client import DEFAULT_TENANT, get_collection
from src.generation.Citation_system import CitedAnswer
from src.generation.chain import generate
from src.generation.profiles import get_prompt_version
from src.generation.providers import ProviderOverrides
from src.ingestion.pipeline import ingest as _ingest_pipeline

__all__ = ["RAGService", "DEFAULT_TENANT"]


class RAGService:
    """Stateless facade over ingestion, generation, and document storage."""

    def ingest(
        self,
        tenant_id: str,
        path: str | Path,
        workspace: str = Config.DEFAULT_WORKSPACE,
    ) -> dict:
        """Run the full PDF pipeline. Returns {"pages": int, "chunks": int}."""
        logger.info(f"Ingest tenant={tenant_id} workspace={workspace} path={path}")
        return _ingest_pipeline(str(path), tenant_id=tenant_id, workspace=workspace)

    def generate_answer(
        self,
        tenant_id: str,
        query: str,
        bm25_index,
        provider_overrides: ProviderOverrides | None = None,
        workspace: str = Config.DEFAULT_WORKSPACE,
    ) -> CitedAnswer:
        """Generate a citation-validated answer scoped to `tenant_id` + `workspace`."""
        logger.debug(
            "Query tenant={} workspace={} prompt={}",
            tenant_id,
            workspace,
            get_prompt_version(workspace),
        )
        return generate(
            query,
            bm25_index,
            provider_overrides=provider_overrides,
            tenant_id=tenant_id,
            workspace=workspace,
        )

    def list_documents(self, tenant_id: str) -> list[str]:
        """Distinct doc_ids for one tenant, sorted. Cross-tenant IDs never returned."""
        results = get_collection().get(where={"tenant_id": tenant_id})
        doc_ids = {
            meta.get("doc_id")
            for meta in (results.get("metadatas") or [])
            if meta and meta.get("doc_id")
        }
        return sorted(doc_ids)

    def delete_document(self, tenant_id: str, doc_id: str) -> None:
        """Delete a document's chunks — predicate requires BOTH tenant and doc match."""
        get_collection().delete(
            where={
                "$and": [
                    {"tenant_id": {"$eq": tenant_id}},
                    {"doc_id": {"$eq": doc_id}},
                ]
            }
        )
        logger.info(f"Deleted doc={doc_id} tenant={tenant_id}")
