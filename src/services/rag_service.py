"""Application service — single entry point for RAG operations.

PLAN.md PR-1: every method takes an explicit `tenant_id` context argument
(function-scoped propagation, no ambient global state). The Chroma layer
enforces the matching `where={"tenant_id": ...}` predicate on all reads
and writes. Legacy chunks without the partition key are backfilled to
`DEFAULT_TENANT` at vectorstore init.
"""

from pathlib import Path
import shutil

from loguru import logger

from src.config import Config
from src.db.qdrant_client import DEFAULT_TENANT, get_collection, purge_tenant
from src.generation.Citation_system import CitedAnswer
from src.generation.chain import generate
from src.generation.profiles import get_prompt_version
from src.generation.providers import ProviderOverrides
from src.ingestion.pipeline import ingest as _ingest_pipeline
from src.services.bm25_cache import invalidate

__all__ = ["RAGService", "DEFAULT_TENANT"]


class RAGService:
    """Stateless facade over ingestion, generation, and document storage."""

    def ingest(
        self,
        tenant_id: str,
        path: str | Path,
        workspace: str = Config.DEFAULT_WORKSPACE,
        provenance: dict | None = None,
    ) -> dict:
        """Run the full PDF pipeline. Returns {"pages": int, "chunks": int}.

        `provenance` carries optional doc-level upload inputs (date,
        version, jurisdiction) stamped onto every chunk (PR-4b-iv)."""
        logger.info(f"Ingest tenant={tenant_id} workspace={workspace} path={path}")
        return _ingest_pipeline(
            str(path), tenant_id=tenant_id, workspace=workspace, provenance=provenance
        )

    def generate_answer(
        self,
        tenant_id: str,
        query: str,
        bm25_index,
        provider_overrides: ProviderOverrides | None = None,
        workspace: str = Config.DEFAULT_WORKSPACE,
        history: list[dict] | None = None,
    ) -> CitedAnswer:
        """Generate a citation-validated answer scoped to `tenant_id` + `workspace`.

        `history` (optional) carries prior sanitized chat turns so the
        pipeline can rewrite follow-ups and ground the prompt in context."""
        logger.debug(
            "Query tenant={} workspace={} prompt={} history_turns={}",
            tenant_id,
            workspace,
            get_prompt_version(workspace),
            len(history or []),
        )
        return generate(
            query,
            bm25_index,
            provider_overrides=provider_overrides,
            tenant_id=tenant_id,
            workspace=workspace,
            history=history,
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

    def delete_tenant_data(self, tenant_id: str) -> dict:
        """Purge everything stored for one tenant (PR-3b account deletion).

        Order: chunks first (vector store is the retryable side), raw files
        after. Returns {"chunks": n, "files": m} for logging/tests.
        """
        chunks = purge_tenant(tenant_id)

        files = 0
        raw_dir = Config.DATA_DIR / "raw" / tenant_id
        if raw_dir.is_dir():
            files = sum(1 for p in raw_dir.iterdir() if p.is_file())
            try:
                shutil.rmtree(raw_dir)
            except OSError as exc:
                logger.warning(f"Raw dir purge failed dir={raw_dir}: {exc}")

        invalidate(tenant_id)
        logger.info(
            f"Purged tenant={tenant_id} chunks={chunks} files={files}",
        )
        return {"chunks": chunks, "files": files}
