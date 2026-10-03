"""Application service — single entry point for RAG operations.

PLAN.md PR-0. Every method takes `tenant_id`; tenant filtering itself lands
in PR-1. Until then all tenants share one corpus and `tenant_id` is logged
but not enforced — do not treat it as an isolation boundary yet.
"""

from pathlib import Path

from loguru import logger

from src.db.chroma_client import get_collection
from src.generation.Citation_system import CitedAnswer
from src.generation.chain import generate
from src.generation.providers import ProviderOverrides
from src.ingestion.pipeline import ingest as _ingest_pipeline

DEFAULT_TENANT = "default"


class RAGService:
    """Stateless facade over ingestion, generation, and document storage."""

    def ingest(self, tenant_id: str, path: str | Path) -> dict:
        """Run the full PDF pipeline. Returns {"pages": int, "chunks": int}."""
        logger.info(f"Ingest tenant={tenant_id} path={path}")
        return _ingest_pipeline(str(path))

    def query(
        self,
        tenant_id: str,
        query: str,
        bm25_index,
        provider_overrides: ProviderOverrides | None = None,
    ) -> CitedAnswer:
        """Generate a citation-validated answer for `query`."""
        logger.debug(f"Query tenant={tenant_id}")
        return generate(query, bm25_index, provider_overrides=provider_overrides)

    def list_docs(self, tenant_id: str) -> list[str]:
        """Distinct doc_ids in the vector store, sorted."""
        results = get_collection().get()
        doc_ids = {
            meta.get("doc_id")
            for meta in (results.get("metadatas") or [])
            if meta and meta.get("doc_id")
        }
        return sorted(doc_ids)

    def delete(self, tenant_id: str, doc_id: str) -> None:
        """Remove all chunks for `doc_id` from the vector store."""
        get_collection().delete(where={"doc_id": doc_id})
        logger.info(f"Deleted doc={doc_id} tenant={tenant_id}")
