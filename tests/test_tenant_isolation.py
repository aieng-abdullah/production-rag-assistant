"""Cross-tenant isolation conformance suite (PLAN.md PR-1 merge gate).

Runs against a REAL persistent Chroma collection (tmp_path) with synthetic
deterministic embeddings — no model download, no mocked storage layer.
Proves tenant A's data is invisible to tenant B on every read path and
that deletes cannot cross partitions.
"""

from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

DIM = 8


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-chars embeddings — enough for storage-path tests."""

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * DIM
        for i, ch in enumerate(text.lower()):
            v[ord(ch) % DIM] += 1.0 + (i % 3) * 0.01
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def tenant_store(monkeypatch, tmp_path):
    """Real Chroma persistent store isolated to tmp_path; fake embeddings."""
    import src.db.chroma_client as mod
    from src.config import Config

    monkeypatch.setattr(Config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(mod, "get_embedding_model", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        "src.retrieval.chroma_search.embed_query", FakeEmbeddings().embed_query
    )
    mod.reset_client()
    yield mod
    mod.reset_client()


def _chunks(doc_id: str, n: int = 2) -> list[dict]:
    return [
        {
            "text": f"content of {doc_id} chunk {i}",
            "doc_id": doc_id,
            "page_num": i,
            "chunk_index": i,
        }
        for i in range(n)
    ]


class TestTenantIsolation:
    def test_counts_are_partition_scoped(self, tenant_store):
        tenant_store.upsert_chunks(_chunks("a-doc"), tenant_id="tenant-a")

        assert tenant_store.count_chunks("tenant-a") == 2
        assert tenant_store.count_chunks("tenant-b") == 0
        assert tenant_store.has_chunks("tenant-a") is True
        assert tenant_store.has_chunks("tenant-b") is False

    def test_has_chunks_accepts_workspace_filter(self, tenant_store):
        """has_chunks(tenant, workspace) must scope to one niche (PR-4)."""
        legal = [{**c, "workspace": "legal"} for c in _chunks("law.pdf")]
        academic = [{**c, "workspace": "academic"} for c in _chunks("paper.pdf")]
        tenant_store.upsert_chunks(legal + academic, tenant_id="tenant-a")

        assert tenant_store.has_chunks("tenant-a", workspace="legal") is True
        assert tenant_store.has_chunks("tenant-a", workspace="academic") is True
        assert tenant_store.has_chunks("tenant-a", workspace="missing") is False
        assert tenant_store.has_chunks("tenant-b", workspace="legal") is False

    def test_load_all_chunks_never_crosses_partitions(self, tenant_store):
        tenant_store.upsert_chunks(_chunks("a-doc"), tenant_id="tenant-a")

        assert tenant_store.load_all_chunks("tenant-b") == []
        loaded_a = tenant_store.load_all_chunks("tenant-a")
        assert {c["doc_id"] for c in loaded_a} == {"a-doc"}
        assert all(c["tenant_id"] == "tenant-a" for c in loaded_a)

    def test_vector_search_returns_empty_for_foreign_tenant(self, tenant_store):
        tenant_store.upsert_chunks(_chunks("a-doc"), tenant_id="tenant-a")

        from src.retrieval.chroma_search import vector_search

        assert vector_search("content of a-doc", top_k=5, tenant_id="tenant-b") == []
        hits = vector_search("content of a-doc", top_k=5, tenant_id="tenant-a")
        assert {h["doc_id"] for h in hits} == {"a-doc"}

    def test_same_doc_id_both_tenants_delete_stays_in_partition(self, tenant_store):
        """Both tenants own doc_id 'shared.pdf'; A's delete must not touch B."""
        tenant_store.upsert_chunks(_chunks("shared.pdf"), tenant_id="tenant-a")
        tenant_store.upsert_chunks(_chunks("shared.pdf"), tenant_id="tenant-b")
        assert tenant_store.count_chunks("tenant-a") == 2
        assert tenant_store.count_chunks("tenant-b") == 2

        from src.services import RAGService

        RAGService().delete_document("tenant-a", "shared.pdf")

        assert tenant_store.count_chunks("tenant-a") == 0
        assert tenant_store.count_chunks("tenant-b") == 2

    def test_list_documents_is_partition_scoped(self, tenant_store):
        from src.services import RAGService

        tenant_store.upsert_chunks(_chunks("a-doc"), tenant_id="tenant-a")
        tenant_store.upsert_chunks(_chunks("b-doc"), tenant_id="tenant-b")
        service = RAGService()

        assert service.list_documents("tenant-a") == ["a-doc"]
        assert service.list_documents("tenant-b") == ["b-doc"]

    def test_ingest_stamps_partition_key_end_to_end(self, tenant_store):
        """ingest() threads tenant_id into the metadata written by upsert."""
        with patch(
            "src.ingestion.pipeline.extract_pages",
            return_value=[{"page_num": 1, "text": "hello world"}],
        ), patch(
            "src.ingestion.pipeline.chunk_pages",
            return_value=[{"text": "hello world", "page_num": 1, "chunk_index": 0,
                           "doc_id": "x.pdf"}],
        ), patch(
            "src.ingestion.pipeline.embed_chunks",
            side_effect=lambda chunks: chunks,
        ):
            from src.services import RAGService

            result = RAGService().ingest("tenant-a", "data/raw/x.pdf")

        assert result == {"pages": 1, "chunks": 1}
        assert tenant_store.count_chunks("tenant-a") == 1
        assert tenant_store.count_chunks("tenant-b") == 0


class TestLegacyBackfill:
    def test_chunks_without_partition_key_normalize_to_default_tenant(
        self, tenant_store
    ):
        """Pre-PR-1 chunks (no tenant_id) must land in DEFAULT_TENANT, not vanish."""
        fake = FakeEmbeddings()
        collection = tenant_store.get_collection()
        collection.add(
            ids=["legacy_0"],
            documents=["legacy text"],
            embeddings=[fake.embed_query("legacy text")],
            metadatas=[{"doc_id": "legacy.pdf", "chunk_index": 0}],
        )
        assert tenant_store.load_all_chunks("tenant-a") == []  # not silently absorbed

        tenant_store.reset_client()  # re-init triggers idempotent backfill
        defaulted = tenant_store.load_all_chunks(tenant_store.DEFAULT_TENANT)

        assert [c["chunk_id"] for c in defaulted] == ["legacy.pdf_chunk_0"]
        assert defaulted[0]["tenant_id"] == "default"
