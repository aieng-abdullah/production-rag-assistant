"""Corpus chunk cache tests.

The cache exists so enabling context widening in #125 does not turn
every user query into a full corpus read. Its one real hazard is
staleness: a cached chunk list that disagrees with the BM25 index built
from the same read.
"""

import pytest

from src.config import Config
from src.services import bm25_cache, corpus_cache


@pytest.fixture(autouse=True)
def clean_caches():
    corpus_cache.clear_chunks()
    bm25_cache.clear_all()
    yield
    corpus_cache.clear_chunks()
    bm25_cache.clear_all()


@pytest.fixture
def fake_load(monkeypatch):
    calls = []

    def _load(tenant_id="default", workspace=None):
        calls.append((tenant_id, workspace))
        return [{"doc_id": "d", "chunk_id": "d_chunk_0", "text": "t"}]

    monkeypatch.setattr(corpus_cache, "load_all_chunks", _load)
    return calls


class TestCaching:
    def test_second_call_does_not_reread(self, fake_load):
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t1", "legal")
        assert len(fake_load) == 1

    def test_workspaces_are_cached_separately(self, fake_load):
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t1", "academic")
        assert len(fake_load) == 2

    def test_tenants_never_share(self, fake_load):
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t2", "legal")
        assert len(fake_load) == 2

    def test_empty_workspace_is_cached_not_reread(self, monkeypatch):
        calls = []

        def _empty(tenant_id="default", workspace=None):
            calls.append(workspace)
            return []

        monkeypatch.setattr(corpus_cache, "load_all_chunks", _empty)
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t1", "legal")
        assert len(calls) == 1, "an empty workspace must not re-query every request"


class TestInvalidation:
    def test_invalidate_drops_that_tenant_only(self, fake_load):
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t2", "legal")
        corpus_cache.invalidate_chunks("t1")
        corpus_cache.get_chunks("t1", "legal")
        corpus_cache.get_chunks("t2", "legal")
        assert len(fake_load) == 3

    def test_bm25_invalidate_also_drops_chunks(self, fake_load):
        """Both caches derive from one read and must go stale together.

        If only the BM25 index were dropped, context widening would
        search chunks that no longer match the index.
        """
        corpus_cache.get_chunks("t1", "legal")
        bm25_cache.invalidate("t1")
        corpus_cache.get_chunks("t1", "legal")
        assert len(fake_load) == 2

    def test_bm25_clear_all_drops_chunks_too(self, fake_load):
        corpus_cache.get_chunks("t1", "legal")
        bm25_cache.clear_all()
        corpus_cache.get_chunks("t1", "legal")
        assert len(fake_load) == 2


class TestNoImportCycle:
    """`corpus_cache` imports `qdrant_client` -> `embedder` -> `chain`.

    Importing it at the top of chain.py closes that loop, which unit
    tests miss because nothing imports `rag_service` at module load.
    """

    def test_app_and_service_import_cleanly(self):
        import importlib

        for module in ("src.api.app", "src.services.rag_service", "src.generation.chain"):
            assert importlib.import_module(module) is not None


class TestGenerationIntegration:
    def test_no_corpus_read_when_widening_is_off(self, monkeypatch, fake_load):
        """The default must cost nothing — no corpus read per query."""
        from src.generation import chain

        monkeypatch.setattr(Config, "CONTEXT_SECTION_WIDTH", 0)
        monkeypatch.setattr(Config, "CONTEXT_NEIGHBOR_WINDOW", 0)
        assert chain._corpus_chunks("t1", "legal") is None
        assert fake_load == []

    def test_corpus_is_cached_when_widening_is_on(self, monkeypatch, fake_load):
        from src.generation import chain

        monkeypatch.setattr(Config, "CONTEXT_SECTION_WIDTH", 1)
        chain._corpus_chunks("t1", "legal")
        chain._corpus_chunks("t1", "legal")
        assert len(fake_load) == 1
