"""Tests for the RAGService application-service facade (PLAN.md PR-0)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.generation.Citation_system import CitedAnswer, Source
from src.generation.providers import ProviderOverrides
from src.services import DEFAULT_TENANT, RAGService


@pytest.fixture
def service():
    return RAGService()


class TestIngest:
    @patch("src.services.rag_service._ingest_pipeline")
    def test_delegates_to_pipeline_with_string_path(self, mock_pipeline, service):
        mock_pipeline.return_value = {"pages": 3, "chunks": 42}

        result = service.ingest(DEFAULT_TENANT, Path("data/raw/doc.pdf"))

        mock_pipeline.assert_called_once_with("data/raw/doc.pdf")
        assert result == {"pages": 3, "chunks": 42}

    @patch("src.services.rag_service._ingest_pipeline")
    def test_accepts_str_path(self, mock_pipeline, service, tmp_path):
        mock_pipeline.return_value = {"pages": 1, "chunks": 2}
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")

        service.ingest("tenant-a", str(pdf))

        mock_pipeline.assert_called_once_with(str(pdf))


class TestQuery:
    @patch("src.services.rag_service.generate")
    def test_forwards_query_bm25_and_overrides(self, mock_generate, service):
        mock_generate.return_value = CitedAnswer(
            answer="[SOURCE 1]", sources=[Source(doc_id="d1", page_num=1, text="x")]
        )
        bm25 = MagicMock()
        overrides = ProviderOverrides(openai_api_key="sk-test", openai_model="gpt-4o")

        result = service.query(DEFAULT_TENANT, "what is X?", bm25, overrides)

        mock_generate.assert_called_once_with(
            "what is X?", bm25, provider_overrides=overrides
        )
        assert isinstance(result, CitedAnswer)

    @patch("src.services.rag_service.generate")
    def test_overrides_default_to_none(self, mock_generate, service):
        mock_generate.return_value = CitedAnswer(answer="[SOURCE 1]", sources=[])
        bm25 = MagicMock()

        service.query(DEFAULT_TENANT, "q", bm25)

        mock_generate.assert_called_once_with("q", bm25, provider_overrides=None)


class TestListDocs:
    @patch("src.services.rag_service.get_collection")
    def test_returns_sorted_distinct_doc_ids(self, mock_get_collection, service):
        collection = MagicMock()
        collection.get.return_value = {
            "metadatas": [
                {"doc_id": "b.pdf", "page_num": 1},
                {"doc_id": "a.pdf", "page_num": 1},
                {"doc_id": "b.pdf", "page_num": 2},
                {"page_num": 3},
                None,
            ]
        }
        mock_get_collection.return_value = collection

        assert service.list_docs(DEFAULT_TENANT) == ["a.pdf", "b.pdf"]

    @patch("src.services.rag_service.get_collection")
    def test_empty_store_returns_empty_list(self, mock_get_collection, service):
        collection = MagicMock()
        collection.get.return_value = {"metadatas": []}
        mock_get_collection.return_value = collection

        assert service.list_docs(DEFAULT_TENANT) == []


class TestDelete:
    @patch("src.services.rag_service.get_collection")
    def test_deletes_by_doc_id_where_filter(self, mock_get_collection, service):
        collection = MagicMock()
        mock_get_collection.return_value = collection

        service.delete(DEFAULT_TENANT, "doc.pdf")

        collection.delete.assert_called_once_with(where={"doc_id": "doc.pdf"})

    @patch("src.services.rag_service.get_collection")
    def test_tenant_id_is_accepted_but_not_enforced_yet(self, mock_get_collection, service):
        """PR-1 adds tenant_id to the where filter; until then shared corpus."""
        collection = MagicMock()
        mock_get_collection.return_value = collection

        service.delete("tenant-b", "doc.pdf")

        collection.delete.assert_called_once_with(where={"doc_id": "doc.pdf"})
