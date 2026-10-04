"""Tests for the RAGService application-service facade (PLAN.md PR-0 + PR-1)."""

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
    def test_delegates_to_pipeline_with_tenant(self, mock_pipeline, service):
        mock_pipeline.return_value = {"pages": 3, "chunks": 42}

        result = service.ingest("tenant-a", Path("data/raw/doc.pdf"))

        mock_pipeline.assert_called_once_with(
            "data/raw/doc.pdf", tenant_id="tenant-a", workspace="academic"
        )
        assert result == {"pages": 3, "chunks": 42}

    @patch("src.services.rag_service._ingest_pipeline")
    def test_accepts_str_path(self, mock_pipeline, service, tmp_path):
        mock_pipeline.return_value = {"pages": 1, "chunks": 2}
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")

        service.ingest("tenant-a", str(pdf))

        mock_pipeline.assert_called_once_with(
            str(pdf), tenant_id="tenant-a", workspace="academic"
        )


class TestGenerateAnswer:
    @patch("src.services.rag_service.generate")
    def test_forwards_query_bm25_overrides_and_tenant(self, mock_generate, service):
        mock_generate.return_value = CitedAnswer(
            answer="[SOURCE 1]", sources=[Source(doc_id="d1", page_num=1, text="x")]
        )
        bm25 = MagicMock()
        overrides = ProviderOverrides(openai_api_key="sk-test", openai_model="gpt-4o")

        result = service.generate_answer(
            "tenant-a", "what is X?", bm25, overrides
        )

        mock_generate.assert_called_once_with(
            "what is X?",
            bm25,
            provider_overrides=overrides,
            tenant_id="tenant-a",
            workspace="academic",
        )
        assert isinstance(result, CitedAnswer)

    @patch("src.services.rag_service.generate")
    def test_defaults_overrides_to_none(self, mock_generate, service):
        mock_generate.return_value = CitedAnswer(answer="[SOURCE 1]", sources=[])
        bm25 = MagicMock()

        service.generate_answer(DEFAULT_TENANT, "q", bm25)

        mock_generate.assert_called_once_with(
            "q", bm25, provider_overrides=None, tenant_id=DEFAULT_TENANT,
            workspace="academic",
        )


class TestListDocuments:
    @patch("src.services.rag_service.get_collection")
    def test_queries_with_tenant_predicate(self, mock_get_collection, service):
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

        assert service.list_documents("tenant-a") == ["a.pdf", "b.pdf"]
        collection.get.assert_called_once_with(where={"tenant_id": "tenant-a"})

    @patch("src.services.rag_service.get_collection")
    def test_empty_tenant_returns_empty_list(self, mock_get_collection, service):
        collection = MagicMock()
        collection.get.return_value = {"metadatas": []}
        mock_get_collection.return_value = collection

        assert service.list_documents("tenant-b") == []
        collection.get.assert_called_once_with(where={"tenant_id": "tenant-b"})


class TestDeleteDocument:
    @patch("src.services.rag_service.get_collection")
    def test_predicate_requires_tenant_and_doc_match(self, mock_get_collection, service):
        collection = MagicMock()
        mock_get_collection.return_value = collection

        service.delete_document("tenant-a", "doc.pdf")

        collection.delete.assert_called_once_with(
            where={
                "$and": [
                    {"tenant_id": {"$eq": "tenant-a"}},
                    {"doc_id": {"$eq": "doc.pdf"}},
                ]
            }
        )

    @patch("src.services.rag_service.get_collection")
    def test_tenant_predicate_differs_per_tenant(self, mock_get_collection, service):
        """B deleting same doc_id must target B's partition only."""
        collection = MagicMock()
        mock_get_collection.return_value = collection

        service.delete_document("tenant-b", "doc.pdf")

        args = collection.delete.call_args.kwargs
        assert {"$and": [{"tenant_id": {"$eq": "tenant-b"}}, {"doc_id": {"$eq": "doc.pdf"}}]} == args["where"]
