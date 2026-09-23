"""Tests for ingestion pipeline - using real chunk_pages instead of mocking it."""
import pytest
from unittest.mock import patch, MagicMock

from src.ingestion.pipeline import ingest
from src.ingestion.chunker import chunk_pages


class TestIngestSuccess:
    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_full_pipeline_returns_counts(self, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1, "doc_id": "d1"}]
        mock_embed.return_value = [{"text": "chunk", "embedding": [0.1]}]
        mock_upsert.return_value = 3

        result = ingest("/fake/doc.pdf")

        assert result == {"pages": 1, "chunks": 3}

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_real_chunker_is_used(self, mock_extract, mock_embed, mock_upsert):
        """Verify that real chunk_pages runs, not a mock."""
        page_text = "This is a test paragraph with enough text to trigger chunking. " * 10
        mock_extract.return_value = [{"text": page_text, "page_num": 1, "doc_id": "d1"}]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf")

        embedded_chunks = mock_embed.call_args[0][0]
        assert len(embedded_chunks) > 0
        for chunk in embedded_chunks:
            assert "text" in chunk
            assert "chunk_id" in chunk
            assert "page_num" in chunk

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_chunk_metadata_preserved_through_pipeline(self, mock_extract, mock_embed, mock_upsert):
        """Verify doc_id and page_num survive from extract through chunk to embed."""
        mock_extract.return_value = [
            {"text": "A" * 200, "page_num": 3, "doc_id": "doc_abc"},
            {"text": "B" * 200, "page_num": 7, "doc_id": "doc_abc"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf")

        embedded_chunks = mock_embed.call_args[0][0]
        page_nums = {c["page_num"] for c in embedded_chunks}
        assert 3 in page_nums
        assert 7 in page_nums

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_empty_pages_skipped_by_chunker(self, mock_extract, mock_embed, mock_upsert):
        """Verify that empty pages from extract_pages are filtered by chunk_pages."""
        mock_extract.return_value = [
            {"text": "", "page_num": 1, "doc_id": "d1"},
            {"text": "A" * 200, "page_num": 2, "doc_id": "d1"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf")

        embedded_chunks = mock_embed.call_args[0][0]
        page_nums = {c["page_num"] for c in embedded_chunks}
        assert 1 not in page_nums
        assert 2 in page_nums

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_chunks_have_total_chunks_metadata(self, mock_extract, mock_embed, mock_upsert):
        """Verify chunk_pages adds total_chunks metadata."""
        mock_extract.return_value = [
            {"text": "A" * 200, "page_num": 1, "doc_id": "d1"},
            {"text": "B" * 200, "page_num": 2, "doc_id": "d1"},
        ]
        mock_embed.return_value = []
        mock_upsert.return_value = 0

        ingest("/fake/doc.pdf")

        embedded_chunks = mock_embed.call_args[0][0]
        for chunk in embedded_chunks:
            assert "total_chunks" in chunk
            assert chunk["total_chunks"] == len(embedded_chunks)


class TestIngestErrors:
    @patch("src.ingestion.pipeline.extract_pages")
    def test_file_not_found(self, mock_extract):
        mock_extract.side_effect = FileNotFoundError("not found")
        with pytest.raises(FileNotFoundError):
            ingest("/missing/file.pdf")

    @patch("src.ingestion.pipeline.extract_pages")
    def test_extraction_error(self, mock_extract):
        mock_extract.side_effect = RuntimeError("extraction failed")
        with pytest.raises(RuntimeError, match="Failed to extract pages"):
            ingest("/fake/doc.pdf")

    @patch("src.ingestion.pipeline.extract_pages")
    def test_no_pages(self, mock_extract):
        mock_extract.return_value = []
        with pytest.raises(ValueError, match="No pages extracted"):
            ingest("/fake/doc.pdf")

    @patch("src.ingestion.pipeline.chunk_pages")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_chunking_error(self, mock_extract, mock_chunk):
        mock_extract.return_value = [{"text": "page", "page_num": 1}]
        mock_chunk.side_effect = RuntimeError("chunk failed")
        with pytest.raises(RuntimeError, match="Failed to chunk pages"):
            ingest("/fake/doc.pdf")

    @patch("src.ingestion.pipeline.chunk_pages")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_no_chunks(self, mock_extract, mock_chunk):
        mock_extract.return_value = [{"text": "page", "page_num": 1}]
        mock_chunk.return_value = []
        with pytest.raises(ValueError, match="No chunks created"):
            ingest("/fake/doc.pdf")

    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.chunk_pages")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_embedding_error(self, mock_extract, mock_chunk, mock_embed):
        mock_extract.return_value = [{"text": "page", "page_num": 1}]
        mock_chunk.return_value = [{"text": "chunk", "page_num": 1}]
        mock_embed.side_effect = RuntimeError("embed failed")
        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            ingest("/fake/doc.pdf")

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.chunk_pages")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_storage_error(self, mock_extract, mock_chunk, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "page", "page_num": 1}]
        mock_chunk.return_value = [{"text": "chunk", "page_num": 1}]
        mock_embed.return_value = [{"text": "chunk", "page_num": 1}]
        mock_upsert.side_effect = RuntimeError("storage failed")
        with pytest.raises(RuntimeError, match="Failed to store chunks"):
            ingest("/fake/doc.pdf")
