"""Tests for ingestion pipeline."""

from unittest.mock import patch, MagicMock
import pytest


@patch("src.ingestion.pipeline.upsert_chunks")
@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_success(mock_extract, mock_chunk, mock_embed, mock_upsert):
    mock_extract.return_value = [{"text": "page", "page_num": 1}]
    mock_chunk.return_value = [{"text": "chunk", "page_num": 1}]
    mock_embed.return_value = [{"text": "chunk", "page_num": 1, "embedding": [0.1]}]
    mock_upsert.return_value = 1

    from src.ingestion.pipeline import ingest
    result = ingest("/fake/doc.pdf")
    assert result == {"pages": 1, "chunks": 1}


@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_file_not_found(mock_extract):
    mock_extract.side_effect = FileNotFoundError("not found")
    from src.ingestion.pipeline import ingest
    with pytest.raises(FileNotFoundError):
        ingest("/missing/file.pdf")


@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_extraction_error(mock_extract):
    mock_extract.side_effect = RuntimeError("extraction failed")
    from src.ingestion.pipeline import ingest
    with pytest.raises(RuntimeError, match="Failed to extract pages"):
        ingest("/fake/doc.pdf")


@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_no_pages(mock_extract):
    mock_extract.return_value = []
    from src.ingestion.pipeline import ingest
    with pytest.raises(ValueError, match="No pages extracted"):
        ingest("/fake/doc.pdf")


@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_chunking_error(mock_extract, mock_chunk):
    mock_extract.return_value = [{"text": "page", "page_num": 1}]
    mock_chunk.side_effect = RuntimeError("chunk failed")
    from src.ingestion.pipeline import ingest
    with pytest.raises(RuntimeError, match="Failed to chunk pages"):
        ingest("/fake/doc.pdf")


@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_no_chunks(mock_extract, mock_chunk):
    mock_extract.return_value = [{"text": "page", "page_num": 1}]
    mock_chunk.return_value = []
    from src.ingestion.pipeline import ingest
    with pytest.raises(ValueError, match="No chunks created"):
        ingest("/fake/doc.pdf")


@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_embedding_error(mock_extract, mock_chunk, mock_embed):
    mock_extract.return_value = [{"text": "page", "page_num": 1}]
    mock_chunk.return_value = [{"text": "chunk", "page_num": 1}]
    mock_embed.side_effect = RuntimeError("embed failed")
    from src.ingestion.pipeline import ingest
    with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
        ingest("/fake/doc.pdf")


@patch("src.ingestion.pipeline.upsert_chunks")
@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_storage_error(mock_extract, mock_chunk, mock_embed, mock_upsert):
    mock_extract.return_value = [{"text": "page", "page_num": 1}]
    mock_chunk.return_value = [{"text": "chunk", "page_num": 1}]
    mock_embed.return_value = [{"text": "chunk", "page_num": 1}]
    mock_upsert.side_effect = RuntimeError("storage failed")
    from src.ingestion.pipeline import ingest
    with pytest.raises(RuntimeError, match="Failed to store chunks"):
        ingest("/fake/doc.pdf")
