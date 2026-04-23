import pytest
from unittest.mock import patch

from src.ingestion.pipeline import ingest


# ✅ SUCCESS CASE
@patch("src.ingestion.pipeline.upsert_chunks")
@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_success(
    mock_extract,
    mock_chunk,
    mock_embed,
    mock_upsert,
):
    mock_extract.return_value = [{"text": "page1"}]
    mock_chunk.return_value = [{"chunk": "chunk1"}]
    mock_embed.return_value = [{"embedding": [0.1, 0.2]}]
    mock_upsert.return_value = 1

    result = ingest("fake.pdf")

    assert result["pages"] == 1
    assert result["chunks"] == 1


# ❌ FILE NOT FOUND
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_file_not_found(mock_extract):
    mock_extract.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        ingest("missing.pdf")


# ❌ EMPTY PAGES
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_empty_pages(mock_extract):
    mock_extract.return_value = []

    with pytest.raises(ValueError, match="No pages"):
        ingest("fake.pdf")


# ❌ CHUNKING FAIL
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_chunking_failure(mock_extract, mock_chunk):
    mock_extract.return_value = [{"text": "page"}]
    mock_chunk.side_effect = Exception("chunk error")

    with pytest.raises(RuntimeError, match="Failed to chunk"):
        ingest("fake.pdf")


# ❌ EMPTY CHUNKS
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_empty_chunks(mock_extract, mock_chunk):
    mock_extract.return_value = [{"text": "page"}]
    mock_chunk.return_value = []

    with pytest.raises(RuntimeError):
        ingest("fake.pdf")


# ❌ EMBEDDING FAIL
@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_embedding_failure(mock_extract, mock_chunk, mock_embed):
    mock_extract.return_value = [{"text": "page"}]
    mock_chunk.return_value = [{"chunk": "c"}]
    mock_embed.side_effect = Exception("embed fail")

    with pytest.raises(ValueError, match="Embedding failed"):
        ingest("fake.pdf")


# ❌ UPSERT FAIL
@patch("src.ingestion.pipeline.upsert_chunks")
@patch("src.ingestion.pipeline.embed_chunks")
@patch("src.ingestion.pipeline.chunk_pages")
@patch("src.ingestion.pipeline.extract_pages")
def test_ingest_upsert_failure(
    mock_extract,
    mock_chunk,
    mock_embed,
    mock_upsert,
):
    mock_extract.return_value = [{"text": "page"}]
    mock_chunk.return_value = [{"chunk": "c"}]
    mock_embed.return_value = [{"embedding": [0.1]}]
    mock_upsert.side_effect = ValueError

    with pytest.raises(ValueError, match="error while saivinf"):
        ingest("fake.pdf")
