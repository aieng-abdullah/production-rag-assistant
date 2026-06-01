"""Tests for ChromaDB client module."""

from unittest.mock import patch, MagicMock, PropertyMock
import pytest


@pytest.fixture(autouse=True)
def reset_vectorstore():
    import src.db.chroma_client as mod
    mod._vectorstore = None
    yield
    mod._vectorstore = None


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_get_vectorstore_init(mock_embed, mock_chroma):
    mock_embed.return_value = MagicMock()
    mock_chroma.return_value = MagicMock()
    from src.db.chroma_client import _get_vectorstore
    vs = _get_vectorstore()
    assert vs is not None
    mock_chroma.assert_called_once()


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_get_vectorstore_singleton(mock_embed, mock_chroma):
    mock_embed.return_value = MagicMock()
    mock_chroma.return_value = MagicMock()
    from src.db.chroma_client import _get_vectorstore
    vs1 = _get_vectorstore()
    vs2 = _get_vectorstore()
    assert vs1 is vs2


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_get_collection(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_collection = MagicMock()
    mock_vs._collection = mock_collection
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import get_collection
    result = get_collection()
    assert result is mock_collection


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_upsert_chunks(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import upsert_chunks
    chunks = [
        {"text": "hello", "doc_id": "d1", "page_num": 1, "chunk_index": 0, "chunk_id": "d1_c0"},
        {"text": "world", "doc_id": "d2", "page_num": 2, "chunk_index": 1, "chunk_id": "d2_c1"},
    ]
    count = upsert_chunks(chunks)
    assert count == 2
    mock_vs.add_documents.assert_called_once()


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_load_all_chunks(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["text1", "text2"],
        "metadatas": [
            {"doc_id": "d1", "chunk_index": 0, "page_num": 1},
            {"doc_id": "d2", "chunk_index": 1, "page_num": 2},
        ],
    }
    mock_vs._collection = mock_collection
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import load_all_chunks
    chunks = load_all_chunks()
    assert len(chunks) == 2
    assert chunks[0]["text"] == "text1"
    assert chunks[0]["chunk_id"] == "d1_chunk_0"


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_has_chunks_true(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_vs._collection = mock_collection
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import has_chunks
    assert has_chunks() is True


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_has_chunks_false(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_vs._collection = mock_collection
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import has_chunks
    assert has_chunks() is False


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_count_chunks(mock_embed, mock_chroma):
    mock_vs = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_vs._collection = mock_collection
    mock_chroma.return_value = mock_vs
    mock_embed.return_value = MagicMock()
    from src.db.chroma_client import count_chunks
    assert count_chunks() == 42


def test_reset_client():
    import src.db.chroma_client as mod
    mod._vectorstore = MagicMock()
    from src.db.chroma_client import reset_client
    reset_client()
    assert mod._vectorstore is None


@patch("src.db.chroma_client.Chroma")
@patch("src.db.chroma_client.get_embedding_model")
def test_get_vectorstore(mock_embed, mock_chroma):
    mock_embed.return_value = MagicMock()
    mock_chroma.return_value = MagicMock()
    from src.db.chroma_client import get_vectorstore
    vs = get_vectorstore()
    assert vs is not None
