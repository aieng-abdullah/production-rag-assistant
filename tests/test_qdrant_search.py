"""Tests for vector search module."""

from unittest.mock import patch, MagicMock
import pytest


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_success(mock_embed, mock_get_col):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["text1", "text2"]],
        "metadatas": [[
            {"doc_id": "d1", "chunk_index": 0, "page_num": 1},
            {"doc_id": "d2", "chunk_index": 1, "page_num": 2},
        ]],
    }
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    results = vector_search("test query", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "text1"
    assert results[0]["chunk_id"] == "d1_chunk_0"
    assert results[1]["text"] == "text2"
    assert results[1]["chunk_id"] == "d2_chunk_1"


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_error(mock_embed, mock_get_col):
    mock_embed.return_value = [0.1]
    mock_collection = MagicMock()
    mock_collection.query.side_effect = RuntimeError("query failed")
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    with pytest.raises(RuntimeError, match="Error while vector search"):
        vector_search("test", top_k=5)


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_calls_embed(mock_embed, mock_get_col):
    mock_embed.return_value = [0.5]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    vector_search("my query", top_k=3)
    mock_embed.assert_called_once_with("my query")


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_applies_tenant_predicate(mock_embed, mock_get_col):
    """PR-1: collection.query must carry where={"tenant_id": ...}."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    vector_search("q", top_k=5, tenant_id="tenant-b")

    kwargs = mock_collection.query.call_args.kwargs
    assert kwargs["where"].must[0].key == "tenant_id"
    assert kwargs["where"].must[0].match.value == "tenant-b"


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_defaults_to_default_tenant(mock_embed, mock_get_col):
    mock_embed.return_value = [0.1]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    vector_search("q", top_k=5)

    kwargs = mock_collection.query.call_args.kwargs
    assert kwargs["where"].must[0].key == "tenant_id"
    assert kwargs["where"].must[0].match.value == "default"


@patch("src.retrieval.qdrant_search.get_collection")
@patch("src.retrieval.qdrant_search.embed_query")
def test_vector_search_adds_workspace_predicate(mock_embed, mock_get_col):
    """PR-4: workspace narrows the tenant predicate to one niche."""
    mock_embed.return_value = [0.1]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_get_col.return_value = mock_collection

    from src.retrieval.qdrant_search import vector_search
    vector_search("q", top_k=5, tenant_id="tenant-b", workspace="legal")

    kwargs = mock_collection.query.call_args.kwargs
    assert kwargs["where"].must[0].key == "tenant_id"
    assert kwargs["where"].must[0].match.value == "tenant-b"
    assert kwargs["where"].must[1].key == "workspace"
    assert kwargs["where"].must[1].match.value == "legal"
