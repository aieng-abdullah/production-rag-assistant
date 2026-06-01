"""Tests for retrieval pipeline."""

from unittest.mock import patch, MagicMock
import pytest


@patch("src.retrieval.pipeline.rerank")
@patch("src.retrieval.pipeline.rrf_fusion")
@patch("src.retrieval.pipeline.vector_search")
@patch("src.retrieval.pipeline.bm25_search")
def test_retrieval_success(mock_bm25, mock_vector, mock_rrf, mock_rerank):
    mock_bm25.return_value = [{"chunk_id": "c1", "text": "a"}]
    mock_vector.return_value = [{"chunk_id": "c1", "text": "a"}]
    mock_rrf.return_value = [{"chunk_id": "c1", "text": "a"}]
    mock_rerank.return_value = [{"chunk_id": "c1", "text": "a", "rerank_score": 0.9}]

    from src.retrieval.pipeline import retrieval
    result = retrieval("query", MagicMock(), top_k=5)
    assert len(result) == 1
    assert result[0]["rerank_score"] == 0.9


@patch("src.retrieval.pipeline.bm25_search")
def test_retrieval_bm25_error(mock_bm25):
    mock_bm25.side_effect = RuntimeError("bm25 failed")
    from src.retrieval.pipeline import retrieval
    with pytest.raises(RuntimeError, match="Error while BM25 search"):
        retrieval("query", MagicMock())


@patch("src.retrieval.pipeline.vector_search")
@patch("src.retrieval.pipeline.bm25_search")
def test_retrieval_vector_error(mock_bm25, mock_vector):
    mock_bm25.return_value = []
    mock_vector.side_effect = RuntimeError("vector failed")
    from src.retrieval.pipeline import retrieval
    with pytest.raises(RuntimeError, match="Error while vector search"):
        retrieval("query", MagicMock())


@patch("src.retrieval.pipeline.rrf_fusion")
@patch("src.retrieval.pipeline.vector_search")
@patch("src.retrieval.pipeline.bm25_search")
def test_retrieval_rrf_error(mock_bm25, mock_vector, mock_rrf):
    mock_bm25.return_value = []
    mock_vector.return_value = []
    mock_rrf.side_effect = RuntimeError("rrf failed")
    from src.retrieval.pipeline import retrieval
    with pytest.raises(RuntimeError, match="Error while RRF fusion"):
        retrieval("query", MagicMock())


@patch("src.retrieval.pipeline.rerank")
@patch("src.retrieval.pipeline.rrf_fusion")
@patch("src.retrieval.pipeline.vector_search")
@patch("src.retrieval.pipeline.bm25_search")
def test_retrieval_rerank_error(mock_bm25, mock_vector, mock_rrf, mock_rerank):
    mock_bm25.return_value = []
    mock_vector.return_value = []
    mock_rrf.return_value = []
    mock_rerank.side_effect = RuntimeError("rerank failed")
    from src.retrieval.pipeline import retrieval
    with pytest.raises(RuntimeError, match="Error while reranking"):
        retrieval("query", MagicMock())
