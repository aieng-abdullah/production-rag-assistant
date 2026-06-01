"""Tests for cross-encoder reranking."""

from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def reset_model():
    import src.retrieval.cross_encoder as mod
    mod._model = None
    yield
    mod._model = None


@patch("src.retrieval.cross_encoder.CrossEncoder")
def test_rerank_basic(mock_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3, 0.7]
    mock_cls.return_value = mock_model

    from src.retrieval.cross_encoder import rerank
    chunks = [
        {"text": "first", "doc_id": "d1", "page_num": 1},
        {"text": "second", "doc_id": "d2", "page_num": 2},
        {"text": "third", "doc_id": "d3", "page_num": 3},
    ]
    result = rerank("query", chunks, top_k=2)
    assert len(result) == 2
    assert result[0]["rerank_score"] == 0.9
    assert result[0]["text"] == "first"
    assert result[1]["rerank_score"] == 0.7


@patch("src.retrieval.cross_encoder.CrossEncoder")
def test_rerank_empty_chunks(mock_cls):
    from src.retrieval.cross_encoder import rerank
    result = rerank("query", [], top_k=5)
    assert result == []


def test_rerank_invalid_top_k():
    from src.retrieval.cross_encoder import rerank
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rerank("query", [{"text": "x"}], top_k=0)


@patch("src.retrieval.cross_encoder.CrossEncoder")
def test_rerank_with_threshold(mock_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.1]
    mock_cls.return_value = mock_model

    from src.retrieval.cross_encoder import rerank
    chunks = [{"text": t} for t in ["a", "b", "c", "d"]]
    result = rerank("query", chunks, top_k=4, score_threshold=0.5)
    assert len(result) == 2
    scores = [r["rerank_score"] for r in result]
    assert all(s >= 0.5 for s in scores)


@patch("src.retrieval.cross_encoder.CrossEncoder")
def test_rerank_inference_error(mock_cls):
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("model error")
    mock_cls.return_value = mock_model

    from src.retrieval.cross_encoder import rerank
    with pytest.raises(RuntimeError):
        rerank("query", [{"text": "x"}], top_k=1)


@patch("src.retrieval.cross_encoder.CrossEncoder")
def test_rerank_model_singleton(mock_cls):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5]
    mock_cls.return_value = mock_model

    from src.retrieval.cross_encoder import rerank, _get_model
    m1 = _get_model()
    m2 = _get_model()
    assert m1 is m2
