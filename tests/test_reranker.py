"""Tests for Voyage API reranking (mocked HTTP — no network in the fast suite)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.retrieval import reranker
from src.retrieval.reranker import rerank


def _response(payload: dict, status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response with JSON + status."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = str(payload)
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _results(*pairs: tuple[int, float]) -> dict:
    """Voyage /rerank payload from (index, relevance_score) pairs."""
    return {
        "results": [
            {"index": index, "relevance_score": score} for index, score in pairs
        ],
        "usage": {"total_tokens": 42},
    }


@pytest.fixture
def fast_retry():
    """No-op tenacity sleep so retry-exhaustion tests stay fast."""
    original = reranker._post_rerank.retry.sleep
    reranker._post_rerank.retry.sleep = lambda *args, **kwargs: None
    yield
    reranker._post_rerank.retry.sleep = original


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_basic(mock_post):
    mock_post.return_value = _response(_results((0, 0.9), (2, 0.7), (1, 0.3)))

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
    assert result[1]["text"] == "third"


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_sends_model_and_documents_in_order(mock_post):
    mock_post.return_value = _response(_results((0, 0.4), (1, 0.8)))

    from src.config import Config

    rerank("my query", [{"text": "alpha"}, {"text": "beta"}], top_k=2)

    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == Config.VOYAGE_RERANKER_MODEL
    assert payload["query"] == "my query"
    assert payload["documents"] == ["alpha", "beta"]
    assert kwargs["headers"]["Authorization"] == f"Bearer {Config.VOYAGE_API_KEY}"
    assert kwargs["timeout"] == Config.RERANK_TIMEOUT_S


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_empty_chunks(mock_post):
    result = rerank("query", [], top_k=5)
    assert result == []
    mock_post.assert_not_called()


def test_rerank_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rerank("query", [{"text": "x"}], top_k=0)


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_with_threshold(mock_post):
    # Voyage scores are in [0, 1] — threshold filters before the top_k slice.
    mock_post.return_value = _response(_results((0, 0.9), (1, 0.3), (2, 0.7), (3, 0.1)))

    chunks = [{"text": t} for t in ["a", "b", "c", "d"]]
    result = rerank("query", chunks, top_k=4, score_threshold=0.5)
    assert len(result) == 2
    scores = [r["rerank_score"] for r in result]
    assert all(s >= 0.5 for s in scores)


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_client_error_propagates(mock_post):
    mock_post.return_value = _response({"detail": "bad api key"}, status_code=401)

    with pytest.raises(RuntimeError, match="Voyage rerank rejected request"):
        rerank("query", [{"text": "x"}], top_k=1)


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_transport_error_retries_then_propagates(mock_post, fast_retry):
    mock_post.side_effect = httpx.ConnectError("connection refused")

    with pytest.raises(httpx.HTTPError):
        rerank("query", [{"text": "x"}], top_k=1)
    assert mock_post.call_count == 3  # initial attempt + 2 retries


@patch("src.retrieval.reranker.Config")
def test_rerank_missing_key_raises(mock_config):
    mock_config.VOYAGE_API_KEY = ""
    mock_config.VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
    mock_config.VOYAGE_RERANKER_MODEL = "rerank-3-lite"
    mock_config.RERANK_TIMEOUT_S = 10

    with pytest.raises(EnvironmentError, match="VOYAGE_API_KEY is not set"):
        rerank("query", [{"text": "x"}], top_k=1)


@patch("src.retrieval.reranker.httpx.post")
def test_rerank_skips_malformed_results(mock_post):
    mock_post.return_value = _response(
        {
            "results": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 99, "relevance_score": 0.7},  # out of range
                "not-a-dict",  # malformed
                {"index": 0},  # missing score
            ]
        }
    )

    result = rerank("query", [{"text": "a"}, {"text": "b"}], top_k=5)
    assert len(result) == 1
    assert result[0]["text"] == "b"
    assert result[0]["rerank_score"] == 0.8
