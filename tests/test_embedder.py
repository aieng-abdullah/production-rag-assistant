"""Tests for the Voyage embedder (API-only — HTTP mocked, no network)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ingestion import embedder
from src.ingestion.embedder import _get_model, embed_chunks, embed_query


def _response(vectors: list[list[float]], input_type: str | None = None) -> MagicMock:
    """Fake Voyage /embeddings response (index order shuffled on purpose)."""
    data = [
        {"index": i, "embedding": vector}
        for i, vector in reversed(list(enumerate(vectors)))
    ]
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = "{}"
    resp.json.return_value = {"data": data, "usage": {"total_tokens": 7}}
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def fast_retry():
    """No-op tenacity sleep so retry-exhaustion tests stay fast."""
    original = embedder._post_embeddings.retry.sleep
    embedder._post_embeddings.retry.sleep = lambda *args, **kwargs: None
    yield
    embedder._post_embeddings.retry.sleep = original


def _echo_vectors(*vectors: list[float]):
    """httpx.post side effect: return one vector per requested input."""
    def _side_effect(*args, **kwargs):
        inputs = kwargs["json"]["input"]
        return _response([vectors[i % len(vectors)] for i in range(len(inputs))])
    return _side_effect


@patch("src.ingestion.embedder.httpx.post")
def test_embed_query(mock_post):
    mock_post.return_value = _response([[3.0, 4.0]])

    embedding = embed_query("This is a test query for embedding.")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

    # L2-normalized (parity with the local model's normalize_embeddings=True)
    import math

    norm = math.sqrt(sum(x * x for x in embedding))
    assert 0.99 < norm < 1.01, f"Embedding not normalized, norm={norm}"

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["input_type"] == "query"
    assert kwargs["json"]["input"] == ["This is a test query for embedding."]


@patch("src.ingestion.embedder.httpx.post")
def test_embed_chunks_batches_and_preserves_metadata(mock_post):
    from src.config import Config

    mock_post.side_effect = _echo_vectors([0.6, 0.8], [1.0, 0.0])
    chunks = [
        {"text": "First test chunk about AI technology.", "doc_id": "doc1", "chunk_id": "c1"},
        {"text": "Second test chunk about machine learning.", "doc_id": "doc1", "chunk_id": "c2"},
        {"text": "Third test chunk about deep learning.", "doc_id": "doc2", "chunk_id": "c3"},
    ]

    result = embed_chunks(chunks, batch_size=2)

    assert len(result) == 3
    assert mock_post.call_count == 2  # 3 texts at batch_size=2 → 2 requests

    for call in mock_post.call_args_list:
        assert call.kwargs["json"]["input_type"] == "document"
    assert mock_post.call_args_list[0].kwargs["json"]["input"] == [
        "First test chunk about AI technology.",
        "Second test chunk about machine learning.",
    ]
    assert mock_post.call_args_list[1].kwargs["json"]["input"] == [
        "Third test chunk about deep learning."
    ]
    assert mock_post.call_args_list[0].kwargs["timeout"] == Config.EMBED_TIMEOUT_S

    for chunk in result:
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) > 0
        assert "doc_id" in chunk
        assert "chunk_id" in chunk

    dims = [len(c["embedding"]) for c in result]
    assert all(d == dims[0] for d in dims)


@patch("src.ingestion.embedder.httpx.post")
def test_embed_chunks_empty_makes_no_call(mock_post):
    assert embed_chunks([]) == []
    mock_post.assert_not_called()


@patch("src.ingestion.embedder.httpx.post")
def test_embed_query_normalizes_zero_vector_safely(mock_post):
    mock_post.return_value = _response([[0.0, 0.0]])

    embedding = embed_query("zero")
    assert embedding == [0.0, 0.0]


@patch("src.ingestion.embedder.httpx.post")
def test_embed_count_mismatch_raises(mock_post):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = "{}"
    resp.json.return_value = {"data": [{"index": 0, "embedding": [1.0]}]}
    mock_post.return_value = resp

    with pytest.raises(RuntimeError, match="embeddings for 2 inputs"):
        embed_chunks([{"text": "a"}, {"text": "b"}])


@patch("src.ingestion.embedder.httpx.post")
def test_embed_client_error_propagates(mock_post):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 401
    resp.text = "bad key"
    resp.json.return_value = {}
    mock_post.return_value = resp

    with pytest.raises(RuntimeError, match="Voyage embeddings rejected request"):
        embed_query("x")


@patch("src.ingestion.embedder.httpx.post")
def test_embed_transport_error_retries_then_propagates(mock_post, fast_retry):
    mock_post.side_effect = httpx.ConnectError("connection refused")

    with pytest.raises(httpx.HTTPError):
        embed_query("x")
    assert mock_post.call_count == 3  # initial attempt + 2 retries


@patch("src.ingestion.embedder.Config")
def test_embed_missing_key_raises(mock_config):
    mock_config.VOYAGE_API_KEY = ""
    mock_config.VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
    mock_config.VOYAGE_EMBEDDING_MODEL = "voyage-4-lite"
    mock_config.EMBED_TIMEOUT_S = 30
    mock_config.VOYAGE_EMBED_BATCH_SIZE = 128

    with pytest.raises(EnvironmentError, match="VOYAGE_API_KEY is not set"):
        embed_query("x")


def test_model_singleton():
    """Client is lazy-loaded and reused (no heavyweight model to load)."""
    embedder._model = None
    try:
        model1 = _get_model()
        model2 = _get_model()
        assert model1 is model2
    finally:
        embedder._model = None


@pytest.mark.slow
@pytest.mark.skipif(
    not embedder.Config.VOYAGE_API_KEY, reason="VOYAGE_API_KEY not configured"
)
def test_embed_query_live():
    """Live API check: format + normalization + semantic ordering."""
    import math

    embedding = embed_query("This is a test query for embedding.")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    norm = math.sqrt(sum(x * x for x in embedding))
    assert 0.99 < norm < 1.01, f"Embedding not normalized, norm={norm}"

    text1 = "Artificial intelligence and machine learning"
    text2 = "Machine learning and AI technologies"
    text3 = "Basketball is a popular sport"
    emb1, emb2, emb3 = embed_query(text1), embed_query(text2), embed_query(text3)

    cosine_sim = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    sim_1_2 = cosine_sim(emb1, emb2)
    sim_1_3 = cosine_sim(emb1, emb3)
    assert sim_1_2 > sim_1_3, (
        f"Similar texts should have higher similarity: {sim_1_2} vs {sim_1_3}"
    )
