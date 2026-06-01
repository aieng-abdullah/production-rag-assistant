"""Tests for RRF fusion edge cases (overlapping chunk_ids)."""

from src.retrieval.hybrid_fusion import rrf_fusion


def test_rrf_overlapping_chunk_ids():
    bm25 = [{"chunk_id": "a", "text": "from bm25"}, {"chunk_id": "b", "text": "only bm25"}]
    vector = [{"chunk_id": "a", "text": "from vector"}, {"chunk_id": "c", "text": "only vector"}]
    result = rrf_fusion(bm25, vector, top_k=3)
    assert len(result) == 3
    chunk_ids = [r["chunk_id"] for r in result]
    assert "a" in chunk_ids
    assert "b" in chunk_ids
    assert "c" in chunk_ids


def test_rrf_overlapping_uses_bm25_text():
    bm25 = [{"chunk_id": "a", "text": "bm25 version"}]
    vector = [{"chunk_id": "a", "text": "vector version"}]
    result = rrf_fusion(bm25, vector, top_k=1)
    assert result[0]["text"] == "bm25 version"


def test_rrf_vector_only_chunks():
    bm25 = []
    vector = [{"chunk_id": "x", "text": "vec1"}, {"chunk_id": "y", "text": "vec2"}]
    result = rrf_fusion(bm25, vector, top_k=2)
    assert len(result) == 2
    assert result[0]["rrf_score"] >= result[1]["rrf_score"]


def test_rrf_bm25_only_chunks():
    bm25 = [{"chunk_id": "x", "text": "bm1"}, {"chunk_id": "y", "text": "bm2"}]
    vector = []
    result = rrf_fusion(bm25, vector, top_k=2)
    assert len(result) == 2
