from src.retrieval.hybrid_fusion import rrf_fusion

def test_rrf_returns_top_k():
    bm25 = [{"chunk_id": "a", "text": "hi"}, {"chunk_id": "b", "text": "bye"}]
    vector = [{"chunk_id": "b", "text": "bye"}, {"chunk_id": "a", "text": "hi"}]
    result = rrf_fusion(bm25, vector, top_k=1)
    assert len(result) == 1

def test_rrf_has_score():
    bm25 = [{"chunk_id": "a", "text": "hi"}]
    vector = [{"chunk_id": "a", "text": "hi"}]
    result = rrf_fusion(bm25, vector, top_k=1)
    assert "rrf_score" in result[0]
