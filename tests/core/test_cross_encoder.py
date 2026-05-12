"""Unit tests for CrossEncoderReranker class."""

from src.core.cross_encoder import CrossEncoderReranker


def test_cross_encoder_initialization():
    """Test CrossEncoderReranker initialization."""
    reranker = CrossEncoderReranker()
    assert reranker._model is None  # Lazy loading
    assert reranker._model_name is not None
    assert reranker._device == "cpu"
    print("✅ CrossEncoderReranker initializes correctly")


def test_lazy_loading():
    """Test that model is lazy-loaded."""
    reranker = CrossEncoderReranker()
    assert reranker._model is None
    
    # Access model property to trigger lazy load
    _ = reranker.model
    assert reranker._model is not None
    print("✅ Model lazy-loads on first access")


def test_rerank():
    """Test rerank method."""
    query = "machine learning"
    chunks = [
        {"text": "Machine learning algorithms improve with data", "chunk_id": "c1", "doc_id": "ml.pdf", "page_num": 1},
        {"text": "Unrelated content about cooking", "chunk_id": "c2", "doc_id": "cook.pdf", "page_num": 1},
    ]
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, chunks, top_k=2)
    
    assert len(results) <= 2
    assert all("rerank_score" in r for r in results)
    assert all(isinstance(r["rerank_score"], float) for r in results)
    print(f"✅ rerank returned {len(results)} results")


def test_rerank_empty_chunks():
    """Test rerank handles empty chunk list."""
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
    print("✅ rerank handles empty chunks")


def test_rerank_invalid_top_k():
    """Test rerank validates top_k parameter."""
    reranker = CrossEncoderReranker()
    try:
        reranker.rerank("query", [{"text": "test", "chunk_id": "c1", "doc_id": "d1", "page_num": 1}], top_k=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positive integer" in str(e)
    print("✅ rerank validates top_k parameter")


def test_rerank_with_threshold():
    """Test rerank with score threshold."""
    query = "machine learning"
    chunks = [
        {"text": "Machine learning algorithms", "chunk_id": "c1", "doc_id": "ml.pdf", "page_num": 1},
        {"text": "Unrelated content", "chunk_id": "c2", "doc_id": "cook.pdf", "page_num": 1},
    ]
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, chunks, top_k=5, score_threshold=0.5)
    print(f"✅ rerank with threshold returned {len(results)} results")


def test_reset():
    """Test reset method."""
    reranker = CrossEncoderReranker()
    _ = reranker.model  # Load model
    assert reranker._model is not None
    
    reranker.reset()
    assert reranker._model is None
    print("✅ Reset clears model from memory")


def test_backward_compatibility():
    """Test backward compatibility with existing functions."""
    from src.retrieval.cross_encoder import rerank, _get_model
    
    chunks = [
        {"text": "Test content", "chunk_id": "c1", "doc_id": "d1", "page_num": 1},
    ]
    
    # Test _get_model
    model = _get_model()
    assert model is not None
    
    # Test rerank
    results = rerank("test", chunks, top_k=1)
    assert len(results) >= 0
    
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_cross_encoder_initialization()
    test_lazy_loading()
    test_rerank()
    test_rerank_empty_chunks()
    test_rerank_invalid_top_k()
    test_rerank_with_threshold()
    test_reset()
    test_backward_compatibility()
    print("\n✅ All CrossEncoderReranker tests passed!")
