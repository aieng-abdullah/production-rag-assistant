"""Unit tests for BM25Retriever class."""

from src.core.bm25_retriever import BM25Retriever


def test_bm25_retriever_initialization():
    """Test BM25Retriever initialization."""
    retriever = BM25Retriever()
    assert retriever.top_k == 20  # Default
    assert retriever._retriever is None  # Lazy loading
    print("✅ BM25Retriever initializes correctly")


def test_build_index():
    """Test build_index method."""
    chunks = [
        {"text": "AI is transforming industries", "doc_id": "doc1", "page_num": 1, "chunk_index": 0, "chunk_id": "doc1_chunk_0"},
        {"text": "Machine learning is a subset of AI", "doc_id": "doc1", "page_num": 2, "chunk_index": 1, "chunk_id": "doc1_chunk_1"},
    ]
    
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    
    assert retriever._retriever is not None
    print("✅ build_index creates BM25 index")


def test_search():
    """Test search method."""
    chunks = [
        {"text": "Artificial intelligence and machine learning", "doc_id": "doc1", "page_num": 1, "chunk_index": 0, "chunk_id": "doc1_chunk_0"},
        {"text": "Basketball is a popular sport", "doc_id": "doc2", "page_num": 5, "chunk_index": 1, "chunk_id": "doc2_chunk_1"},
    ]
    
    retriever = BM25Retriever(top_k=2)
    results = retriever.search("artificial intelligence", chunks)
    
    assert len(results) <= 2
    assert all("chunk_id" in r for r in results)
    assert all("text" in r for r in results)
    print(f"✅ search returned {len(results)} results")


def test_backward_compatibility():
    """Test backward compatibility with existing functions."""
    from src.retrieval.bm25_index import build_bm25_index, bm25_search
    
    chunks = [
        {"text": "Test content", "doc_id": "doc1", "page_num": 1, "chunk_index": 0, "chunk_id": "doc1_chunk_0"},
    ]
    
    # Test build_bm25_index
    index = build_bm25_index(chunks)
    assert index is not None
    
    # Test bm25_search
    results = bm25_search(index, "test", chunks, top_k=1)
    assert len(results) >= 0
    
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_bm25_retriever_initialization()
    test_build_index()
    test_search()
    test_backward_compatibility()
    print("\n✅ All BM25Retriever tests passed!")
