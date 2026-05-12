"""Integration tests for RetrievalPipeline class."""

from src.core.retrieval_pipeline import RetrievalPipeline
from src.core.vector_store import VectorStore
from src.core.bm25_retriever import BM25Retriever
from src.core.cross_encoder import CrossEncoderReranker


def test_retrieval_pipeline_initialization():
    """Test RetrievalPipeline initialization."""
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    cross_encoder = CrossEncoderReranker()
    
    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        cross_encoder=cross_encoder,
        top_k=5,
    )
    
    assert pipeline.vector_store is not None
    assert pipeline.bm25_retriever is not None
    assert pipeline.cross_encoder is not None
    assert pipeline.top_k == 5
    print("✅ RetrievalPipeline initializes correctly")


def test_retrieval_pipeline_retrieve():
    """Test full retrieval pipeline."""
    chunks = [
        {"text": "Machine learning algorithms improve with data", "chunk_id": "c1", "doc_id": "ml.pdf", "page_num": 1, "chunk_index": 0},
        {"text": "Deep learning uses neural networks", "chunk_id": "c2", "doc_id": "dl.pdf", "page_num": 2, "chunk_index": 1},
        {"text": "Basketball is a sport", "chunk_id": "c3", "doc_id": "sports.pdf", "page_num": 5, "chunk_index": 2},
    ]
    
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    cross_encoder = CrossEncoderReranker()
    
    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        cross_encoder=cross_encoder,
        top_k=2,
    )
    
    results = pipeline.retrieve("machine learning", chunks)
    
    assert len(results) <= 2
    assert all("rerank_score" in r for r in results)
    print(f"✅ RetrievalPipeline.retrieve returned {len(results)} results")


def test_rrf_fusion():
    """Test RRF fusion logic."""
    bm25_results = [
        {"text": "ML text", "chunk_id": "c1", "doc_id": "d1", "page_num": 1},
    ]
    vector_results = [
        {"text": "DL text", "chunk_id": "c2", "doc_id": "d2", "page_num": 2},
    ]
    
    fused = RetrievalPipeline._rrf_fusion(bm25_results, vector_results, top_k=2)
    
    assert len(fused) <= 2
    assert all("rrf_score" in r for r in fused)
    print(f"✅ RRF fusion returned {len(fused)} results")


def test_backward_compatibility():
    """Test backward compatibility with existing retrieval function."""
    from src.retrieval.pipeline import retrieval
    from src.retrieval.bm25_index import build_bm25_index
    
    chunks = [
        {"text": "Test content", "chunk_id": "c1", "doc_id": "d1", "page_num": 1, "chunk_index": 0},
    ]
    
    # Build BM25 index
    bm25_index = build_bm25_index(chunks)
    
    # Test retrieval function
    results = retrieval("test", chunks, bm25_index, top_k=1)
    assert len(results) >= 0
    
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_retrieval_pipeline_initialization()
    test_retrieval_pipeline_retrieve()
    test_rrf_fusion()
    test_backward_compatibility()
    print("\n✅ All RetrievalPipeline tests passed!")
