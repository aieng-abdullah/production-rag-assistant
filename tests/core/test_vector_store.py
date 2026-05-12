"""Unit tests for VectorStore class."""

from src.core.vector_store import VectorStore


def test_vector_store_initialization():
    """Test VectorStore initialization."""
    store = VectorStore()
    assert store._vectorstore is None  # Lazy loading
    assert store.collection_name is not None
    assert store.persist_directory is not None
    assert store.embedding_model is not None
    print("✅ VectorStore initializes correctly")


def test_lazy_loading():
    """Test that vectorstore is lazy-loaded."""
    store = VectorStore()
    assert store._vectorstore is None
    
    # Access vectorstore property to trigger lazy load
    _ = store.vectorstore
    assert store._vectorstore is not None
    print("✅ VectorStore lazy-loads on first access")


def test_reset():
    """Test reset method."""
    store = VectorStore()
    _ = store.vectorstore  # Load vectorstore
    assert store._vectorstore is not None
    
    store.reset()
    assert store._vectorstore is None
    print("✅ Reset clears vectorstore from memory")


def test_backward_compatibility():
    """Test backward compatibility with existing functions."""
    from src.db.chroma_client import get_collection, upsert_chunks, load_all_chunks, reset_client
    
    # Test get_collection
    collection = get_collection()
    assert collection is not None
    
    # Test reset_client
    reset_client()
    
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_vector_store_initialization()
    test_lazy_loading()
    test_reset()
    test_backward_compatibility()
    print("\n✅ All VectorStore tests passed!")
