"""Unit tests for EmbeddingModel class."""

from src.core.embedding_model import EmbeddingModel


def test_embedding_model_initialization():
    """Test EmbeddingModel initialization."""
    model = EmbeddingModel()
    assert model._model is None  # Lazy loading
    assert model._model_name is not None
    assert model._device == "cpu"
    print("✅ EmbeddingModel initializes correctly")


def test_embed_query():
    """Test embed_query method."""
    model = EmbeddingModel()
    embedding = model.embed_query("test query")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 produces 384-dim vectors
    assert all(isinstance(x, float) for x in embedding)
    print("✅ embed_query produces 384-dim vectors")


def test_embed_documents():
    """Test embed_documents method."""
    model = EmbeddingModel()
    texts = ["first text", "second text", "third text"]
    embeddings = model.embed_documents(texts)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(isinstance(emb, list) for emb in embeddings)
    print("✅ embed_documents produces correct number of embeddings")


def test_lazy_loading():
    """Test that model is lazy-loaded."""
    model = EmbeddingModel()
    assert model._model is None
    
    # Access model property to trigger lazy load
    _ = model.model
    assert model._model is not None
    print("✅ Model lazy-loads on first access")


def test_reset():
    """Test reset method."""
    model = EmbeddingModel()
    _ = model.model  # Load model
    assert model._model is not None
    
    model.reset()
    assert model._model is None
    print("✅ Reset clears model from memory")


def test_backward_compatibility():
    """Test backward compatibility with existing functions."""
    from src.ingestion.embedder import embed_query, embed_chunks
    
    # Test embed_query
    embedding = embed_query("test")
    assert len(embedding) == 384
    
    # Test embed_chunks
    chunks = [{"text": "test chunk", "doc_id": "test", "chunk_index": 0}]
    result = embed_chunks(chunks)
    assert "embedding" in result[0]
    assert len(result[0]["embedding"]) == 384
    
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_embedding_model_initialization()
    test_embed_query()
    test_embed_documents()
    test_lazy_loading()
    test_reset()
    test_backward_compatibility()
    print("\n✅ All EmbeddingModel tests passed!")
