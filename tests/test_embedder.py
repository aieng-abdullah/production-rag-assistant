"""Test LangChain embedder integration."""

from src.ingestion.embedder import embed_query, embed_chunks, _get_model


def test_embed_query():
    """Test single query embedding returns correct format."""
    text = "This is a test query for embedding."
    embedding = embed_query(text)

    # Should return a list of floats
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

    # Should be normalized (length close to 1)
    import math
    norm = math.sqrt(sum(x * x for x in embedding))
    assert 0.99 < norm < 1.01, f"Embedding not normalized, norm={norm}"

    print(f"✅ embed_query: {len(embedding)} dimensions, norm={norm:.4f}")


def test_embed_chunks():
    """Test batch chunk embedding preserves metadata."""
    chunks = [
        {"text": "First test chunk about AI technology.", "doc_id": "doc1", "chunk_id": "c1"},
        {"text": "Second test chunk about machine learning.", "doc_id": "doc1", "chunk_id": "c2"},
        {"text": "Third test chunk about deep learning.", "doc_id": "doc2", "chunk_id": "c3"},
    ]

    result = embed_chunks(chunks, batch_size=2)

    # Should return same number of chunks
    assert len(result) == 3

    # Each chunk should have embedding added
    for chunk in result:
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) > 0
        # Original metadata preserved
        assert "doc_id" in chunk
        assert "chunk_id" in chunk

    # All embeddings should have same dimension
    dims = [len(c["embedding"]) for c in result]
    assert all(d == dims[0] for d in dims)

    print(f"✅ embed_chunks: {len(result)} chunks, {dims[0]} dimensions each")


def test_model_singleton():
    """Test that model is lazy-loaded and reused."""
    # Reset first
    from src.ingestion import embedder
    embedder._model = None

    # First call should load model
    model1 = _get_model()
    model2 = _get_model()

    # Should be same instance
    assert model1 is model2
    print("✅ Model singleton: same instance reused")


def test_similarity_consistency():
    """Test that similar texts have higher similarity."""
    text1 = "Artificial intelligence and machine learning"
    text2 = "Machine learning and AI technologies"
    text3 = "Basketball is a popular sport"

    emb1 = embed_query(text1)
    emb2 = embed_query(text2)
    emb3 = embed_query(text3)

    # Cosine similarity (dot product since normalized)
    def cosine_sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_1_2 = cosine_sim(emb1, emb2)  # Should be high (similar topic)
    sim_1_3 = cosine_sim(emb1, emb3)  # Should be low (different topic)

    assert sim_1_2 > sim_1_3, f"Similar texts should have higher similarity: {sim_1_2} vs {sim_1_3}"

    print(f"✅ Similarity: similar texts={sim_1_2:.4f}, different texts={sim_1_3:.4f}")


if __name__ == "__main__":
    test_embed_query()
    test_embed_chunks()
    test_model_singleton()
    test_similarity_consistency()
    print("\n✅ All embedder tests passed!")
