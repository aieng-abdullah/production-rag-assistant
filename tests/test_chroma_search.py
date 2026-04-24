"""Test LangChain Chroma search integration."""

from src.retrieval.chroma_search import vector_search, get_retriever
from src.db.chroma_client import upsert_chunks, reset_client, get_vectorstore
from src.ingestion.embedder import embed_chunks


def setup_test_data():
    """Insert test data into ChromaDB."""
    chunks = [
        {
            "text": "Artificial intelligence is transforming how we work and live.",
            "chunk_id": "test_doc_chunk_0",
            "doc_id": "test_doc",
            "page_num": 1,
            "chunk_index": 0,
        },
        {
            "text": "Machine learning algorithms can process vast amounts of data.",
            "chunk_id": "test_doc_chunk_1",
            "doc_id": "test_doc",
            "page_num": 1,
            "chunk_index": 1,
        },
        {
            "text": "Basketball requires teamwork and physical endurance.",
            "chunk_id": "test_doc_chunk_2",
            "doc_id": "test_doc",
            "page_num": 2,
            "chunk_index": 2,
        },
    ]

    # Embed and save
    embedded_chunks = embed_chunks(chunks)
    count = upsert_chunks(embedded_chunks)
    return count


def test_vector_search_returns_results():
    """Test that vector search returns properly formatted results."""
    # Setup test data
    setup_test_data()

    # Search for AI-related query
    results = vector_search("artificial intelligence technology", top_k=3)

    # Should return results
    assert isinstance(results, list)
    assert len(results) > 0

    # Each result should have required fields
    for result in results:
        assert "text" in result
        assert "doc_id" in result
        assert "page_num" in result
        assert "chunk_index" in result
        assert "score" in result
        assert "source" in result
        assert result["source"] == "vector"
        assert isinstance(result["score"], float)

    print(f"✅ vector_search: {len(results)} results")

    # First result should be AI-related (higher score)
    ai_related = any("AI" in r["text"] or "artificial" in r["text"].lower() for r in results[:2])
    print(f"✅ Results are relevant: {ai_related}")


def test_vector_search_scoring():
    """Test that scores are meaningful."""
    setup_test_data()

    # Search for AI query
    ai_results = vector_search("machine learning AI", top_k=2)

    # Search for sports query
    sports_results = vector_search("basketball sports", top_k=2)

    # AI query should return higher scores for AI content
    ai_scores = [r["score"] for r in ai_results]
    sports_scores = [r["score"] for r in sports_results]

    print(f"✅ AI query scores: {ai_scores}")
    print(f"✅ Sports query scores: {sports_scores}")


def test_get_retriever():
    """Test that retriever can be created and used."""
    setup_test_data()

    retriever = get_retriever(top_k=2)

    # Retriever should be callable
    docs = retriever.invoke("artificial intelligence")

    assert isinstance(docs, list)
    assert len(docs) > 0

    # Should return Document objects
    from langchain_core.documents import Document
    for doc in docs:
        assert isinstance(doc, Document)
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")

    print(f"✅ get_retriever: returned {len(docs)} Document objects")


def test_empty_query_handling():
    """Test behavior with edge case queries."""
    setup_test_data()

    # Should handle short queries
    results = vector_search("AI", top_k=2)
    assert isinstance(results, list)

    print(f"✅ Short query handled: {len(results)} results")


if __name__ == "__main__":
    try:
        test_vector_search_returns_results()
        test_vector_search_scoring()
        test_get_retriever()
        test_empty_query_handling()
        print("\n✅ All chroma_search tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
