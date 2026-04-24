"""Test LangChain generation module.

Tests prompts formatting, chain structure, and (optionally) LLM calls.
Set GROQ_API_KEY env var to run live LLM tests.
"""

import os
from src.generation.prompts import format_context, RAG_PROMPT, CITATION_VALIDATION_PROMPT
from src.generation.chain import _extract_citations, create_rag_chain


def test_format_context():
    """Test context formatting with source numbers."""
    chunks = [
        {
            "text": "First chunk about AI.",
            "doc_id": "doc1",
            "page_num": 1,
            "chunk_index": 0,
        },
        {
            "text": "Second chunk about ML.",
            "doc_id": "doc1",
            "page_num": 2,
            "chunk_index": 1,
        },
    ]

    formatted = format_context(chunks)

    # Should contain source markers
    assert "[SOURCE 1]" in formatted
    assert "[SOURCE 2]" in formatted

    # Should contain text
    assert "First chunk about AI" in formatted
    assert "Second chunk about ML" in formatted

    # Should contain metadata
    assert "doc1" in formatted
    assert "(Page 1)" in formatted
    assert "(Page 2)" in formatted

    print("✅ format_context: properly formats chunks with sources")


def test_rag_prompt_structure():
    """Test RAG prompt template has correct structure."""
    # Check prompt has expected variables
    assert "context" in RAG_PROMPT.input_variables
    assert "question" in RAG_PROMPT.input_variables

    # Format the prompt
    messages = RAG_PROMPT.format_messages(
        context="Test context here",
        question="What is AI?",
    )

    # Should have system and human messages
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"

    # System message should have instructions
    assert "citation" in messages[0].content.lower()
    assert "Test context here" in messages[0].content

    # Human message should have question
    assert "What is AI?" in messages[1].content

    print("✅ RAG_PROMPT: correct structure with system and human messages")


def test_extract_citations():
    """Test citation extraction from answer text."""
    chunks = [
        {"text": "Chunk 1 content", "doc_id": "d1", "page_num": 1, "chunk_index": 0},
        {"text": "Chunk 2 content", "doc_id": "d1", "page_num": 1, "chunk_index": 1},
        {"text": "Chunk 3 content", "doc_id": "d2", "page_num": 5, "chunk_index": 2},
    ]

    answer = """
    According to research [SOURCE 1], AI is transforming industries.
    Studies show [SOURCE 2] that machine learning is key.
    More evidence [SOURCE 1] supports this claim.
    Additional data [SOURCE 3] from other papers confirms it.
    """

    citations = _extract_citations(answer, chunks)

    # Should extract unique citations only (SOURCE 1 appears twice but only once in output)
    assert len(citations) == 3

    # Should have correct source numbers
    source_nums = [c["source_number"] for c in citations]
    assert 1 in source_nums
    assert 2 in source_nums
    assert 3 in source_nums

    # Should have correct metadata
    citation_1 = next(c for c in citations if c["source_number"] == 1)
    assert citation_1["doc_id"] == "d1"
    assert citation_1["page_num"] == 1
    assert citation_1["chunk_index"] == 0

    print("✅ _extract_citations: extracts unique citations with metadata")


def test_extract_citations_out_of_range():
    """Test citation extraction handles out-of-range source numbers."""
    chunks = [
        {"text": "Only chunk", "doc_id": "d1", "page_num": 1, "chunk_index": 0},
    ]

    answer = "Claims [SOURCE 1] and [SOURCE 5] and [SOURCE 999]"

    citations = _extract_citations(answer, chunks)

    # Only SOURCE 1 should be extracted (others out of range)
    assert len(citations) == 1
    assert citations[0]["source_number"] == 1

    print("✅ _extract_citations: handles out-of-range source numbers")


def test_chain_structure():
    """Test that RAG chain has correct structure without calling LLM."""
    # This test checks chain composition without needing API key
    try:
        chain = create_rag_chain()

        # Chain should be runnable
        assert chain is not None

        print("✅ create_rag_chain: chain is properly constructed")
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print("⚠️ create_rag_chain: requires GROQ_API_KEY (expected)")
        else:
            raise


def test_live_generation():
    """Test live LLM generation - requires GROQ_API_KEY."""
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ Skipping live test: GROQ_API_KEY not set")
        return

    from src.generation.chain import generate_answer

    chunks = [
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "doc_id": "ai_doc",
            "page_num": 1,
            "chunk_index": 0,
        },
        {
            "text": "Machine learning is a subset of AI that uses algorithms to learn from data.",
            "doc_id": "ai_doc",
            "page_num": 2,
            "chunk_index": 1,
        },
    ]

    result = generate_answer(
        question="What is artificial intelligence?",
        context_chunks=chunks,
    )

    # Should return dict with expected keys
    assert "answer" in result
    assert "citations" in result
    assert "context_used" in result

    # Answer should be a non-empty string
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0

    # Should have at least one citation
    assert len(result["citations"]) > 0

    print(f"✅ generate_answer: produced answer with {len(result['citations'])} citations")
    print(f"   Answer preview: {result['answer'][:100]}...")


if __name__ == "__main__":
    test_format_context()
    test_rag_prompt_structure()
    test_extract_citations()
    test_extract_citations_out_of_range()
    test_chain_structure()
    test_live_generation()
    print("\n✅ All generation module tests completed!")
