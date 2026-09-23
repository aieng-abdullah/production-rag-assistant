"""Tests for document chunking logic."""

from src.ingestion.chunker import chunk_pages


def test_chunk_pages_basic():
    pages = [
        {"text": "word " * 60, "page_num": 1, "doc_id": "doc1", "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert chunk["doc_id"] == "doc1"
        assert chunk["page_num"] == 1
        assert chunk["filename"] == "test.pdf"


def test_chunk_pages_empty_page_skipped():
    pages = [
        {"text": "", "page_num": 1, "doc_id": "doc1", "source": "test.pdf"},
        {"text": "word " * 60, "page_num": 2, "doc_id": "doc1", "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages)
    assert all(c["page_num"] == 2 for c in chunks)


def test_chunk_pages_metadata_preserved():
    pages = [
        {"text": "word " * 60, "page_num": 3, "doc_id": "mydoc", "source": "file.pdf"},
    ]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["doc_id"] == "mydoc"
        assert chunk["page_num"] == 3
        assert chunk["filename"] == "file.pdf"
        assert "chunk_id" in chunk
        assert "chunk_index" in chunk
        assert "char_count" in chunk


def test_chunk_pages_total_chunks_added():
    pages = [
        {"text": "word " * 60, "page_num": 1, "doc_id": "d1", "source": "f.pdf"},
        {"text": "word " * 60, "page_num": 2, "doc_id": "d1", "source": "f.pdf"},
    ]
    chunks = chunk_pages(pages)
    total = len(chunks)
    assert total > 0
    for chunk in chunks:
        assert chunk["total_chunks"] == total


def test_chunk_pages_small_chunks_filtered():
    page_text = "short " * 10
    pages = [{"text": page_text, "page_num": 1, "doc_id": "d1", "source": "f.pdf"}]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert len(chunk["text"]) >= 50


def test_chunk_pages_no_pages():
    chunks = chunk_pages([])
    assert chunks == []


def test_chunk_pages_global_index_increments():
    pages = [
        {"text": "word " * 60, "page_num": 1, "doc_id": "d1", "source": "f.pdf"},
        {"text": "word " * 60, "page_num": 2, "doc_id": "d1", "source": "f.pdf"},
    ]
    chunks = chunk_pages(pages)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
