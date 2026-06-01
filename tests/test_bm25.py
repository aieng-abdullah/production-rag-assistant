"""Tests for BM25 index module."""

from unittest.mock import patch, MagicMock
from src.retrieval.bm25_index import (
    _chunks_to_documents,
    _documents_to_chunks,
    build_bm25_index,
    bm25_search,
    get_bm25_retriever,
)
from langchain_core.documents import Document


def test_chunks_to_documents():
    chunks = [
        {"text": "hello", "doc_id": "d1", "page_num": 1, "chunk_index": 0, "chunk_id": "d1_c0"},
        {"text": "world", "doc_id": "d2", "page_num": 2, "chunk_index": 1, "chunk_id": "d2_c1"},
    ]
    docs = _chunks_to_documents(chunks)
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["doc_id"] == "d1"
    assert docs[0].metadata["source"] == "bm25"


def test_chunks_to_documents_defaults():
    chunks = [{"text": "test"}]
    docs = _chunks_to_documents(chunks)
    assert docs[0].metadata["doc_id"] == "unknown"
    assert docs[0].metadata["page_num"] == -1


def test_documents_to_chunks():
    docs = [
        Document(page_content="hello", metadata={"doc_id": "d1", "page_num": 1, "chunk_index": 0}),
        Document(page_content="world", metadata={"doc_id": "d2", "page_num": 2, "chunk_index": 1}),
    ]
    chunks = _documents_to_chunks(docs)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "hello"
    assert chunks[0]["doc_id"] == "d1"
    assert chunks[0]["source"] == "bm25"
    assert chunks[0]["chunk_id"] == "d1_chunk_0"


def test_documents_to_chunks_defaults():
    docs = [Document(page_content="test", metadata={})]
    chunks = _documents_to_chunks(docs)
    assert chunks[0]["doc_id"] == "unknown"
    assert chunks[0]["page_num"] == -1


@patch("src.retrieval.bm25_index.BM25Retriever")
def test_build_bm25_index(mock_bm25_cls):
    chunks = [{"text": "hello world", "doc_id": "d1", "page_num": 1, "chunk_index": 0, "chunk_id": "d1_c0"}]
    mock_retriever = MagicMock()
    mock_bm25_cls.from_documents.return_value = mock_retriever
    result = build_bm25_index(chunks)
    assert result is mock_retriever
    mock_bm25_cls.from_documents.assert_called_once()


@patch("src.retrieval.bm25_index.BM25Retriever")
def test_bm25_search(mock_bm25_cls):
    mock_retriever = MagicMock()
    mock_doc = Document(page_content="result", metadata={"doc_id": "d1", "page_num": 1, "chunk_index": 0})
    mock_retriever.invoke.return_value = [mock_doc]
    results = bm25_search(mock_retriever, "query", top_k=5)
    assert len(results) == 1
    assert results[0]["text"] == "result"
    assert results[0]["source"] == "bm25"


@patch("src.retrieval.bm25_index.BM25Retriever")
def test_bm25_search_error(mock_bm25_cls):
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = RuntimeError("search failed")
    with pytest.raises(RuntimeError, match="Error while ranking documents"):
        bm25_search(mock_retriever, "query", top_k=5)


@patch("src.retrieval.bm25_index.BM25Retriever")
def test_get_bm25_retriever(mock_bm25_cls):
    chunks = [{"text": "hello", "doc_id": "d1", "page_num": 1, "chunk_index": 0, "chunk_id": "d1_c0"}]
    mock_retriever = MagicMock()
    mock_bm25_cls.from_documents.return_value = mock_retriever
    result = get_bm25_retriever(chunks, top_k=10)
    assert result is mock_retriever


import pytest
