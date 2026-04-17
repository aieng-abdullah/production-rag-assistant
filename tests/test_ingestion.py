"""Tests for ingestion pipeline."""

import pytest
from pathlib import Path

from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_pages


class TestParser:
    """Test PDF parsing functionality."""
    
    def test_parse_pdf_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_pdf("nonexistent.pdf")
    
    def test_parse_returns_pages(self, tmp_path):
        """Test that parser returns list of page dicts."""
        # Note: This requires a test PDF file
        # Create a minimal test or mock
        pass


class TestChunker:
    """Test text chunking functionality."""
    
    def test_chunk_pages_basic(self):
        """Test basic chunking with sample pages."""
        pages = [
            {"page_num": 1, "text": "This is page one. " * 50, "char_count": 600},
            {"page_num": 2, "text": "This is page two. " * 50, "char_count": 600}
        ]
        
        chunks = chunk_pages(pages, "test-doc", "test.pdf")
        
        # Assert chunks created
        assert len(chunks) > 0
        
        # Assert metadata preserved
        for chunk in chunks:
            assert "doc_id" in chunk
            assert "filename" in chunk
            assert "page_num" in chunk
            assert "chunk_index" in chunk
            assert chunk["doc_id"] == "test-doc"
    
    def test_chunk_skips_empty(self):
        """Test that tiny chunks are skipped."""
        pages = [{"page_num": 1, "text": "Short.", "char_count": 6}]
        chunks = chunk_pages(pages, "test", "test.pdf")
        assert len(chunks) == 0
