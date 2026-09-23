"""Tests for PDF parser."""

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from src.ingestion.parser import extract_pages


def test_extract_pages_file_not_found():
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_pages("/nonexistent/file.pdf")


@patch("src.ingestion.parser.fitz")
def test_extract_pages_success(mock_fitz):
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "A" * 200
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "B" * 200

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))
    mock_fitz.open.return_value = mock_doc

    with patch.object(Path, "exists", return_value=True):
        pages = extract_pages("/fake/test.pdf", doc_id="test_doc")

    assert len(pages) == 2
    assert pages[0]["doc_id"] == "test_doc"
    assert pages[0]["page_num"] == 1
    assert pages[0]["total_pages"] == 2


@patch("src.ingestion.parser.fitz")
def test_extract_pages_skips_short_pages(mock_fitz):
    mock_short = MagicMock()
    mock_short.get_text.return_value = "short"

    mock_long = MagicMock()
    mock_long.get_text.return_value = "A" * 200

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_short, mock_long]))
    mock_fitz.open.return_value = mock_doc

    with patch.object(Path, "exists", return_value=True):
        pages = extract_pages("/fake/test.pdf")

    assert len(pages) == 1
    assert pages[0]["page_num"] == 2


@patch("src.ingestion.parser.fitz")
def test_extract_pages_default_doc_id(mock_fitz):
    mock_page = MagicMock()
    mock_page.get_text.return_value = "A" * 200

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_fitz.open.return_value = mock_doc

    with patch.object(Path, "exists", return_value=True):
        pages = extract_pages("/fake/myfile.pdf")

    assert pages[0]["doc_id"] == "myfile"
