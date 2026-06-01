"""Tests for generation chain utility functions."""

from src.generation.chain import _usage_from_lc_response, _build_sources
from src.generation.Citation_system import Source


def test_usage_from_lc_response_token_usage():
    response = MagicMock()
    response.response_metadata = {
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    result = _usage_from_lc_response(response)
    assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_usage_from_lc_response_usage_key():
    response = MagicMock()
    response.response_metadata = {
        "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
    }
    result = _usage_from_lc_response(response)
    assert result == {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}


def test_usage_from_lc_response_no_metadata():
    response = MagicMock()
    response.response_metadata = {}
    result = _usage_from_lc_response(response)
    assert result is None


def test_usage_from_lc_response_none_metadata():
    response = MagicMock()
    response.response_metadata = None
    result = _usage_from_lc_response(response)
    assert result is None


def test_usage_from_lc_response_partial_tokens():
    response = MagicMock()
    response.response_metadata = {
        "token_usage": {"prompt_tokens": 10}
    }
    result = _usage_from_lc_response(response)
    assert result == {"prompt_tokens": 10}
    assert "completion_tokens" not in result


def test_usage_from_lc_response_non_numeric():
    response = MagicMock()
    response.response_metadata = {
        "token_usage": {"prompt_tokens": "invalid", "completion_tokens": 5, "total_tokens": 10}
    }
    result = _usage_from_lc_response(response)
    assert result == {"completion_tokens": 5, "total_tokens": 10}


def test_build_sources():
    chunks = [
        {"doc_id": "doc1", "page_num": 1, "text": "hello"},
        {"doc_id": "doc2", "page_num": 5, "text": "world"},
    ]
    sources = _build_sources(chunks)
    assert len(sources) == 2
    assert isinstance(sources[0], Source)
    assert sources[0].doc_id == "doc1"
    assert sources[0].page_num == 1
    assert sources[0].text == "hello"
    assert sources[1].doc_id == "doc2"


def test_build_sources_empty():
    sources = _build_sources([])
    assert sources == []


def test_usage_from_lc_response_usage_not_dict():
    response = MagicMock()
    response.response_metadata = {"token_usage": "not_a_dict"}
    result = _usage_from_lc_response(response)
    assert result is None


from unittest.mock import MagicMock
