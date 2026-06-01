"""Tests for the generate function and _run_pipeline."""

from unittest.mock import patch, MagicMock
import pytest


@patch("src.generation.chain.retrieval")
@patch("src.generation.chain.build_citation_prompt")
@patch("src.generation.chain._invoke_llm")
@patch("src.generation.chain._build_sources")
def test_run_pipeline(mock_sources, mock_llm, mock_prompt, mock_retrieval):
    mock_retrieval.return_value = [{"text": "chunk", "doc_id": "d1", "page_num": 1}]
    mock_prompt.return_value = "prompt"
    mock_llm.return_value = ("answer [SOURCE 1]", None)
    mock_sources.return_value = []

    from src.generation.chain import _run_pipeline
    result = _run_pipeline("query", MagicMock())
    assert result.answer == "answer [SOURCE 1]"


@patch("src.generation.chain.get_langfuse_client")
@patch("src.generation.chain._run_pipeline")
def test_generate_no_langfuse(mock_pipeline, mock_lf):
    mock_lf.return_value = None
    mock_pipeline.return_value = MagicMock(answer="test")

    from src.generation.chain import generate
    result = generate("query", MagicMock())
    assert result.answer == "test"
    mock_pipeline.assert_called_once()


@patch("src.generation.chain.get_langfuse_client")
@patch("src.generation.chain._generate_traced")
def test_generate_with_langfuse(mock_traced, mock_lf):
    mock_lf.return_value = MagicMock()
    mock_traced.return_value = MagicMock(answer="traced")

    from src.generation.chain import generate
    result = generate("query", MagicMock())
    assert result.answer == "traced"
    mock_traced.assert_called_once()


@patch("src.generation.chain.ChatGroq")
def test_invoke_llm(mock_chat_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "test answer"
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}
    mock_client.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_client

    from src.generation.chain import _invoke_llm
    text, usage = _invoke_llm("test prompt")
    assert text == "test answer"
    assert usage["prompt_tokens"] == 5
