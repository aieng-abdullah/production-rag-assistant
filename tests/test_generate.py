"""Tests for the generate function and _run_pipeline."""
import pytest
from unittest.mock import patch, MagicMock

from src.generation.chain import (
    _usage_from_lc_response,
    _build_sources,
    _run_pipeline,
    _invoke_llm,
    generate,
)
from src.generation.Citation_system import CitedAnswer, Source
from src.generation.providers import Provider


class TestUsageFromLcResponse:
    def test_extracts_token_usage(self):
        response = MagicMock()
        response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            }
        }
        result = _usage_from_lc_response(response)
        assert result == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}

    def test_handles_usage_key_instead_of_token_usage(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 7,
                "total_tokens": 10,
            }
        }
        result = _usage_from_lc_response(response)
        assert result["prompt_tokens"] == 3

    def test_returns_none_when_no_metadata(self):
        response = MagicMock()
        response.response_metadata = None
        assert _usage_from_lc_response(response) is None

    def test_returns_none_when_empty_metadata(self):
        response = MagicMock()
        response.response_metadata = {}
        assert _usage_from_lc_response(response) is None

    def test_returns_none_when_usage_not_dict(self):
        response = MagicMock()
        response.response_metadata = {"token_usage": "not-a-dict"}
        assert _usage_from_lc_response(response) is None

    def test_handles_partial_tokens(self):
        response = MagicMock()
        response.response_metadata = {
            "token_usage": {"prompt_tokens": 5}
        }
        result = _usage_from_lc_response(response)
        assert result == {"prompt_tokens": 5}
        assert "completion_tokens" not in result

    def test_handles_non_numeric_tokens(self):
        response = MagicMock()
        response.response_metadata = {
            "token_usage": {"prompt_tokens": "bad", "completion_tokens": 10, "total_tokens": 15}
        }
        result = _usage_from_lc_response(response)
        assert result == {"completion_tokens": 10, "total_tokens": 15}

    def test_handles_none_token_values(self):
        response = MagicMock()
        response.response_metadata = {
            "token_usage": {"prompt_tokens": None, "completion_tokens": 5, "total_tokens": None}
        }
        result = _usage_from_lc_response(response)
        assert result == {"completion_tokens": 5}

    def test_returns_none_when_usage_not_dict_type(self):
        response = MagicMock()
        response.response_metadata = {"token_usage": [1, 2, 3]}
        assert _usage_from_lc_response(response) is None


class TestBuildSources:
    def test_converts_chunks_to_sources(self):
        chunks = [
            {"doc_id": "d1", "page_num": 1, "text": "hello"},
            {"doc_id": "d2", "page_num": 2, "text": "world"},
        ]
        sources = _build_sources(chunks)
        assert len(sources) == 2
        assert isinstance(sources[0], Source)
        assert sources[0].doc_id == "d1"
        assert sources[0].page_num == 1
        assert sources[0].text == "hello"
        assert sources[1].doc_id == "d2"
        assert sources[1].page_num == 2
        assert sources[1].text == "world"

    def test_empty_chunks(self):
        assert _build_sources([]) == []


class TestRunPipeline:
    """Test _run_pipeline with real build_citation_prompt and _build_sources,
    but mocked LLM and retrieval (external services)."""

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_passes_retrieval_result_to_prompt(self, mock_llm, mock_retrieval):
        """Verify that chunks from retrieval are forwarded to build_citation_prompt."""
        chunks = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = ("answer [SOURCE 1]", None)

        result = _run_pipeline("what is X?", MagicMock())

        assert isinstance(result, CitedAnswer)
        assert result.answer == "answer [SOURCE 1]"
        assert len(result.sources) == 1
        assert result.sources[0].doc_id == "d1"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_llm_receives_citation_prompt(self, mock_llm, mock_retrieval):
        """Verify that _invoke_llm receives a prompt built from chunks."""
        chunks = [{"text": "fact about AI", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = ("[SOURCE 1]", None)

        _run_pipeline("what is AI?", MagicMock())

        call_args = mock_llm.call_args[0][0]
        assert "[SOURCE 1]" in call_args
        assert "fact about AI" in call_args
        assert "what is AI?" in call_args

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_creates_sources_from_chunks(self, mock_llm, mock_retrieval):
        """Verify that sources are built from the retrieval chunks."""
        chunks = [
            {"text": "first", "doc_id": "d1", "page_num": 1},
            {"text": "second", "doc_id": "d2", "page_num": 5},
        ]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = ("[SOURCE 1] and [SOURCE 2]", None)

        result = _run_pipeline("test", MagicMock())

        assert len(result.sources) == 2
        assert result.sources[0].text == "first"
        assert result.sources[1].text == "second"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_raises_on_citation_validation_failure(self, mock_llm, mock_retrieval):
        """Verify that an answer without [SOURCE N] raises ValidationError."""
        mock_retrieval.return_value = [{"text": "x", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = ("answer with no citation", None)

        with pytest.raises(Exception, match="citation"):
            _run_pipeline("test", MagicMock())


class TestGenerate:
    """Test the generate() routing logic (Langfuse vs non-Langfuse)."""

    @patch("src.generation.chain._run_pipeline")
    @patch("src.generation.chain.get_langfuse_client")
    def test_generate_no_langfuse_calls_run_pipeline(self, mock_lf, mock_pipeline):
        mock_lf.return_value = None
        mock_pipeline.return_value = CitedAnswer(
            answer="[SOURCE 1]",
            sources=[Source(doc_id="d1", page_num=1, text="x")],
        )
        bm25 = MagicMock()
        result = generate("query", bm25)
        mock_pipeline.assert_called_once_with("query", bm25)
        assert isinstance(result, CitedAnswer)

    @patch("src.generation.chain._generate_traced")
    @patch("src.generation.chain.get_langfuse_client")
    def test_generate_with_langfuse_calls_traced(self, mock_lf, mock_traced):
        mock_lf.return_value = MagicMock()
        mock_traced.return_value = CitedAnswer(
            answer="[SOURCE 1]",
            sources=[Source(doc_id="d1", page_num=1, text="x")],
        )
        bm25 = MagicMock()
        result = generate("query", bm25)
        mock_traced.assert_called_once_with("query", bm25, mock_lf.return_value)
        assert isinstance(result, CitedAnswer)


class TestInvokeLlmFailover:
    """Test the provider failover logic in _invoke_llm."""

    @patch("src.generation.chain.build_provider_chain")
    @patch("src.generation.chain._call_provider_with_retry")
    def test_first_provider_succeeds(self, mock_call, mock_chain):
        mock_chain.return_value = [
            Provider("groq", "key1", "model1"),
            Provider("anthropic", "key2", "model2"),
        ]
        mock_response = MagicMock()
        mock_response.content = "answer [SOURCE 1]"
        mock_response.response_metadata = {}
        mock_call.return_value = mock_response

        answer, usage = _invoke_llm("prompt")
        assert answer == "answer [SOURCE 1]"
        assert mock_call.call_count == 1

    @patch("src.generation.chain.build_provider_chain")
    @patch("src.generation.chain._call_provider_with_retry")
    def test_failover_to_second_provider(self, mock_call, mock_chain):
        mock_chain.return_value = [
            Provider("groq", "key1", "model1"),
            Provider("anthropic", "key2", "model2"),
        ]
        fail_response = MagicMock()
        fail_response.content = "fail"
        fail_response.response_metadata = {}

        success_response = MagicMock()
        success_response.content = "success [SOURCE 1]"
        success_response.response_metadata = {}

        mock_call.side_effect = [Exception("Groq down"), success_response]

        answer, usage = _invoke_llm("prompt")
        assert answer == "success [SOURCE 1]"
        assert mock_call.call_count == 2

    @patch("src.generation.chain.build_provider_chain")
    @patch("src.generation.chain._call_provider_with_retry")
    def test_all_providers_fail_raises(self, mock_call, mock_chain):
        mock_chain.return_value = [
            Provider("groq", "key1", "model1"),
            Provider("anthropic", "key2", "model2"),
        ]
        mock_call.side_effect = Exception("provider error")

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            _invoke_llm("prompt")
        assert mock_call.call_count == 2

    @patch("src.generation.chain.build_provider_chain")
    @patch("src.generation.chain._call_provider_with_retry")
    def test_single_provider_success(self, mock_call, mock_chain):
        mock_chain.return_value = [Provider("groq", "key1", "model1")]
        mock_response = MagicMock()
        mock_response.content = "answer [SOURCE 1]"
        mock_response.response_metadata = {}
        mock_call.return_value = mock_response

        answer, usage = _invoke_llm("prompt")
        assert answer == "answer [SOURCE 1]"
        assert mock_call.call_count == 1
