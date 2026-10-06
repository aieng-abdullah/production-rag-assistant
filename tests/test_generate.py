"""Tests for the generate function and _run_pipeline."""
import json

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
from src.generation.schema import AnswerVerificationError
from src.generation.providers import Provider
from src.generation.verifier import VERIFY_RETRY, ClaimVerification


def _json_answer(text="The fact holds.", source_id=1, quote="chunk text"):
    """Valid structured output whose quote matches the default test chunks."""
    return json.dumps(
        {
            "claims": [
                {"text": text, "citations": [{"source_id": source_id, "quote": quote}]}
            ],
            "abstained": False,
            "abstain_reason": None,
        }
    )


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


@pytest.fixture(autouse=True)
def _judge_all_supported():
    """Offline judge: every claim SUPPORTED (per-test overrides via nested patch)."""

    def fake_judge(structured, chunks, callbacks=None):
        return [
            ClaimVerification(claim_index=i, verdict="SUPPORTED", reason="ok")
            for i in range(len(structured.claims))
        ]

    with patch("src.generation.chain.judge_claims", side_effect=fake_judge):
        yield


class TestRunPipeline:
    """Test _run_pipeline with real build_citation_prompt and _build_sources,
    but mocked LLM and retrieval (external services)."""

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_passes_retrieval_result_to_prompt(self, mock_llm, mock_retrieval):
        """Verify that chunks from retrieval are forwarded to build_citation_prompt."""
        chunks = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = (_json_answer(), None)

        result = _run_pipeline("what is X?", MagicMock())

        assert isinstance(result, CitedAnswer)
        assert result.answer == "The fact holds. [SOURCE 1]"
        assert len(result.sources) == 1
        assert result.sources[0].doc_id == "d1"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_llm_receives_citation_prompt(self, mock_llm, mock_retrieval):
        """Verify that _invoke_llm receives a prompt built from chunks."""
        chunks = [{"text": "fact about AI", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.return_value = (_json_answer(quote="fact about AI"), None)

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
        two_claims = json.dumps(
            {
                "claims": [
                    {"text": "A.", "citations": [{"source_id": 1, "quote": "first"}]},
                    {"text": "B.", "citations": [{"source_id": 2, "quote": "second"}]},
                ],
                "abstained": False,
                "abstain_reason": None,
            }
        )
        mock_llm.return_value = (two_claims, None)

        result = _run_pipeline("test", MagicMock())

        assert len(result.sources) == 2
        assert result.sources[0].text == "first"
        assert result.sources[1].text == "second"
        assert result.answer == "A. [SOURCE 1] B. [SOURCE 2]"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_raises_when_output_unrepairable(self, mock_llm, mock_retrieval):
        """Garbage twice (original + repair) → AnswerVerificationError."""
        mock_retrieval.return_value = [{"text": "x", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = ("answer with no citation", None)

        with pytest.raises(AnswerVerificationError, match="unrepairable"):
            _run_pipeline("test", MagicMock())
        assert mock_llm.call_count == 2

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_repairs_unparseable_output(self, mock_llm, mock_retrieval):
        """First output unparseable, repair returns valid JSON → success."""
        chunks = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.side_effect = [
            ("not json at all", None),
            (_json_answer(), None),
        ]

        result = _run_pipeline("test", MagicMock())

        assert result.answer == "The fact holds. [SOURCE 1]"
        assert mock_llm.call_count == 2
        assert "failed verification" in mock_llm.call_args[0][0]

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_repairs_bad_quote(self, mock_llm, mock_retrieval):
        """Quote not in source → repair re-ask fixes it."""
        chunks = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_retrieval.return_value = chunks
        mock_llm.side_effect = [
            (_json_answer(quote="invented quote"), None),
            (_json_answer(), None),
        ]

        result = _run_pipeline("test", MagicMock())

        assert result.answer == "The fact holds. [SOURCE 1]"
        assert mock_llm.call_count == 2

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_fails_when_repair_quote_still_bad(self, mock_llm, mock_retrieval):
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (_json_answer(quote="invented"), None)

        with pytest.raises(AnswerVerificationError, match="unrepairable"):
            _run_pipeline("test", MagicMock())
        assert mock_llm.call_count == 2

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_verification_status_verified(self, mock_llm, mock_retrieval):
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (_json_answer(), None)

        result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "verified"
        assert result.verification["per_claim"][0]["verdict"] == "SUPPORTED"
        assert result.verification["per_claim"][0]["text"] == "The fact holds."

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_includes_trace_payload(self, mock_llm, mock_retrieval):
        mock_retrieval.return_value = [
            {"text": "chunk text", "doc_id": "d1", "page_num": 7}
        ]
        mock_llm.return_value = (_json_answer(), None)

        result = _run_pipeline("test", MagicMock(), workspace="legal")

        trace = result.trace
        assert trace["workspace"] == "legal"
        assert trace["prompt_version"] == "legal-v4"
        assert trace["chunks"] == [
            {"source_id": 1, "doc_id": "d1", "page_num": 7, "cited": True}
        ]
        assert trace["claims"][0]["citations"][0]["quote"] == "chunk text"
        assert trace["abstained"] is False
        assert trace["verification"]["status"] == "verified"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_judge_rejection_triggers_regen(self, mock_llm, mock_retrieval):
        """One UNSUPPORTED round → feedback re-gen → verified, no exception."""
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.side_effect = [(_json_answer(), None), (_json_answer(), None)]
        rounds = [
            [ClaimVerification(claim_index=0, verdict="UNSUPPORTED", reason="not entailed")],
            [ClaimVerification(claim_index=0, verdict="SUPPORTED", reason="ok")],
        ]
        with patch("src.generation.chain.judge_claims", side_effect=rounds):
            result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "verified"
        assert mock_llm.call_count == 2
        assert "rejected by the verifier" in mock_llm.call_args[0][0]
        # Repair must not nudge the model to shrink the answer: rejected
        # claims are fixed/dropped individually, the rest stay.
        assert "keep the rest" in mock_llm.call_args[0][0]

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_persistent_rejection_ships_unverified(
        self, mock_llm, mock_retrieval
    ):
        """Bounded retry: 1 + VERIFY_RETRY gens, then badge — never raise."""
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (_json_answer(), None)
        always_bad = [
            ClaimVerification(claim_index=0, verdict="UNSUPPORTED", reason="nope")
        ]
        with patch(
            "src.generation.chain.judge_claims",
            side_effect=[always_bad] * (1 + VERIFY_RETRY),
        ):
            result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "unverified"
        assert mock_llm.call_count == 1 + VERIFY_RETRY

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_judge_outage_skips_regen(self, mock_llm, mock_retrieval):
        """Judge outage is not fixable by regenerating — no wasted re-gen."""
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (_json_answer(), None)
        outage = [
            ClaimVerification(
                claim_index=0, verdict="UNSUPPORTED", reason="judge unavailable: 503"
            )
        ]
        with patch("src.generation.chain.judge_claims", side_effect=[outage]):
            result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "unverified"
        assert mock_llm.call_count == 1

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_abstained_skips_judge(self, mock_llm, mock_retrieval):
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (
            json.dumps(
                {
                    "claims": [],
                    "abstained": True,
                    "abstain_reason": "I don't have enough information to answer "
                    "this question based on the provided sources.",
                }
            ),
            None,
        )
        result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "abstained"

    @patch("src.generation.chain.retrieval")
    @patch("src.generation.chain._invoke_llm")
    def test_pipeline_partial_verdicts_reported(self, mock_llm, mock_retrieval):
        mock_retrieval.return_value = [{"text": "chunk text", "doc_id": "d1", "page_num": 1}]
        mock_llm.return_value = (_json_answer(), None)
        rounds = [
            [ClaimVerification(claim_index=0, verdict="PARTIAL", reason="half")],
            [ClaimVerification(claim_index=0, verdict="PARTIAL", reason="half")],
            [ClaimVerification(claim_index=0, verdict="PARTIAL", reason="half")],
        ]
        with patch("src.generation.chain.judge_claims", side_effect=rounds):
            result = _run_pipeline("test", MagicMock())

        assert result.verification["status"] == "partial"
        assert mock_llm.call_count == 1 + VERIFY_RETRY


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
        mock_pipeline.assert_called_once_with(
            "query", bm25, None, "default", workspace="academic"
        )
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
        mock_traced.assert_called_once_with(
            "query", bm25, mock_lf.return_value, None, "default", workspace="academic"
        )
        assert isinstance(result, CitedAnswer)

    @patch("src.generation.chain._run_pipeline")
    @patch("src.generation.chain._generate_traced")
    @patch("src.generation.chain.get_langfuse_client")
    def test_generate_traced_importerror_falls_back(
        self, mock_lf, mock_traced, mock_pipeline
    ):
        """A broken Langfuse integration must degrade to the untraced pipeline."""
        mock_lf.return_value = MagicMock()
        mock_traced.side_effect = ImportError("No module named 'langfuse.langchain'")
        mock_pipeline.return_value = CitedAnswer(
            answer="[SOURCE 1]",
            sources=[Source(doc_id="d1", page_num=1, text="x")],
        )
        bm25 = MagicMock()
        result = generate("query", bm25)
        mock_traced.assert_called_once_with(
            "query", bm25, mock_lf.return_value, None, "default", workspace="academic"
        )
        mock_pipeline.assert_called_once_with(
            "query", bm25, None, "default", workspace="academic"
        )
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
