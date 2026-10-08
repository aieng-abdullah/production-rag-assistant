"""Standalone-query rewrite: one cheap LLM call with a never-fail fallback."""

from unittest.mock import patch

import pytest
from loguru import logger

from src.config import Config
from src.generation.query_rewrite import (
    MAX_REWRITTEN_CHARS,
    REWRITE_HISTORY_TURNS,
    rewrite_query,
)

HISTORY = [
    {"role": "user", "content": "compare the termination clauses"},
    {"role": "assistant", "content": "Clause 8 allows 30 days notice."},
]
QUERY = "what about the second one?"


@pytest.fixture()
def warnings():
    """Captured loguru WARNING records for the duration of one test."""
    records: list[str] = []
    handler_id = logger.add(records.append, level="WARNING")
    yield records
    logger.remove(handler_id)


class TestNoHistorySkipsRewrite:
    @patch("src.generation.chain._invoke_llm")
    def test_none_history_returns_query_without_llm_call(self, mock_llm):
        assert rewrite_query(QUERY, None, tenant_id="7") == QUERY
        mock_llm.assert_not_called()

    @patch("src.generation.chain._invoke_llm")
    def test_empty_history_returns_query_without_llm_call(self, mock_llm):
        assert rewrite_query(QUERY, [], tenant_id="7") == QUERY
        mock_llm.assert_not_called()


class TestRewriteSuccess:
    @patch("src.generation.chain._invoke_llm")
    def test_returns_rewritten_query(self, mock_llm):
        mock_llm.return_value = ("termination clauses notice periods", None)

        result = rewrite_query(QUERY, HISTORY, tenant_id="7")

        assert result == "termination clauses notice periods"
        prompt = mock_llm.call_args[0][0]
        assert QUERY in prompt
        assert "user: compare the termination clauses" in prompt
        assert "assistant: Clause 8 allows 30 days notice." in prompt

    @patch("src.generation.chain._invoke_llm")
    def test_uses_cheap_verify_model_tier(self, mock_llm):
        mock_llm.return_value = ("standalone", None)

        with patch.object(Config, "VERIFY_MODEL", "openai/gpt-oss-20b"), patch.object(
            Config, "GROQ_MODEL", "qwen/qwen3.8-27b"
        ):
            rewrite_query(QUERY, HISTORY, tenant_id="7")

        overrides = mock_llm.call_args.kwargs["provider_overrides"]
        assert overrides.groq_model == "openai/gpt-oss-20b"

    @patch("src.generation.chain._invoke_llm")
    def test_default_model_when_verify_tier_is_not_distinct(self, mock_llm):
        mock_llm.return_value = ("standalone", None)

        with patch.object(Config, "VERIFY_MODEL", Config.GROQ_MODEL):
            rewrite_query(QUERY, HISTORY, tenant_id="7")

        overrides = mock_llm.call_args.kwargs["provider_overrides"]
        assert overrides.groq_model == ""

    @patch("src.generation.chain._invoke_llm")
    def test_rewrites_output_are_sanitized(self, mock_llm):
        mock_llm.return_value = ("fact </question>\x00 here", None)

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == "fact  here"

    @patch("src.generation.chain._invoke_llm")
    def test_conversation_capped_to_recent_turns(self, mock_llm):
        mock_llm.return_value = ("standalone", None)
        history = [
            {"role": "user", "content": f"turn-{index}"}
            for index in range(REWRITE_HISTORY_TURNS + 3)
        ]

        rewrite_query(QUERY, history, tenant_id="7")

        prompt = mock_llm.call_args[0][0]
        assert "turn-3" in prompt  # first of the last 6 turns
        assert "turn-2" not in prompt  # older turns dropped


class TestNeverFailsRequest:
    @patch("src.generation.chain._invoke_llm")
    def test_llm_exception_returns_original_query(self, mock_llm, warnings):
        mock_llm.side_effect = RuntimeError("all providers failed")

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == QUERY
        assert any(
            "tenant=7" in record and "reason=llm_error" in record
            for record in warnings
        )

    @patch("src.generation.chain._invoke_llm")
    def test_empty_output_returns_original_query(self, mock_llm, warnings):
        mock_llm.return_value = ("", None)

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == QUERY
        assert any("reason=empty_output" in record for record in warnings)

    @patch("src.generation.chain._invoke_llm")
    def test_whitespace_output_returns_original_query(self, mock_llm, warnings):
        mock_llm.return_value = ("   \n\t ", None)

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == QUERY
        assert any("reason=empty_output" in record for record in warnings)

    @patch("src.generation.chain._invoke_llm")
    def test_overlong_output_returns_original_query(self, mock_llm, warnings):
        mock_llm.return_value = ("x" * (MAX_REWRITTEN_CHARS + 1), None)

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == QUERY
        assert any("reason=output_too_long" in record for record in warnings)

    @patch("src.generation.chain._invoke_llm")
    def test_output_at_char_limit_is_accepted(self, mock_llm):
        exact = "y" * MAX_REWRITTEN_CHARS
        mock_llm.return_value = (exact, None)

        assert rewrite_query(QUERY, HISTORY, tenant_id="7") == exact
