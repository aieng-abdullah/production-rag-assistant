"""Structured schema + quote verification tests (PLAN PR-4b-i)."""

import pytest
from pydantic import ValidationError

from src.generation.schema import (
    AnswerVerificationError,
    Citation,
    StructuredAnswer,
    parse_structured,
    structured_to_prose,
    verify_quotes,
)

CHUNKS = [
    {"text": "The Transformer uses stacked self-attention layers."},
    {"text": "Training ran for 3.5 days on eight GPUs.\nSecond line."},
]


def _answer(**overrides) -> StructuredAnswer:
    payload = {
        "claims": [
            {
                "text": "The model uses attention.",
                "citations": [{"source_id": 1, "quote": "stacked self-attention"}],
            }
        ]
    }
    payload.update(overrides)
    return StructuredAnswer.model_validate(payload)


class TestSchemaRules:
    def test_valid_claim_passes(self):
        answer = _answer()
        assert answer.claims[0].citations[0].source_id == 1
        assert answer.abstained is False

    def test_claim_without_citation_rejected(self):
        with pytest.raises(ValidationError):
            StructuredAnswer(claims=[{"text": "x", "citations": []}])

    def test_blank_claim_text_rejected(self):
        with pytest.raises(ValidationError):
            StructuredAnswer(
                claims=[{"text": "   ", "citations": [{"source_id": 1, "quote": "q"}]}]
            )

    def test_blank_quote_rejected(self):
        with pytest.raises(ValidationError):
            Citation(source_id=1, quote="   ")

    def test_source_id_below_one_rejected(self):
        with pytest.raises(ValidationError):
            Citation(source_id=0, quote="q")

    def test_empty_claims_without_abstain_rejected(self):
        with pytest.raises(ValidationError):
            StructuredAnswer(claims=[], abstained=False)

    def test_abstain_requires_reason(self):
        with pytest.raises(ValidationError):
            StructuredAnswer(claims=[], abstained=True, abstain_reason=None)

    def test_abstain_forbids_claims(self):
        """Pure abstention only — uncited claims cannot ride along (no laundering)."""
        with pytest.raises(ValidationError):
            StructuredAnswer(
                claims=[
                    {
                        "text": "sneaky claim",
                        "citations": [{"source_id": 1, "quote": "stacked"}],
                    }
                ],
                abstained=True,
                abstain_reason="I don't have enough information.",
            )


class TestVerifyQuotes:
    def test_exact_quote_verifies(self):
        assert verify_quotes(_answer(), CHUNKS) == []

    def test_case_and_whitespace_normalized(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "Training was fast.",
                    "citations": [{"source_id": 2, "quote": "training ran for  3.5 days"}],
                }
            ]
        )
        assert verify_quotes(answer, CHUNKS) == []

    def test_newlines_in_source_normalized(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "GPUs used.",
                    "citations": [{"source_id": 2, "quote": "GPUs.\nSecond line."}],
                }
            ]
        )
        assert verify_quotes(answer, CHUNKS) == []

    def test_quote_not_in_source_reported(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "Made up.",
                    "citations": [{"source_id": 1, "quote": "quantum blockchain"}],
                }
            ]
        )
        problems = verify_quotes(answer, CHUNKS)
        assert problems == [{"claim": 0, "source_id": 1, "issue": "quote_not_found"}]

    def test_quote_spanning_two_chunks_reported(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "Spans chunks.",
                    "citations": [
                        {"source_id": 1, "quote": "self-attention layers. Training ran"}
                    ],
                }
            ]
        )
        problems = verify_quotes(answer, CHUNKS)
        assert problems[0]["issue"] == "quote_not_found"

    def test_source_out_of_range_reported(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "Invented source.",
                    "citations": [{"source_id": 9, "quote": "anything"}],
                }
            ]
        )
        problems = verify_quotes(answer, CHUNKS)
        assert problems == [
            {"claim": 0, "source_id": 9, "issue": "source_out_of_range"}
        ]

    def test_abstained_has_no_quotes_to_check(self):
        answer = StructuredAnswer(
            claims=[], abstained=True, abstain_reason="No info."
        )
        assert verify_quotes(answer, CHUNKS) == []


class TestParseStructured:
    def test_plain_json(self):
        raw = '{"claims": [{"text": "t", "citations": [{"source_id": 1, "quote": "q"}]}]}'
        answer = parse_structured(raw)
        assert answer.claims[0].text == "t"

    def test_fenced_json(self):
        raw = '```json\n{"claims": [{"text": "t", "citations": [{"source_id": 1, "quote": "q"}]}]}\n```'
        assert parse_structured(raw).claims[0].text == "t"

    def test_json_wrapped_in_prose(self):
        raw = 'Here is my answer: {"claims": [{"text": "t", "citations": [{"source_id": 1, "quote": "q"}]}]} hope that helps'
        assert parse_structured(raw).claims[0].text == "t"

    def test_abstain_json_parses(self):
        raw = '{"claims": [], "abstained": true, "abstain_reason": "I don\'t have enough information to answer this question."}'
        answer = parse_structured(raw)
        assert answer.abstained is True

    def test_no_json_object_raises(self):
        with pytest.raises(AnswerVerificationError, match="no JSON object"):
            parse_structured("I refuse to answer.")

    def test_broken_json_raises(self):
        with pytest.raises(AnswerVerificationError, match="invalid JSON"):
            parse_structured('{"claims": [oops')

    def test_schema_violation_raises(self):
        with pytest.raises(AnswerVerificationError, match="schema violation"):
            parse_structured('{"claims": [{"text": "t", "citations": []}]}')

    def test_error_is_value_error(self):
        """Callers catching ValueError keep working."""
        with pytest.raises(ValueError):
            parse_structured("garbage")


class TestStructuredToProse:
    def test_claims_carry_source_tags(self):
        answer = StructuredAnswer(
            claims=[
                {
                    "text": "First point.",
                    "citations": [{"source_id": 2, "quote": "a"}, {"source_id": 1, "quote": "b"}],
                },
                {"text": "Second point.", "citations": [{"source_id": 1, "quote": "c"}]},
            ]
        )
        prose = structured_to_prose(answer)
        assert prose == "First point. [SOURCE 1][SOURCE 2] Second point. [SOURCE 1]"

    def test_abstain_returns_reason(self):
        reason = "I don't have enough information to answer this question based on the provided sources."
        answer = StructuredAnswer(claims=[], abstained=True, abstain_reason=f"  {reason}  ")
        assert structured_to_prose(answer) == reason

    def test_prose_satisfies_legacy_citation_shape(self):
        """Prose keeps [SOURCE N] markers so UI highlighting still works."""
        prose = structured_to_prose(_answer())
        assert "[SOURCE 1]" in prose
