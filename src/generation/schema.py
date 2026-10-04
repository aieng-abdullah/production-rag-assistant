"""Structured answer schema + deterministic quote verification (PLAN PR-4b-i).

The model emits JSON claims; each claim carries verbatim quotes with a
`source_id` pointing at one retrieved chunk. `verify_quotes` proves every
quote exists in its cited chunk (whitespace/case-normalized containment) and
range-checks ids. `doc_id`/`page_num` are resolved server-side from retrieval
chunks — the model never writes them.

Parsing (`parse_structured`) accepts plain JSON, ```json fences, and prose
around the object; anything unparseable or schema-invalid raises
`AnswerVerificationError` (the chain's repair loop re-asks once, then fails).
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

__all__ = [
    "AnswerVerificationError",
    "Citation",
    "Claim",
    "StructuredAnswer",
    "parse_structured",
    "structured_to_prose",
    "verify_quotes",
]


class AnswerVerificationError(ValueError):
    """Structured output failed parsing, schema, or quote verification."""


class Citation(BaseModel):
    source_id: int = Field(ge=1)
    quote: str = Field(min_length=1)

    @field_validator("quote")
    @classmethod
    def _strip_quote(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("quote must not be blank")
        return stripped


class Claim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("claim text must not be blank")
        return stripped


class StructuredAnswer(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.abstained:
            if not (self.abstain_reason and self.abstain_reason.strip()):
                raise ValueError("abstain_reason required when abstained")
            if self.claims:
                raise ValueError("abstained answers must carry no claims")
        elif not self.claims:
            raise ValueError("at least one claim required unless abstained")
        return self


def _normalize(text: str) -> str:
    """Whitespace-collapse + casefold so quote checks survive formatting."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify_quotes(structured: StructuredAnswer, chunks: list[dict]) -> list[dict]:
    """Deterministic check: every quote exists in its cited chunk.

    Returns problem dicts (`claim` index, `source_id`, `issue`) — empty list
    means fully verified. Abstentions carry no quotes by construction.
    """
    if structured.abstained:
        return []

    problems: list[dict] = []
    chunk_count = len(chunks)
    for claim_index, claim in enumerate(structured.claims):
        for citation in claim.citations:
            if citation.source_id > chunk_count:
                problems.append(
                    {
                        "claim": claim_index,
                        "source_id": citation.source_id,
                        "issue": "source_out_of_range",
                    }
                )
                continue
            source_text = _normalize(str(chunks[citation.source_id - 1].get("text", "")))
            if _normalize(citation.quote) not in source_text:
                problems.append(
                    {
                        "claim": claim_index,
                        "source_id": citation.source_id,
                        "issue": "quote_not_found",
                    }
                )
    return problems


def parse_structured(raw: str) -> StructuredAnswer:
    """Parse model output into a StructuredAnswer (fences/prose tolerated)."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start == -1:
        raise AnswerVerificationError("model output contains no JSON object")
    end = text.rfind("}")
    payload = text[start : end + 1] if end != -1 else text[start:]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AnswerVerificationError(f"invalid JSON: {exc}") from exc
    try:
        return StructuredAnswer.model_validate(data)
    except ValidationError as exc:
        raise AnswerVerificationError(f"schema violation: {exc}") from exc


def structured_to_prose(structured: StructuredAnswer) -> str:
    """Assemble the legacy `answer` string from claims (UI/API compat)."""
    if structured.abstained:
        return (structured.abstain_reason or "").strip()
    parts = []
    for claim in structured.claims:
        source_ids = sorted({c.source_id for c in claim.citations})
        tags = "".join(f"[SOURCE {i}]" for i in source_ids)
        parts.append(f"{claim.text} {tags}")
    return " ".join(parts)
