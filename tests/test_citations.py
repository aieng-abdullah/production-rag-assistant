"""CitedAnswer DTO + citation prompt contract (PLAN PR-4b).

The old per-sentence prose regex validator is retired — claims are validated
by src/generation.schema (see tests/test_structured_schema.py).
"""

from src.generation.Citation_system import CitedAnswer, build_citation_prompt


def test_cited_answer_is_plain_dto():
    """No validation on assembly — prose comes from already-verified claims."""
    answer = CitedAnswer(answer="free-form prose without markers", sources=[])
    assert answer.answer == "free-form prose without markers"
    assert answer.sources == []


def test_build_prompt_contains_sources():
    chunks = [{"text": "hello world", "doc_id": "paper", "page_num": 1}]
    prompt = build_citation_prompt("what is this?", chunks)
    assert "[SOURCE 1]" in prompt


def test_prompt_contains_json_contract():
    prompt = build_citation_prompt("what is this?", [{"text": "t"}])
    assert '"claims"' in prompt
    assert "source_id" in prompt
    assert "word-for-word" in prompt


def test_contract_demands_pure_abstain_object():
    prompt = build_citation_prompt("q?", [{"text": "t"}])
    assert '"abstained": true' in prompt
    assert '"claims": []' in prompt


def test_contract_forbids_prose_response():
    prompt = build_citation_prompt("q?", [{"text": "t"}])
    assert "ONLY one JSON object" in prompt
