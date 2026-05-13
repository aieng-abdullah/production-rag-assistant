import pytest
from src.generation.Citation_system import CitedAnswer, Source, build_citation_prompt

def test_valid_citation_accepted():
    answer = CitedAnswer(
        answer="The sky is blue [SOURCE 1].",
        sources=[Source(doc_id="paper", page_num=1, text="The sky is blue.")]
    )
    assert answer.answer is not None

def test_missing_citation_rejected():
    with pytest.raises(Exception):
        CitedAnswer(answer="The sky is blue.", sources=[])

def test_build_prompt_contains_sources():
    chunks = [{"text": "hello world", "doc_id": "paper", "page_num": 1}]
    prompt = build_citation_prompt("what is this?", chunks)
    assert "[SOURCE 1]" in prompt
