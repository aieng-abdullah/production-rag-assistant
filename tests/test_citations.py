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


def test_abstention_accepted_without_citation():
    """Refusal is first-class: profile prompts require it on weak evidence."""
    answer = CitedAnswer(
        answer=(
            "I don't have enough information to answer this question "
            "based on the provided sources."
        ),
        sources=[],
    )
    assert "enough information" in answer.answer


def test_uncited_non_abstain_still_rejected():
    with pytest.raises(Exception, match="citation"):
        CitedAnswer(answer="Liability follows from remoteness.", sources=[])


def test_abstention_phrase_is_case_insensitive():
    answer = CitedAnswer(
        answer="I DO NOT HAVE ENOUGH INFORMATION TO ANSWER THIS QUESTION BASED ON THE PROVIDED SOURCES.",
        sources=[],
    )
    assert answer.sources == []


def test_abstain_phrase_cannot_launder_uncited_claims():
    """Pure abstention only — refusal + claims with zero markers must fail."""
    with pytest.raises(Exception, match="citation"):
        CitedAnswer(
            answer=(
                "I don't have enough information to answer this question "
                "based on the provided sources. The defendant is liable "
                "for all losses."
            ),
            sources=[],
        )
