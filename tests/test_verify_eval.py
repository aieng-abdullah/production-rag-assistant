"""Citation regression eval metrics (PLAN PR-4b-v) — offline, no LLM."""

import pytest

from eval.verify_eval import (
    abstention_correct,
    check_gates,
    citation_precision,
    citation_recall,
    compute_metrics,
    evaluate_item,
    load_dataset,
    normalize,
)
from src.generation.Citation_system import CitedAnswer, Source


def _cited(claims=None, verification=None, sources=None) -> CitedAnswer:
    return CitedAnswer(
        answer="text",
        sources=sources
        if sources is not None
        else [
            Source(doc_id="doc1", page_num=1, text="the burden lies on the plaintiff", source_id=1)
        ],
        verification=verification
        if verification is not None
        else {"status": "verified", "per_claim": []},
        trace={"claims": claims} if claims is not None else {"claims": []},
    )


def test_normalize_collapses_whitespace_case():
    assert normalize("  The  BURDEN\nlies ") == "the burden lies"


def test_precision_all_valid():
    cited = _cited(
        claims=[
            {
                "text": "claim",
                "citations": [{"source_id": 1, "quote": "burden lies on the plaintiff"}],
            }
        ]
    )
    assert citation_precision(cited) == (1, 1)


def test_precision_rejects_quote_not_in_source():
    cited = _cited(
        claims=[
            {
                "text": "claim",
                "citations": [{"source_id": 1, "quote": "invented words"}],
            }
        ]
    )
    assert citation_precision(cited) == (0, 1)


def test_precision_rejects_unresolvable_source_id():
    cited = _cited(
        claims=[
            {
                "text": "claim",
                "citations": [{"source_id": 9, "quote": "burden"}],
            }
        ]
    )
    assert citation_precision(cited) == (0, 1)


def test_precision_empty_claims_is_zero_over_zero():
    assert citation_precision(_cited()) == (0, 0)


def test_recall_hit_requires_cited_source_doc_and_quote():
    cited = _cited(
        claims=[
            {"text": "claim", "citations": [{"source_id": 1, "quote": "anything"}]}
        ]
    )
    assert citation_recall(cited, "BURDEN lies", "doc1") is True
    assert citation_recall(cited, "burden lies", "other-doc") is False
    assert citation_recall(cited, "missing fragment", "doc1") is False


def test_recall_misses_when_expected_source_not_cited():
    cited = _cited(
        claims=[
            {"text": "claim", "citations": [{"source_id": 2, "quote": "x"}]}
        ],
        sources=[
            Source(doc_id="doc1", page_num=1, text="the burden lies", source_id=1),
            Source(doc_id="doc2", page_num=2, text="unrelated", source_id=2),
        ],
    )
    assert citation_recall(cited, "burden lies", "doc1") is False


def test_abstention_expectations():
    abstained = _cited(verification={"status": "abstained", "per_claim": []})
    verified = _cited(verification={"status": "verified", "per_claim": []})

    assert abstention_correct(abstained, "abstain") is True
    assert abstention_correct(verified, "abstain") is False
    assert abstention_correct(verified, "cite") is True
    assert abstention_correct(abstained, "cite") is False
    no_verification = CitedAnswer(answer="x", sources=[], verification=None)
    assert abstention_correct(no_verification, "cite") is False


def test_compute_metrics_pooling():
    results = [
        {
            "expect": "cite",
            "precision_valid": 2,
            "precision_total": 2,
            "recall_hit": True,
            "abstention_correct": True,
        },
        {
            "expect": "abstain",
            "precision_valid": 0,
            "precision_total": 0,
            "recall_hit": None,
            "abstention_correct": False,
        },
    ]
    metrics = compute_metrics(results)
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0  # 1 of 1 cite item
    assert metrics["abstention_accuracy"] == 0.5


def test_check_gates_pass_and_fail():
    gates = check_gates(
        {"citation_precision": 1.0, "citation_recall": 0.5, "abstention_accuracy": 1.0}
    )
    assert gates["citation_precision"]["status"] == "PASS"
    assert gates["citation_recall"]["status"] == "FAIL"
    assert gates["abstention_accuracy"]["status"] == "PASS"


def test_evaluate_item_wires_everything():
    cited = _cited(
        claims=[
            {"text": "claim", "citations": [{"source_id": 1, "quote": "burden lies"}]}
        ]
    )
    item = {
        "id": "x",
        "class": "conflicting-sources",
        "expect": "cite",
        "expected_quote": "burden lies",
        "expected_doc_id": "doc1",
    }
    outcome = evaluate_item(item, cited)
    assert outcome["precision_valid"] == 1
    assert outcome["recall_hit"] is True
    assert outcome["abstention_correct"] is True


def test_dataset_schema_and_classes():
    data = load_dataset()
    assert len(data) >= 6
    classes = {item["class"] for item in data}
    # PLAN PR-4b-v failure classes must stay covered.
    assert {"proviso", "missing-cross-ref", "conflicting-sources", "out-of-corpus"} <= classes
    assert {item["workspace"] for item in data} == {"legal", "academic"}
    assert any(item["expect"] == "abstain" for item in data)
    assert any(item["expect"] == "cite" for item in data)


def test_dataset_rejects_bad_expect(tmp_path):
    bad = tmp_path / "d.json"
    bad.write_text('[{"id": "x", "class": "c", "workspace": "legal", "question": "q", "expect": "maybe"}]')
    with pytest.raises(ValueError, match="cite|abstain"):
        load_dataset(str(bad))
