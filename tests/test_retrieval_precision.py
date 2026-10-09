"""Deterministic retrieval-precision gate tests (issue #109).

No live retrieval here — scoring, aggregation and gating are pure
functions over synthetic chunks.
"""

from eval.retrieval_precision import (
    THRESHOLDS,
    aggregate,
    check_thresholds,
    content_words,
    load_items,
    per_workspace,
    score_item,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _cite(**overrides):
    item = {
        "id": "cite-1",
        "workspace": "legal",
        "class": "proviso",
        "signal": "cite",
        "question": "q?",
        "expected_doc_id": "act1872",
        "ground_truth": "The burden of proof lies on the plaintiff",
    }
    item.update(overrides)
    return item


def test_content_words_drops_stopwords_and_short_tokens():
    words = content_words("The burden of proof is on the plaintiff")
    assert "burden" in words and "proof" in words and "plaintiff" in words
    assert "the" not in words and "of" not in words and "is" not in words


def test_doc_hit_is_rank_sensitive():
    """Gold at rank 2: missed at k=1, hit at k=3."""
    chunks = [
        {"doc_id": "other", "text": "unrelated"},
        {"doc_id": "act1872", "text": "burden of proof lies on the plaintiff"},
    ]
    row = score_item(_cite(), chunks)
    assert row["doc_hit@1"] is False
    assert row["doc_hit@3"] is True


def test_coverage_is_one_when_ground_truth_present():
    chunks = [{"doc_id": "a", "text": "the burden of proof lies on the plaintiff"}]
    assert score_item(_cite(), chunks)["ground_truth_coverage@5"] == 1.0


def test_coverage_is_zero_when_nothing_overlaps():
    chunks = [{"doc_id": "a", "text": "capital gains tax thresholds for 2024"}]
    assert score_item(_cite(), chunks)["ground_truth_coverage@5"] == 0.0


def test_empty_retrieval_scores_zero_not_missing():
    row = score_item(_cite(), [])
    assert row["doc_hit@5"] is False
    assert row["ground_truth_coverage@5"] == 0.0


class TestAbstainItems:
    """The two datasets mark abstention differently; both must survive."""

    def test_expect_field_marks_abstain(self):
        row = score_item({"id": "a", "expect": "abstain", "signal": "abstain"}, [])
        assert row["signal"] == "abstain"
        assert row["abstain_retrieval_empty"] is True

    def test_abstain_never_contributes_to_cite_averages(self):
        abstain = score_item({"id": "a", "signal": "abstain"}, [])
        cite = score_item(_cite(), [{"doc_id": "act1872", "text": "burden of proof"}])
        scores = aggregate([cite, abstain])
        assert scores["doc_hit@5"] == 1.0

    def test_abstain_reports_whether_retrieval_was_empty(self):
        rows = [
            score_item({"id": "a", "signal": "abstain"}, []),
            score_item({"id": "b", "signal": "abstain"}, [{"doc_id": "x", "text": "y"}]),
        ]
        assert aggregate(rows)["abstain_retrieval_empty"] == 0.5


class TestDatasetLoading:
    def test_real_datasets_load_both_signals(self):
        items = load_items()
        assert items, "expected dataset items"
        signals = {item["signal"] for item in items}
        assert signals == {"cite", "abstain"}

    def test_eval_dataset_abstain_items_are_not_dropped(self):
        """eval_dataset.json uses `class: abstain`, verify_eval uses `expect`.

        Reading only `expect` silently loses 6 items.
        """
        items = load_items()
        by_question = {item.get("question", ""): item for item in items}
        assert by_question["What is the capital of France?"]["signal"] == "abstain"


class TestGates:
    def test_missing_metric_fails_rather_than_raising(self):
        results = check_thresholds({})
        assert all(r["status"] == "FAIL" for r in results.values())
        assert all(r["score"] is None for r in results.values())

    def test_partial_scores_still_gate_the_rest(self):
        results = check_thresholds({name: 1.0 for name in THRESHOLDS})
        assert all(r["status"] == "PASS" for r in results.values())

    def test_failing_metric_reported(self):
        scores = {name: 0.0 for name in THRESHOLDS}
        results = check_thresholds(scores)
        assert all(r["status"] == "FAIL" for r in results.values())


def test_per_workspace_splits_rows():
    rows = [
        score_item(_cite(workspace="legal"), [{"doc_id": "act1872", "text": "burden"}]),
        score_item(_cite(id="c2", workspace="academic"), [{"doc_id": "x", "text": "unrelated"}]),
    ]
    split = per_workspace(rows)
    assert split["legal"]["doc_hit@5"] == 1.0
    assert split["academic"]["doc_hit@5"] == 0.0