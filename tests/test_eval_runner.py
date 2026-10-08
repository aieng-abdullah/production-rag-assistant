"""RAGAS eval runner helpers — offline, no live API calls.

`ragas.evaluate` and the RAG pipeline are never invoked here; only the pure
score-extraction / threshold-gate / reporting helpers are exercised.
"""

import json

import pandas as pd
import pytest

from eval.eval_runner import (
    THRESHOLDS,
    attach_metric_scores,
    check_thresholds,
    extract_metric_scores,
    gates_failed,
    load_eval_dataset,
    mean_metric_scores,
    parse_args,
    per_sample_report,
    resolve_metric_column,
    resolve_workspace,
)


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestResolveMetricColumn:
    def test_exact_match(self):
        assert resolve_metric_column(["question", "faithfulness"], "faithfulness") == "faithfulness"

    def test_normalised_match_ignores_case_and_separators(self):
        assert (
            resolve_metric_column(["Answer Relevancy"], "answer_relevancy")
            == "Answer Relevancy"
        )

    def test_missing_column_raises_loudly(self):
        with pytest.raises(KeyError, match="not found"):
            resolve_metric_column(["question", "answer"], "context_precision")

    def test_ambiguous_matches_raise(self):
        with pytest.raises(KeyError, match="multiple columns"):
            resolve_metric_column(["Faithfulness", "FAITHFULNESS"], "faithfulness")


class TestExtractMetricScores:
    def test_explicit_per_metric_means(self):
        df = _frame(
            faithfulness=[1.0, 0.5],
            answer_relevancy=[0.8, 0.6],
            unrelated_numeric=[999.0, 999.0],
        )
        scores = extract_metric_scores(df, ["faithfulness", "answer_relevancy"])
        assert scores["faithfulness"] == pytest.approx(0.75)
        assert scores["answer_relevancy"] == pytest.approx(0.7)
        assert "unrelated_numeric" not in scores

    def test_missing_metric_column_raises(self):
        df = _frame(faithfulness=[1.0])
        with pytest.raises(KeyError, match="context_recall"):
            extract_metric_scores(df, ["faithfulness", "context_recall"])

    def test_non_numeric_values_skipped(self):
        df = _frame(context_precision=[0.5, None])
        scores = extract_metric_scores(df, ["context_precision"])
        assert scores["context_precision"] == pytest.approx(0.5)

    def test_column_with_no_numeric_scores_raises(self):
        df = _frame(context_recall=[None, None])
        with pytest.raises(ValueError, match="no numeric scores"):
            extract_metric_scores(df, ["context_recall"])


class TestMeanMetricScores:
    def test_single_frame(self):
        df = _frame(faithfulness=[1.0, 0.0])
        assert mean_metric_scores([df], ["faithfulness"])["faithfulness"] == pytest.approx(0.5)

    def test_count_weighted_mean_across_frames(self):
        cite = _frame(faithfulness=[1.0, 1.0])
        abstain = _frame(faithfulness=[0.0])
        scores = mean_metric_scores([cite, abstain], ["faithfulness"])
        assert scores["faithfulness"] == pytest.approx(2 / 3)

    def test_metric_absent_from_every_frame_raises(self):
        cite = _frame(faithfulness=[1.0])
        with pytest.raises(KeyError, match="context_precision"):
            mean_metric_scores([cite], list(THRESHOLDS))

    def test_metric_present_in_at_least_one_frame_is_enough(self):
        cite = _frame(faithfulness=[1.0], context_precision=[0.8])
        abstain = _frame(faithfulness=[0.5])
        scores = mean_metric_scores([cite, abstain], ["faithfulness", "context_precision"])
        assert scores["context_precision"] == pytest.approx(0.8)


class TestAttachMetricScores:
    def test_rows_align_positionally(self):
        samples = [{"answer": "a"}, {"answer": "b"}]
        df = _frame(faithfulness=[1.0, 0.25], context_recall=[0.5, 0.75])
        attach_metric_scores(samples, df, ["faithfulness", "context_recall"])
        assert samples[0]["scores"] == {"faithfulness": 1.0, "context_recall": 0.5}
        assert samples[1]["scores"] == {"faithfulness": 0.25, "context_recall": 0.75}

    def test_absent_metric_recorded_as_null(self):
        samples = [{"answer": "a"}]
        df = _frame(faithfulness=[1.0])
        attach_metric_scores(samples, df, ["faithfulness", "context_precision"])
        assert samples[0]["scores"]["context_precision"] is None

    def test_row_count_mismatch_raises(self):
        samples = [{"answer": "a"}]
        with pytest.raises(ValueError, match="rows"):
            attach_metric_scores(samples, _frame(faithfulness=[1.0, 0.0]), ["faithfulness"])


class TestThresholdGates:
    def test_check_thresholds_pass_and_fail(self):
        results = check_thresholds(
            {"faithfulness": 0.9, "answer_relevancy": 0.5,
             "context_precision": 0.7, "context_recall": 1.0}
        )
        assert results["faithfulness"]["status"] == "PASS"
        assert results["answer_relevancy"]["status"] == "FAIL"
        assert results["context_precision"]["status"] == "PASS"
        assert results["context_recall"]["status"] == "PASS"

    def test_check_thresholds_missing_score_raises(self):
        with pytest.raises(KeyError, match="faithfulness"):
            check_thresholds({})

    def test_boundary_score_passes(self):
        results = check_thresholds(
            {name: threshold for name, threshold in THRESHOLDS.items()}
        )
        assert all(result["status"] == "PASS" for result in results.values())

    def test_gates_failed_lists_only_failures(self):
        results = check_thresholds(
            {"faithfulness": 0.9, "answer_relevancy": 0.5,
             "context_precision": 0.5, "context_recall": 1.0}
        )
        assert gates_failed(results) == ["answer_relevancy", "context_precision"]

    def test_gates_failed_empty_when_all_pass(self):
        results = check_thresholds(
            {name: threshold for name, threshold in THRESHOLDS.items()}
        )
        assert gates_failed(results) == []


class TestWorkspaceResolution:
    def test_default_workspace(self):
        assert resolve_workspace({"question": "q"}, "academic") == "academic"

    def test_per_sample_override(self):
        item = {"question": "q", "workspace": "legal"}
        assert resolve_workspace(item, "academic") == "legal"

    def test_unknown_workspace_raises(self):
        item = {"question": "q", "workspace": "medical"}
        with pytest.raises(ValueError, match="unknown workspace"):
            resolve_workspace(item, "academic")


class TestPerSampleReport:
    def test_report_contains_scores_and_context_count(self):
        samples = [
            {
                "question": "q1",
                "answer": "a1",
                "contexts": ["c1", "c2"],
                "ground_truth": "gt1",
                "workspace": "academic",
                "class": "cite",
                "scores": {"faithfulness": 1.0},
            },
            {
                "question": "q2",
                "answer": "a2",
                "contexts": [],
                "ground_truth": "gt2",
                "workspace": "legal",
                "class": "abstain",
            },
        ]
        report = per_sample_report(samples)
        assert report[0]["contexts_count"] == 2
        assert report[0]["scores"] == {"faithfulness": 1.0}
        assert report[1]["contexts_count"] == 0
        assert report[1]["scores"] == {}
        assert report[1]["class"] == "abstain"
        assert "contexts" not in report[0]

    def test_report_is_json_serialisable(self):
        samples = [{
            "question": "q", "answer": "a", "contexts": ["c"],
            "ground_truth": "gt", "workspace": "academic", "class": "cite",
            "scores": {"faithfulness": None},
        }]
        json.dumps(per_sample_report(samples))


class TestCliAndDatasetLoading:
    def test_default_workspace_is_academic(self):
        assert parse_args([]).workspace == "academic"

    def test_workspace_flag_parsed(self):
        assert parse_args(["--workspace", "legal"]).workspace == "legal"

    def test_invalid_workspace_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--workspace", "medical"])

    def test_load_eval_dataset_reads_json(self, tmp_path):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps([{"question": "q", "ground_truth": "gt"}]))
        data = load_eval_dataset(str(path))
        assert data == [{"question": "q", "ground_truth": "gt"}]

    def test_load_eval_dataset_rejects_item_without_question(self, tmp_path):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps([{"ground_truth": "gt"}]))
        with pytest.raises(ValueError, match="missing 'question'"):
            load_eval_dataset(str(path))


class TestGenerationFailureHandling:
    def test_build_eval_samples_records_failure_and_continues(self, monkeypatch):
        from src.generation.schema import AnswerVerificationError

        import eval.eval_runner as runner

        calls = []

        def fake_run_rag(question, workspace):
            calls.append(question)
            if question == "bad?":
                raise AnswerVerificationError("quote_not_found")
            return {"answer": "a", "contexts": ["c"]}

        monkeypatch.setattr(runner, "run_rag", fake_run_rag)
        dataset = [
            {"question": "bad?", "ground_truth": "gt1"},
            {"question": "good?", "ground_truth": "gt2"},
        ]
        samples = runner.build_eval_samples(dataset, "academic")
        assert len(calls) == 2
        assert samples[0]["answer"] is None
        assert "quote_not_found" in samples[0]["generation_error"]
        assert samples[1]["answer"] == "a"
        assert samples[1]["generation_error"] is None

    def test_evaluate_dataset_excludes_failed_samples(self, monkeypatch):
        import eval.eval_runner as runner

        def fake_evaluate(samples, metrics):
            assert len(samples) == 1
            assert samples[0]["question"] == "good?"
            return pd.DataFrame({metric: [0.9] for metric in metrics})

        monkeypatch.setattr(runner, "evaluate_samples", fake_evaluate)
        good = {
            "question": "good?", "answer": "a", "contexts": ["c"],
            "ground_truth": "gt", "workspace": "academic", "class": "cite",
            "generation_error": None,
        }
        bad = {**good, "question": "bad?", "generation_error": "boom"}
        scores = runner.evaluate_dataset([good, bad])
        assert scores == {metric: pytest.approx(0.9) for metric in THRESHOLDS}
