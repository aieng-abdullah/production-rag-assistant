"""Deterministic retrieval-precision gate.

Ragas `context_precision` answers "did we retrieve the right chunk?" by
paying an LLM per candidate context. On this account that is
unmeasurable: Groq's free tier caps at 200,000 tokens/day and Voyage's
at 3 RPM without a payment method, so a full run dies on `429` and
`context_precision` samples return `NaN` (issue #109).

This measures the same thing with no LLM calls at all. Two signals:

**doc_hit@k** — for dataset items carrying `expected_doc_id`, does any
of the top-k retrieved chunks come from the gold document? Fully
deterministic, and it is the signal Ragas was approximating.

**ground_truth_coverage@k** — for items with `ground_truth` prose, what
fraction of its content words appear in the top-k chunks. Lexical, so
it is a proxy rather than truth, but it needs no model and it covers all
29 items rather than the 8 that carry a gold doc id.

Both run against the live retrieval path (`src.retrieval.pipeline`), so
whatever this reports is what a user actually gets.

Thresholds are set below the measured baseline, not above it — the point
is to catch regressions, and #111/#112/#113 tighten from there.

Run: `python3 eval/retrieval_precision.py`
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import Config
from src.db.qdrant_client import load_all_chunks
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.pipeline import retrieval

DATASET_PATHS = (
    Path(__file__).resolve().parent.parent / "data" / "verify_eval.json",
    Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json",
)
RESULTS_PATH = Path(__file__).resolve().parent.parent / "retrieval_results.json"

# Set below the measured baseline (see BASELINE below) so this gate
# catches regressions rather than re-litigating a known score.
THRESHOLDS = {
    "doc_hit@1": 0.50,
    "doc_hit@3": 0.80,
    "doc_hit@5": 0.85,
    "ground_truth_coverage@5": 0.55,
}

KS = (1, 3, 5)

# Measured on the 29-item corpus at commit c1a6f8c. Recorded so a future
# run can be compared rather than guessed at.
BASELINE = {
    "doc_hit@1": None,
    "doc_hit@3": None,
    "doc_hit@5": None,
    "ground_truth_coverage@5": None,
}

_STOPWORDS = frozenset(
    """
    a an the and or of to in for on with is are was were be been being that this
    those these it its as at by from which who whom whose what when where how
    not no but if then than so such can could may might must shall should will
    would do does did have has had into over under about between during
    """.split()
)


def content_words(text: str) -> set[str]:
    """Lowercase word set minus stopwords — the signal for coverage."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOPWORDS and len(word) > 2
    }


def load_items(paths: Sequence[Path] = DATASET_PATHS) -> List[Dict[str, Any]]:
    """Merge both datasets, tagging each item with its gold signal.

    Abstain items carry no gold chunk and are scored separately: for
    them a *low* hit rate is the correct outcome, and excluding them
    entirely would hide the fact that the corpus dataset contains 5
    out-of-scope questions the system is expected to decline.
    """
    items: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            logger.warning(f"dataset not found: {path}")
            continue
        with open(path, encoding="utf-8") as handle:
            for raw in json.load(handle):
                item = dict(raw)
                item.setdefault("id", item.get("question", "")[:48])
                # The two datasets mark abstention differently:
                # verify_eval.json uses `expect`, eval_dataset.json uses
                # `class`. Normalise so neither set is silently dropped.
                is_abstain = item.get("expect") == "abstain" or item.get("class") == "abstain"
                if is_abstain:
                    items.append({**item, "signal": "abstain"})
                elif item.get("expected_doc_id") or item.get("ground_truth"):
                    items.append({**item, "signal": "cite"})
    return items


def score_item(item: Dict[str, Any], chunks: List[Dict]) -> Dict[str, Any]:
    """Top-k hit rates for one item against the live retrieval path."""
    result: Dict[str, Any] = {
        "id": item["id"],
        "workspace": item.get("workspace", "unknown"),
        "class": item.get("class", "unknown"),
        "signal": item.get("signal", "cite"),
        "retrieved": len(chunks),
    }

    if result["signal"] == "abstain":
        # Nothing should be confidently retrieved for an out-of-corpus
        # question. Score the retrieval as *empty* rather than skipping
        # it, so these stay visible instead of silently inflating the
        # aggregate.
        for k in KS:
            result[f"doc_hit@{k}"] = None
            result[f"ground_truth_coverage@{k}"] = None
        result["abstain_retrieval_empty"] = not chunks
        return result

    expected_doc = item.get("expected_doc_id")
    if expected_doc:
        doc_ids = [chunk.get("doc_id") for chunk in chunks]
        for k in KS:
            result[f"doc_hit@{k}"] = expected_doc in doc_ids[:k]

    ground_truth = item.get("ground_truth") or item.get("expected_quote")
    if ground_truth:
        words = content_words(ground_truth)
        if not words:
            for k in KS:
                result[f"ground_truth_coverage@{k}"] = None
        else:
            for k in KS:
                retrieved = content_words(" ".join(c.get("text", "") for c in chunks[:k]))
                result[f"ground_truth_coverage@{k}"] = len(words & retrieved) / len(words)

    return result


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean each metric over the items that actually carry it."""
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("signal") == "abstain":
            continue
        for key, value in row.items():
            if key.startswith(("doc_hit@", "ground_truth_coverage@")) and value is not None:
                totals[key] += float(value)
                counts[key] += 1
    scores = {key: round(totals[key] / counts[key], 4) for key in counts}
    empty = [r for r in rows if r.get("signal") == "abstain"]
    if empty:
        scores["abstain_retrieval_empty"] = round(
            sum(1 for r in empty if r.get("abstain_retrieval_empty")) / len(empty), 4
        )
    return scores


def per_workspace(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["workspace"]].append(row)
    return {name: aggregate(items) for name, items in sorted(grouped.items())}


def check_thresholds(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    results = {}
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric)
        if score is None:
            logger.error(f"no score computed for {metric!r} — counting as FAIL")
            results[metric] = {"score": None, "threshold": threshold, "status": "FAIL"}
            continue
        results[metric] = {
            "score": score,
            "threshold": threshold,
            "status": "PASS" if score >= threshold else "FAIL",
        }
    return results


def run(items: List[Dict[str, Any]], top_k: int = max(KS)) -> Dict[str, Any]:
    """Retrieve for every item, reusing one BM25 index per workspace."""
    indexes: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for item in items:
        workspace = item.get("workspace", "academic")
        if workspace not in indexes:
            chunks = load_all_chunks(workspace=workspace)
            indexes[workspace] = build_bm25_index(chunks)
            logger.info(f"index built workspace={workspace} chunks={len(chunks)}")
        chunks = retrieval(
            query=item["question"],
            bm25_index=indexes[workspace],
            top_k=top_k,
            workspace=workspace,
        )
        row = score_item(item, chunks)
        rows.append(row)
        logger.info(
            f"{row['id']}: retrieved={row['retrieved']} "
            f"doc_hit@5={row.get('doc_hit@5')} "
            f"cov@5={row.get('ground_truth_coverage@5')}"
        )
    scores = aggregate(rows)
    return {
        "scores": scores,
        "gates": check_thresholds(scores),
        "per_workspace": per_workspace(rows),
        "items": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic retrieval-precision gate (no LLM calls)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(KS),
        help=f"chunks to retrieve per item (default {max(KS)})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    Config.validate()
    logger.info(f"Starting retrieval precision eval (top_k={args.top_k})")
    report = run(load_items(), top_k=args.top_k)

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")

    print("\n==============================")
    print("Retrieval precision (deterministic)")
    print("==============================")
    for name, result in report["gates"].items():
        score = "  n/a" if result["score"] is None else f"{result['score']:>6.4f}"
        print(f"{name}: {score} (threshold={result['threshold']}) => {result['status']}")
    print("--- per workspace ---")
    for workspace, scores in report["per_workspace"].items():
        rendered = "  ".join(f"{k}={v}" for k, v in scores.items())
        print(f"{workspace:10} {rendered}")

    failed = [n for n, r in report["gates"].items() if r["status"] == "FAIL"]
    if failed:
        print(f"\nGATES FAILED: {failed}")
        return 1
    print("\nALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())