"""RAGAS quality eval (local-only gate — not in CI).

Runs Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall over
`data/eval_dataset.json`, compares per-metric means against thresholds,
writes `results.json`, and exits 1 when any gate fails.

    python3 eval/eval_runner.py [--workspace academic|legal]

Requires GROQ + VOYAGE + QDRANT keys in `.env` and an ingested corpus.
Samples marked `"class": "abstain"` are judged only on the metrics that do
not read `ground_truth` (faithfulness, answer_relevancy).
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# Voyage trial paces at 3 RPM — serialise ragas metric workers. Must be set
# before ragas is imported below.
os.environ["RAGAS_MAX_WORKERS"] = "1"

# Run as a script (`python3 eval/eval_runner.py`), the repo root is not on
# sys.path — and a stale PYTHONPATH could shadow `src` with another project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from loguru import logger
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

from src.config import Config
from src.db.qdrant_client import load_all_chunks
from src.generation.chain import generate
from src.generation.schema import AnswerVerificationError
from src.ingestion import embedder as _embedder
from src.ingestion.embedder import _get_model as get_embedding_model
from src.retrieval.bm25_index import build_bm25_index

EVAL_DATASET_PATH = "data/eval_dataset.json"
RESULTS_PATH = "results.json"
WORKSPACES = ("academic", "legal")

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}

METRIC_NAMES = tuple(THRESHOLDS)

# Ground-truth metrics are meaningless for out-of-corpus/abstain samples.
ABSTAIN_METRICS = ("faithfulness", "answer_relevancy")

# The generation chain's internal repair loop fails on unlucky draws
# (quote mismatches); a fresh sample usually verifies.
GENERATE_ATTEMPTS = 3

# Columns ragas consumes — extra bookkeeping keys are stripped before the
# dataset is handed to `evaluate()`.
RAGAS_INPUT_COLUMNS = ("question", "answer", "contexts", "ground_truth")

# Voyage trial accounts allow 3 RPM on /embeddings. Retrieval query embeds
# and the ragas answer-relevancy embeds share that pool and neither path
# paces itself (embed_chunks paces only ingestion batches), so the eval run
# serialises every embeddings API call through _post_embeddings.
VOYAGE_EMBED_PACE_S = float(os.getenv("VOYAGE_EMBED_PACE_S", "21"))

_pace_lock = threading.Lock()
_pace_next_embed = 0.0
_voyage_paced = False


def wait_for_voyage_slot() -> None:
    """Block until this process may make the next embeddings API call."""
    global _pace_next_embed
    with _pace_lock:
        now = time.monotonic()
        wait = _pace_next_embed - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _pace_next_embed = now + VOYAGE_EMBED_PACE_S


def install_voyage_pacing() -> None:
    """Wrap the embeddings transport with the rate-limit slot waiter.

    Installed from main() only — unit tests import this module but never
    make API calls.
    """
    global _voyage_paced
    if _voyage_paced:
        return
    original = _embedder._post_embeddings

    def paced(*args: Any, **kwargs: Any) -> Any:
        wait_for_voyage_slot()
        return original(*args, **kwargs)

    _embedder._post_embeddings = paced
    _voyage_paced = True
    logger.info(f"Voyage embedding pacing installed ({VOYAGE_EMBED_PACE_S}s/call)")

_llm = None
_embeddings = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = LangchainLLMWrapper(
            ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL)
        )
    return _llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    return _embeddings


def _build_metrics(names: Sequence[str]) -> List[Any]:
    factories = {
        "faithfulness": lambda: Faithfulness(llm=_get_llm()),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=_get_llm(), embeddings=_get_embeddings()
        ),
        "context_precision": lambda: ContextPrecision(llm=_get_llm()),
        "context_recall": lambda: ContextRecall(llm=_get_llm()),
    }
    metrics = []
    for name in names:
        if name not in factories:
            raise KeyError(
                f"unknown metric {name!r}; known metrics: {sorted(factories)}"
            )
        metrics.append(factories[name]())
    return metrics


_bm25_indices: Dict[str, Any] = {}


def _get_pipeline(workspace: str):
    # Same workspace scope as generate()'s default vector filter —
    # a tenant-wide index would fuse cross-workspace chunks (PR-4).
    index = _bm25_indices.get(workspace)
    if index is None:
        chunks = load_all_chunks(workspace=workspace)
        index = build_bm25_index(chunks)
        _bm25_indices[workspace] = index
        logger.info(f"BM25 index built workspace={workspace} chunks={len(chunks)}")
    return index


def load_eval_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "question" not in item:
            raise ValueError(f"eval sample missing 'question': {item!r}")
    logger.info(f"Loaded {len(data)} evaluation samples")
    return data


def resolve_workspace(item: Dict[str, Any], default_workspace: str) -> str:
    """Per-sample `workspace` field overrides the CLI default."""
    workspace = item.get("workspace", default_workspace)
    if workspace not in WORKSPACES:
        raise ValueError(
            f"eval sample {item.get('question')!r}: "
            f"unknown workspace {workspace!r} (expected one of {WORKSPACES})"
        )
    return workspace


def run_rag(question: str, workspace: str) -> Dict[str, Any]:
    bm25_index = _get_pipeline(workspace)
    cited_answer = None
    for attempt in range(1, GENERATE_ATTEMPTS + 1):
        try:
            cited_answer = generate(
                query=question,
                bm25_index=bm25_index,
                workspace=workspace,
            )
            break
        except AnswerVerificationError as exc:
            logger.warning(
                f"generate() failed verification for {question[:60]!r} "
                f"(attempt {attempt}/{GENERATE_ATTEMPTS}): {exc}"
            )
            if attempt == GENERATE_ATTEMPTS:
                raise
    return {
        "answer": cited_answer.answer,
        "contexts": [source.text for source in cited_answer.sources],
    }


def build_eval_samples(
    eval_dataset: List[Dict[str, Any]], workspace: str
) -> List[Dict[str, Any]]:
    samples = []
    for item in eval_dataset:
        sample_workspace = resolve_workspace(item, workspace)
        rag_output = run_rag(item["question"], workspace=sample_workspace)
        samples.append({
            "question": item["question"],
            "answer": rag_output["answer"],
            "contexts": rag_output["contexts"],
            "ground_truth": item.get("ground_truth", ""),
            "workspace": sample_workspace,
            "class": item.get("class", "cite"),
        })
        logger.info(f"Sample created: {item['question'][:50]}")
    return samples


def resolve_metric_column(columns: Iterable[Any], metric: str) -> str:
    """Map a metric name onto the actual result column: exact match first,
    then a normalised (case/separator-insensitive) match.

    Raises KeyError when no column matches — a missing metric must fail
    loudly, never silently score 0.
    """
    available = [str(column) for column in columns]
    if metric in available:
        return metric

    def normalise(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    matches = [column for column in available if normalise(column) == normalise(metric)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"metric {metric!r} matches multiple columns: {matches}")
    raise KeyError(
        f"metric column {metric!r} not found in eval result; "
        f"available columns: {available}"
    )


def extract_metric_scores(df: pd.DataFrame, metrics: Sequence[str]) -> Dict[str, float]:
    """Explicit per-metric column means (never a blanket numeric mean)."""
    scores: Dict[str, float] = {}
    for metric in metrics:
        column = resolve_metric_column(df.columns, metric)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"metric column {column!r} contains no numeric scores")
        scores[metric] = float(values.mean())
    return scores


def mean_metric_scores(
    frames: Sequence[pd.DataFrame], metrics: Sequence[str]
) -> Dict[str, float]:
    """Count-weighted mean per metric across one or more result frames.

    A metric absent from every frame raises KeyError — ground-truth metrics
    must never fall back to 0 when their column is missing.
    """
    scores: Dict[str, float] = {}
    missing: List[str] = []
    for metric in metrics:
        total = 0.0
        count = 0
        seen = False
        for df in frames:
            try:
                column = resolve_metric_column(df.columns, metric)
            except KeyError:
                continue
            seen = True
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            total += float(values.sum())
            count += int(values.size)
        if not seen or count == 0:
            missing.append(metric)
            continue
        scores[metric] = total / count
    if missing:
        raise KeyError(f"no scores computed for metrics: {missing}")
    return scores


def attach_metric_scores(
    samples: List[Dict[str, Any]], df: pd.DataFrame, metrics: Sequence[str]
) -> None:
    """Write each result row's scores onto the positionally-matching sample.

    Metrics the frame does not carry (e.g. ground-truth metrics for abstain
    samples) are recorded as null.
    """
    if len(df) != len(samples):
        raise ValueError(
            f"eval result has {len(df)} rows but {len(samples)} samples were run"
        )
    for sample, (_, row) in zip(samples, df.iterrows()):
        scores: Dict[str, float | None] = {}
        for metric in metrics:
            try:
                column = resolve_metric_column(df.columns, metric)
            except KeyError:
                scores[metric] = None
                continue
            value = row[column]
            scores[metric] = None if pd.isna(value) else round(float(value), 4)
        sample["scores"] = scores


def evaluate_samples(samples: List[Dict[str, Any]], metrics: Sequence[str]) -> pd.DataFrame:
    """Live ragas call — the only place API traffic happens."""
    rows = [{column: sample.get(column) for column in RAGAS_INPUT_COLUMNS} for sample in samples]
    dataset = Dataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=_build_metrics(metrics))
    return result.to_pandas()


def evaluate_dataset(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Score cite samples on all metrics, abstain samples on non-ground-truth
    metrics only, then combine into threshold-level means."""
    scored = [sample for sample in samples if sample.get("class") != "abstain"]
    abstain = [sample for sample in samples if sample.get("class") == "abstain"]
    frames: List[pd.DataFrame] = []
    if scored:
        df = evaluate_samples(scored, METRIC_NAMES)
        attach_metric_scores(scored, df, METRIC_NAMES)
        frames.append(df)
        logger.info(f"Scored {len(scored)} cite samples on {len(METRIC_NAMES)} metrics")
    if abstain:
        df = evaluate_samples(abstain, ABSTAIN_METRICS)
        attach_metric_scores(abstain, df, METRIC_NAMES)
        frames.append(df)
        logger.info(
            f"Scored {len(abstain)} abstain samples on {len(ABSTAIN_METRICS)} metrics "
            f"(ground-truth metrics skipped)"
        )
    if not frames:
        raise ValueError("evaluation dataset produced no samples to score")
    return mean_metric_scores(frames, METRIC_NAMES)


def check_thresholds(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    results = {}
    for metric, threshold in THRESHOLDS.items():
        if metric not in scores:
            raise KeyError(f"no score computed for threshold metric {metric!r}")
        score = scores[metric]
        results[metric] = {
            "score": round(score, 4),
            "threshold": threshold,
            "status": "PASS" if score >= threshold else "FAIL",
        }
    return results


def gates_failed(threshold_results: Dict[str, Dict[str, Any]]) -> List[str]:
    return [
        metric
        for metric, result in threshold_results.items()
        if result["status"] == "FAIL"
    ]


def per_sample_report(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact per-sample rows for results.json (context texts dropped)."""
    return [
        {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts_count": len(sample["contexts"]),
            "ground_truth": sample["ground_truth"],
            "workspace": sample["workspace"],
            "class": sample["class"],
            "scores": sample.get("scores", {}),
        }
        for sample in samples
    ]


def save_results(
    scores: Dict[str, float],
    threshold_results: Dict[str, Dict[str, Any]],
    per_sample: List[Dict[str, Any]],
    workspace: str,
    output_path: str,
) -> None:
    report = {
        "workspace": workspace,
        "metric_scores": scores,
        "threshold_results": threshold_results,
        "per_sample": per_sample,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def parse_args(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description="Run the RAGAS quality eval (local-only gate).")
    parser.add_argument(
        "--workspace",
        choices=WORKSPACES,
        default="academic",
        help="Default workspace for samples without an explicit workspace field.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    install_voyage_pacing()
    Config.validate()
    logger.info(f"Starting RAG evaluation (default workspace={args.workspace})")
    eval_dataset = load_eval_dataset(EVAL_DATASET_PATH)
    samples = build_eval_samples(eval_dataset, workspace=args.workspace)
    scores = evaluate_dataset(samples)
    threshold_results = check_thresholds(scores)
    save_results(
        scores, threshold_results, per_sample_report(samples), args.workspace, RESULTS_PATH
    )

    print("\n==============================")
    print("RAG Evaluation Results")
    print("==============================")
    for metric, result in threshold_results.items():
        print(
            f"{metric}: {result['score']:.4f} "
            f"(threshold={result['threshold']}) => {result['status']}"
        )
    failed = gates_failed(threshold_results)
    if failed:
        print(f"FAILED gates: {', '.join(failed)}")
        return 1
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
