"""Citation regression eval (PLAN PR-4b-v).

Deterministic gates — no LLM judge: quote grounding, citation
precision/recall, and abstention accuracy checked against
`data/verify_eval.json`. Run before/after prompt or verifier changes.

Requires GROQ key + ingested demo corpora (live model calls).

    python3 eval/verify_eval.py      # exit 1 when any gate fails
"""

import json
import re
import sys
from pathlib import Path

# Run as a script (`python3 eval/verify_eval.py`), the repo root is not on
# sys.path — and a stale PYTHONPATH could shadow `src` with another project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import Config
from src.db.chroma_client import load_all_chunks
from src.generation.chain import generate
from src.retrieval.bm25_index import build_bm25_index

DATASET_PATH = "data/verify_eval.json"
RESULTS_PATH = "verify_results.json"

THRESHOLDS = {
    "citation_precision": 1.0,
    "citation_recall": 0.75,
    "abstention_accuracy": 1.0,
}


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"id", "class", "workspace", "question", "expect"}
    for item in data:
        missing = required - set(item)
        if missing:
            raise ValueError(f"dataset item {item.get('id')!r} missing {sorted(missing)}")
        if item["expect"] not in ("cite", "abstain"):
            raise ValueError(f"dataset item {item['id']!r}: expect must be cite|abstain")
    logger.info(f"Loaded {len(data)} regression samples")
    return data


def _claims(cited) -> list[dict]:
    return (cited.trace or {}).get("claims", [])


def citation_precision(cited) -> tuple[int, int]:
    """(valid, total) claim citations: source_id resolves AND the verbatim
    quote appears in that source's text."""
    sources = {source.source_id: source for source in cited.sources}
    valid = total = 0
    for claim in _claims(cited):
        for citation in claim.get("citations", []):
            total += 1
            source = sources.get(citation.get("source_id"))
            quote = normalize(citation.get("quote"))
            if source is not None and quote and quote in normalize(source.text):
                valid += 1
    return valid, total


def citation_recall(cited, expected_quote: str, expected_doc_id: str) -> bool:
    """A claim-cited source from the expected doc contains the expected quote."""
    cited_ids = {
        citation.get("source_id")
        for claim in _claims(cited)
        for citation in claim.get("citations", [])
    }
    wanted = normalize(expected_quote)
    return any(
        source.source_id in cited_ids
        and source.doc_id == expected_doc_id
        and wanted in normalize(source.text)
        for source in cited.sources
    )


def abstention_correct(cited, expect: str) -> bool:
    status = (cited.verification or {}).get("status")
    if expect == "abstain":
        return status == "abstained"
    return status is not None and status != "abstained"


def compute_metrics(results: list[dict]) -> dict:
    """Aggregate over per-item outcomes (see `evaluate_item`)."""
    cite_items = [item for item in results if item["expect"] == "cite"]
    precision_valid = sum(item["precision_valid"] for item in results)
    precision_total = sum(item["precision_total"] for item in results)
    recall_hits = sum(item["recall_hit"] for item in cite_items)
    abstain_hits = sum(item["abstention_correct"] for item in results)
    return {
        "citation_precision": (
            round(precision_valid / precision_total, 4) if precision_total else 1.0
        ),
        "citation_recall": (
            round(recall_hits / len(cite_items), 4) if cite_items else 1.0
        ),
        "abstention_accuracy": round(abstain_hits / len(results), 4) if results else 0.0,
    }


def check_gates(metrics: dict) -> dict[str, dict]:
    return {
        name: {
            "score": metrics.get(name, 0.0),
            "threshold": threshold,
            "status": "PASS" if metrics.get(name, 0.0) >= threshold else "FAIL",
        }
        for name, threshold in THRESHOLDS.items()
    }


def evaluate_item(item: dict, cited) -> dict:
    precision_valid, precision_total = citation_precision(cited)
    return {
        "id": item["id"],
        "class": item["class"],
        "expect": item["expect"],
        "status": (cited.verification or {}).get("status"),
        "precision_valid": precision_valid,
        "precision_total": precision_total,
        "recall_hit": citation_recall(
            cited, item.get("expected_quote", ""), item.get("expected_doc_id", "")
        ),
        "abstention_correct": abstention_correct(cited, item["expect"]),
    }


def run(dataset: list[dict]) -> dict:
    indices: dict = {}
    results: list[dict] = []
    for item in dataset:
        workspace = item["workspace"]
        if workspace not in indices:
            chunks = load_all_chunks(workspace=workspace)
            indices[workspace] = build_bm25_index(chunks)
            logger.info(f"Index built workspace={workspace} chunks={len(chunks)}")
        cited = generate(
            query=item["question"],
            bm25_index=indices[workspace],
            workspace=workspace,
        )
        outcome = evaluate_item(item, cited)
        results.append(outcome)
        logger.info(
            f"{outcome['id']}: status={outcome['status']} "
            f"recall_hit={outcome['recall_hit']} "
            f"abstain_ok={outcome['abstention_correct']}"
        )
    metrics = compute_metrics(results)
    return {"metrics": metrics, "gates": check_gates(metrics), "items": results}


def main() -> int:
    Config.validate()
    dataset = load_dataset()
    report = run(dataset)

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n==============================")
    print("Citation Regression Eval (PR-4b-v)")
    print("==============================")
    for name, gate in report["gates"].items():
        print(
            f"{name}: {gate['score']:.4f} (threshold={gate['threshold']}) "
            f"=> {gate['status']}"
        )
    failed = [name for name, gate in report["gates"].items() if gate["status"] == "FAIL"]
    if failed:
        print(f"FAILED gates: {', '.join(failed)}")
        return 1
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
