"""Evaluation runner - executes full pipeline on test dataset."""

import json
import asyncio
from pathlib import Path
from typing import Dict, List

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from src.retrieval.pipeline import hybrid_search
from src.generation.chain import generate
from eval.thresholds import assert_thresholds
from src.monitoring.logger import get_logger

logger = get_logger("eval_runner")


def load_dataset(path: str = "eval/dataset.json") -> List[Dict]:
    """Load evaluation dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    
    return data.get("samples", [])


async def run_single(sample: Dict) -> Dict:
    """Run full pipeline on single sample."""
    question = sample["question"]
    
    # Retrieve
    chunks, _ = hybrid_search(question, top_k=5)
    
    # Generate
    result = generate(question, chunks)
    
    return {
        "question": question,
        "ground_truth": sample["ground_truth"],
        "answer": result["answer"],
        "contexts": [c["text"] for c in chunks]
    }


async def run_evaluation(dataset_path: str = "eval/dataset.json") -> Dict:
    """Run full evaluation on dataset."""
    logger.info("Starting evaluation run")
    
    # Load dataset
    samples = load_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} evaluation samples")
    
    # Run pipeline on all samples
    results = []
    for sample in samples:
        result = await run_single(sample)
        results.append(result)
    
    # Convert to HuggingFace Dataset format for Ragas
    eval_data = Dataset.from_list(results)
    
    # Run Ragas evaluation
    scores = evaluate(
        eval_data,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    
    # Convert to dict
    score_dict = {k: float(v) for k, v in scores.items()}
    
    # Assert thresholds
    try:
        assert_thresholds(score_dict)
        score_dict["passed"] = True
    except AssertionError:
        score_dict["passed"] = False
    
    logger.info(f"Evaluation complete: {score_dict}")
    return score_dict


def save_results(scores: Dict, output_path: str = "eval/results.json"):
    """Save evaluation results to JSON."""
    import datetime
    
    output = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "scores": scores,
        "config": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "top_k": 5
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


async def main():
    """Main entry point for evaluation."""
    scores = await run_evaluation()
    save_results(scores)
    
    # Exit with error if failed
    if not scores.get("passed"):
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
