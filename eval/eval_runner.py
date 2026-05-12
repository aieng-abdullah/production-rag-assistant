import json
from typing import List, Dict, Any
import os 
os.environ["RAGAS_MAX_WORKERS"] = "1"

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.db.chroma_client import load_all_chunks
from src.retrieval.bm25_index import build_bm25_index
from src.generation.chain import generate
from src.config import Config

EVAL_DATASET_PATH = "data/eval_dataset.json"
RESULTS_PATH = "results.json"

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}

llm = LangchainLLMWrapper(ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL,model_kwargs={"device": "cpu"}))

_chunks = None
_bm25_index = None

def _get_pipeline():
    global _chunks, _bm25_index
    if _chunks is None:
        _chunks = load_all_chunks()
        _bm25_index = build_bm25_index(_chunks)
    return _chunks, _bm25_index

def load_eval_dataset(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} evaluation samples")
    return data

def run_rag(question: str) -> Dict[str, Any]:
    chunks, bm25_index = _get_pipeline()
    cited_answer = generate(query=question, chunks=chunks, bm25_index=bm25_index)
    return {
        "answer": cited_answer.answer,
        "contexts": [source.text for source in cited_answer.sources],
    }

def build_eval_samples(eval_dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    samples = []
    for item in eval_dataset:
        rag_output = run_rag(item["question"])
        samples.append({
            "question": item["question"],
            "answer": rag_output["answer"],
            "contexts": rag_output["contexts"],
            "ground_truth": item["ground_truth"],
        })
        logger.info(f"Sample created: {item['question'][:50]}")
    return samples

def run_ragas_evaluation(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    dataset = Dataset.from_list(samples)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )

    df = result.to_pandas()

    scores = (
        df
        .select_dtypes(include=["number"])
        .mean()
        .to_dict()
    )

    return scores

def check_thresholds(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    results = {}
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0)
        results[metric] = {
            "score": round(score, 4),
            "threshold": threshold,
            "status": "PASS" if score >= threshold else "FAIL",
        }
    return results

def save_results(samples, scores, threshold_results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metric_scores": scores, "threshold_results": threshold_results, "samples": samples}, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def main():
    logger.info("Starting RAG evaluation")
    eval_dataset = load_eval_dataset(EVAL_DATASET_PATH)
    samples = build_eval_samples(eval_dataset)
    scores = run_ragas_evaluation(samples)
    threshold_results = check_thresholds(scores)
    save_results(samples, scores, threshold_results, RESULTS_PATH)

    print("\n==============================")
    print("RAG Evaluation Results")
    print("==============================")
    for metric, result in threshold_results.items():
        print(f"{metric}: {result['score']:.4f} (threshold={result['threshold']}) => {result['status']}")

if __name__ == "__main__":
    main()