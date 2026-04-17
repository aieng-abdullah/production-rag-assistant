"""Live RAGAS evaluation for single queries."""

from typing import Dict, List
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset

from src.monitoring.logger import get_logger

logger = get_logger("live_ragas")


class LiveRagasEvaluator:
    """Evaluate single RAG queries with RAGAS metrics."""
    
    def __init__(self):
        """Initialize evaluator with metrics."""
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    
    def evaluate_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """Evaluate a single RAG query.
        
        Args:
            question: User question.
            answer: Generated answer.
            contexts: Retrieved context chunks.
            ground_truth: Optional ground truth for comparison.
            
        Returns:
            Dict with RAGAS scores.
        """
        # Build evaluation data
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        try:
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=self.metrics,
                raise_exceptions=False
            )
            
            # Convert to dict
            scores = {k: round(float(v), 3) for k, v in results.items()}
            
            logger.info(f"RAGAS scores: {scores}")
            
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "error": str(e)
            }
    
    def check_quality(self, scores: Dict[str, float], thresholds: Dict[str, float] = None) -> bool:
        """Check if scores meet quality thresholds.
        
        Args:
            scores: RAGAS scores dict.
            thresholds: Min acceptable scores (default: faithfulness > 0.7).
            
        Returns:
            bool: True if quality passes.
        """
        if thresholds is None:
            thresholds = {
                "faithfulness": 0.70,
                "answer_relevancy": 0.70
            }
        
        passed = True
        for metric, threshold in thresholds.items():
            score = scores.get(metric, 0)
            if score < threshold:
                logger.warning(f"Quality check failed: {metric}={score:.3f} < {threshold}")
                passed = False
        
        if passed:
            logger.info("Quality check passed")
        
        return passed


# Global evaluator instance
live_evaluator = LiveRagasEvaluator()
