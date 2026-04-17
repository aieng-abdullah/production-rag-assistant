"""Evaluation thresholds and assertions."""

from typing import Dict

from src.config import Config
from src.monitoring.logger import get_logger

logger = get_logger("eval")


def assert_thresholds(scores: Dict[str, float]) -> bool:
    """Assert that all scores meet minimum thresholds."""
    failed = []
    
    # Check faithfulness
    faith = scores.get("faithfulness", 0)
    if faith < Config.FAITHFULNESS_THRESHOLD:
        failed.append(f"faithfulness: {faith:.3f} < {Config.FAITHFULNESS_THRESHOLD}")
    
    # Check answer relevancy
    rel = scores.get("answer_relevancy", 0)
    if rel < Config.ANSWER_RELEVANCY_THRESHOLD:
        failed.append(f"answer_relevancy: {rel:.3f} < {Config.ANSWER_RELEVANCY_THRESHOLD}")
    
    if failed:
        msg = "Threshold assertion failed: " + ", ".join(failed)
        logger.error(msg)
        raise AssertionError(msg)
    
    logger.info("All thresholds passed")
    return True
