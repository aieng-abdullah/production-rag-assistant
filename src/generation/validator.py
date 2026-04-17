"""Citation validation - ensure all [SOURCE N] references are valid."""

import re
from typing import List

from src.monitoring.logger import get_logger

logger = get_logger("validator")


def extract_citations(answer: str) -> List[int]:
    """Extract all [SOURCE N] citations from answer."""
    # Find all [SOURCE N] patterns
    pattern = r'\[SOURCE\s+(\d+)\]'
    matches = re.findall(pattern, answer)
    
    # Convert to integers and deduplicate
    citations = sorted(set(int(m) for m in matches))
    
    return citations


def validate_citations(answer: str, max_chunk_index: int) -> tuple:
    """Validate that all citations reference valid chunks."""
    citations = extract_citations(answer)
    
    # Check for invalid indices
    invalid = [c for c in citations if c < 0 or c > max_chunk_index]
    
    if invalid:
        logger.error(f"Invalid citations found: {invalid}, max valid: {max_chunk_index}")
        raise ValueError(f"Invalid citations: {invalid}. Valid range: 0-{max_chunk_index}")
    
    # Check if answer has at least one citation
    if not citations:
        logger.error("No citations found in answer")
        raise ValueError("Answer must contain at least one [SOURCE N] citation")
    
    logger.info(f"Validated {len(citations)} citations")
    return True, [], citations
