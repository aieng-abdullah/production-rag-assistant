"""Pydantic models for structured LLM output."""

from typing import List
from pydantic import BaseModel, Field


class CitedAnswer(BaseModel):
    """Structured answer with enforced citations.
    
    Attributes:
        answer: The generated answer text.
        citations: List of source indices cited in the answer.
    """
    answer: str = Field(
        description="The answer text with [SOURCE N] citations embedded"
    )
    citations: List[int] = Field(
        description="List of source indices actually cited in the answer",
        min_length=1
    )
