"""
Citation_system.py

This file contains the citation system for the RAG system.
"""
from pydantic import BaseModel, field_validator




class Source(BaseModel):
    doc_id: str
    page_num: int
    text: str

class CitedAnswer(BaseModel):
    answer: str
    sources: list[Source]
    
    @field_validator("answer")  # ← indented inside class
    @classmethod
    def must_have_citation(cls, validate):
        if "[SOURCE" not in validate:
            raise ValueError("Answer must contain at least one [SOURCE N] citation")
        return validate
    
    
    
def build_citation_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the citation prompt for the RAG system.
    
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(f"[SOURCE {i}] {chunk['text']}")
    SYSTEM_PROMPT = """You are a research assistant. 
    CRITICAL: Cite every fact with [SOURCE N] format.
    Example: The Earth is round [SOURCE 1]."""
    
    sources_text = "\n".join(formatted)
    
    return f"""{SYSTEM_PROMPT}

    Sources:
    {sources_text}

    Question: {query}

    Answer (must cite sources):"""



