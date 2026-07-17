"""
Citation_system.py

This file contains the citation system for the RAG system.
"""
import re
from pydantic import BaseModel, field_validator




class Source(BaseModel):
    doc_id: str
    page_num: int
    text: str

class CitedAnswer(BaseModel):
    answer: str
    sources: list[Source]

    @field_validator("answer")
    @classmethod
    def must_have_citation(cls, validate):
        if not re.search(r'(?:\[SOURCE|SOURCE\s+\d+)', validate):
            raise ValueError("Answer must contain at least one [SOURCE N] citation")

        cleaned = validate.strip()
        # Strip chain-of-thought echo artifacts
        cleaned = re.sub(r'Step\s+\d+:.*', '', cleaned, flags=re.MULTILINE)
        # Strip common LLM preamble/list formatting
        cleaned = re.sub(r'Relevant sources?:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[-*]\s*', '', cleaned, flags=re.MULTILINE)
        # Strip lines that are clearly preamble (no citation and before first cited line)
        lines = cleaned.split('\n')
        content_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r'\[SOURCE\s+\d+', stripped):
                content_lines.append(stripped)
            elif not content_lines:
                continue  # skip preamble before first citation
            else:
                content_lines.append(stripped)
        cleaned = ' '.join(content_lines)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 15:
                continue
            # Match [SOURCE N] or SOURCE N (with or without brackets)
            if not re.search(r'(?:\[SOURCE\s+\d+|SOURCE\s+\d+)', sentence):
                # Allow meta-commentary sentences that don't contain factual claims
                meta_patterns = [
                    r'^(However|Additionally|Furthermore|Moreover|In summary|Note that)',
                    r'^(the question|this|it) (asks|is|refers)',
                    r'^(I|we) (cannot|could not|do not)',
                ]
                if any(re.match(p, sentence, re.IGNORECASE) for p in meta_patterns):
                    continue
                raise ValueError(
                    f"Every sentence must cite a source. Missing citation in: '{sentence}'"
                )
        return validate



def build_citation_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the citation prompt for the RAG system.
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(f"[SOURCE {i}] {chunk['text']}")
    SYSTEM_PROMPT = """You are a research assistant. Follow these rules strictly:

1. ONLY use information from the provided sources. Do NOT add external knowledge.
2. If the sources do not contain enough information to answer the question, say: "I don't have enough information to answer this question based on the provided sources."
3. Cite EVERY factual claim individually with [SOURCE N] format.
4. Before answering, first identify which sources are relevant to the question.
5. Then construct your answer, citing each fact with [SOURCE N].

Example: The Transformer model uses self-attention [SOURCE 1]. It was trained on WMT 2014 data [SOURCE 2]."""

    sources_text = "\n\n".join(formatted)

    return f"""{SYSTEM_PROMPT}

Sources:
{sources_text}

Question: {query}

Answer:"""
