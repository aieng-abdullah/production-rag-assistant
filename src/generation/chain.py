"""LLM generation chain using LangChain ChatGroq.

Provides a RAG generation pipeline with citation enforcement and optional validation.
"""

from typing import List, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from src.config import Config
from src.generation.prompts import RAG_PROMPT, CITATION_VALIDATION_PROMPT, format_context


# Lazy-loaded LLM instance
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    """Get or initialize the Groq LLM.

    Returns:
        ChatGroq instance configured with model from Config.
    """
    global _llm
    if _llm is None:
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment")

        _llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info(f"Initialized Groq LLM: {Config.GROQ_MODEL}")

    return _llm


def create_rag_chain():
    """Create the RAG generation chain.

    Chain pipeline:
    1. Format context from retrieved chunks
    2. Build prompt with context + question
    3. Call LLM
    4. Parse string output

    Returns:
        Runnable chain that takes {"context": chunks, "question": str} and returns answer str.
    """
    llm = _get_llm()

    chain = (
        {
            "context": RunnableLambda(lambda x: format_context(x["context"])),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def generate_answer(question: str, context_chunks: List[Dict]) -> Dict:
    """Generate an answer with citations from retrieved context.

    Args:
        question: The user's question.
        context_chunks: Retrieved document chunks with text and metadata.

    Returns:
        Dictionary with:
        - answer: The generated answer string
        - citations: List of citation dicts with source numbers and metadata
    """
    logger.info(f"Generating answer for: '{question[:50]}...'")
    logger.debug(f"Context: {len(context_chunks)} chunks")

    try:
        chain = create_rag_chain()

        # Run the chain
        answer = chain.invoke({
            "context": context_chunks,
            "question": question,
        })

        # Extract citations from answer
        citations = _extract_citations(answer, context_chunks)

        logger.info(f"Generated answer with {len(citations)} citations")

        return {
            "answer": answer,
            "citations": citations,
            "context_used": len(context_chunks),
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise RuntimeError(f"Answer generation failed: {e}")


def _extract_citations(answer: str, chunks: List[Dict]) -> List[Dict]:
    """Extract and resolve citations from answer text.

    Args:
        answer: Generated answer text with [SOURCE N] citations.
        chunks: Original context chunks used for generation.

    Returns:
        List of citation dictionaries with source number and chunk metadata.
    """
    import re

    # Find all [SOURCE N] patterns
    pattern = r'\[SOURCE\s+(\d+)\]'
    matches = re.findall(pattern, answer)

    citations = []
    seen = set()

    for source_num_str in matches:
        source_num = int(source_num_str)

        # Skip duplicates
        if source_num in seen:
            continue
        seen.add(source_num)

        # Get chunk metadata (1-indexed in text, 0-indexed in list)
        chunk_idx = source_num - 1
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            citations.append({
                "source_number": source_num,
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "chunk_index": chunk.get("chunk_index", -1),
                "text_preview": chunk.get("text", "")[:100] + "...",
            })

    return citations


def validate_answer(answer: str, context_chunks: List[Dict]) -> Dict:
    """Validate that answer citations are grounded in context.

    Args:
        answer: The generated answer to validate.
        context_chunks: Original context chunks.

    Returns:
        Validation result with is_valid boolean and explanation.
    """
    logger.debug("Validating answer citations")

    try:
        llm = _get_llm()
        context_str = format_context(context_chunks)

        messages = CITATION_VALIDATION_PROMPT.format_messages(
            context=context_str,
            answer=answer,
        )

        response = llm.invoke(messages)
        result_text = response.content.strip()

        is_valid = result_text.startswith("VALID")

        return {
            "is_valid": is_valid,
            "explanation": result_text if not is_valid else "All citations verified",
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "is_valid": False,
            "explanation": f"Validation error: {e}",
        }
