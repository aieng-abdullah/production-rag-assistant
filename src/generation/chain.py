"""
file: chain.py
This file do retrive top k chunk and generate citation,call llm and genarate citeted answer

Backward compatibility wrapper around RAGGenerator class.
"""
from time import monotonic
from typing import Any

from loguru import logger
from langchain_groq import ChatGroq
from pydantic import ValidationError

from src.retrieval.pipeline import retrieval
from src.generation.Citation_system import build_citation_prompt
from src.generation.Citation_system import CitedAnswer
from src.generation.Citation_system import Source
from src.config import Config
from src.monitoring.langfuse_tracer import flush_langfuse, get_langfuse_client
from src.core.rag_generator import RAGGenerator
from src.core.retrieval_pipeline import RetrievalPipeline
from src.core.vector_store import VectorStore
from src.core.bm25_retriever import BM25Retriever
from src.core.cross_encoder import CrossEncoderReranker


def _usage_from_lc_response(response: Any) -> dict[str, int] | None:
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage")
    if not usage or not isinstance(usage, dict):
        return None
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage and usage[key] is not None:
            try:
                out[key] = int(usage[key])
            except (TypeError, ValueError):
                pass
    return out or None


def generate(query: str, chunks: list[dict], bm25_index) -> CitedAnswer:
    """
    Generate citation and answer with LLM (backward compatibility wrapper).
    
    Args:
        query: User query string.
        chunks: List of all chunks to search from.
        bm25_index: BM25 index (for backward compatibility).
        
    Returns:
        CitedAnswer with answer text and sources.
    """
    lf = get_langfuse_client()
    t0 = monotonic()

    if lf is None:
        return _generate_untraced(query, chunks, bm25_index, t0)

    from langfuse.langchain import CallbackHandler

    trace_id = lf.create_trace_id()
    trace_context: dict[str, str] = {"trace_id": trace_id}

    try:
        with lf.trace(
            name="rag-generation",
            input={"query": query},
            context=trace_context,
        ) as trace:
            # Step 1: Retrieval
            with trace.span(name="retrieval") as span:
                top_chunks = retrieval(
                    query,
                    chunks,
                    bm25_index,
                    top_k=Config.TOP_K_RERANK,
                    lf_retrieval_parent=span,
                )
                span.update(output={"chunk_count": len(top_chunks)})

            # Step 2: Build prompt
            with trace.span(name="build-prompt") as span:
                citation_prompt = build_citation_prompt(query, top_chunks)
                span.update(output={"prompt_length": len(citation_prompt)})

            # Step 3: LLM invocation
            with trace.span(name="llm-invoke") as span:
                client = ChatGroq(
                    api_key=Config.GROQ_API_KEY,
                    model=Config.GROQ_MODEL,
                )
                response = client.invoke(citation_prompt)
                answer_text = response.content
                usage = _usage_from_lc_response(response)
                span.update(output={"answer_length": len(answer_text), "usage": usage})

            # Step 4: Build sources
            sources = [
                Source(
                    doc_id=chunk["doc_id"],
                    page_num=chunk["page_num"],
                    text=chunk["text"],
                )
                for chunk in top_chunks
            ]

            # Step 5: Validate
            cited_answer = CitedAnswer(answer=answer_text, sources=sources)

            trace.update(
                output={
                    "answer": answer_text,
                    "source_count": len(sources),
                }
            )

            logger.info(
                f"RAG generation completed in {monotonic() - t0:.2f}s, "
                f"sources: {len(sources)}, "
                f"usage: {usage}"
            )

            return cited_answer

    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        raise


def _generate_untraced(
    query: str, chunks: list[dict], bm25_index, t0: float
) -> CitedAnswer:
    try:
        top_chunks = retrieval(query, chunks, bm25_index)
        logger.debug(f"Retrieved {len(top_chunks)} top chunks")
    except Exception as e:
        logger.error(f"Error while retrieving top chunks: {e}")
        raise RuntimeError(f"Failed to retrieve top chunks: {e}")

    try:
        citation_prompt = build_citation_prompt(query, top_chunks)
        logger.debug(f"Generated citation prompt: {citation_prompt}")
    except Exception as e:
        logger.error(f"Error while generating citation prompt: {e}")
        raise RuntimeError(f"Failed to generate citation prompt: {e}")

    client = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
    )
    response = client.invoke(citation_prompt)
    answer_text = response.content

    sources = [
        Source(
            doc_id=chunk["doc_id"],
            page_num=chunk["page_num"],
            text=chunk["text"],
        )
        for chunk in top_chunks
    ]
    _ = (monotonic() - t0) * 1000
    return CitedAnswer(answer=answer_text, sources=sources)
