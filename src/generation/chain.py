"""
file: chain.py
This file do retrive top k chunk and generate citation,call llm and genarate citeted answer
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
from src.db.chroma_client import count_chunks
from src.monitoring.langfuse_tracer import flush_langfuse, get_langfuse_client


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


def generate(query: str, bm25_index) -> CitedAnswer:
    """
    genarate citation and answer with llm
    """
    lf = get_langfuse_client()
    t0 = monotonic()

    if lf is None:
        return _generate_untraced(query, bm25_index)

    from langfuse.langchain import CallbackHandler

    trace_id = lf.create_trace_id()
    trace_context: dict[str, str] = {"trace_id": trace_id}

    try:
        with lf.start_as_current_observation(
            name="rag-generate",
            as_type="chain",
            trace_context=trace_context,
            input={
                "query": query,
                "top_k": Config.TOP_K_RERANK,
                "corpus_chunk_count": count_chunks(),
            },
            metadata={"groq_model": Config.GROQ_MODEL},
        ) as root:
            with root.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input={"query": query, "corpus_chunk_count": count_chunks()},
            ) as retr:
                try:
                    top_chunks = retrieval(
                        query, bm25_index, lf_retrieval_parent=retr
                    )
                    logger.debug(f"Retrieved {len(top_chunks)} top chunks")
                    retr.update(output={"chunks_retrieved": len(top_chunks)})
                except Exception as e:
                    logger.error(f"Error while retrieving top chunks: {e}")
                    raise RuntimeError(f"Failed to retrieve top chunks: {e}")

            with root.start_as_current_observation(
                name="prompt-build",
                as_type="span",
            ) as pb:
                try:
                    citation_prompt = build_citation_prompt(query, top_chunks)
                    logger.debug(f"Generated citation prompt: {citation_prompt}")
                    pb.update(output={"prompt_chars": len(citation_prompt)})
                except Exception as e:
                    logger.error(f"Error while generating citation prompt: {e}")
                    raise RuntimeError(f"Failed to generate citation prompt: {e}")

            client = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
            )
            handler = CallbackHandler(
                public_key=Config.LANGFUSE_PUBLIC_KEY,
                trace_context=trace_context,
            )

            with root.start_as_current_observation(
                name="llm-call",
                as_type="span",
                metadata={"model": Config.GROQ_MODEL},
            ) as llm_span:
                response = client.invoke(
                    citation_prompt,
                    config={"callbacks": [handler]},
                )
                answer_text = response.content
                usage = _usage_from_lc_response(response)
                llm_span.update(
                    output={
                        "answer_chars": len(answer_text) if answer_text else 0,
                    },
                    metadata={"token_usage": usage} if usage else None,
                )

            sources = [
                Source(
                    doc_id=chunk["doc_id"],
                    page_num=chunk["page_num"],
                    text=chunk["text"],
                )
                for chunk in top_chunks
            ]

            with root.start_as_current_observation(
                name="citation-validation",
                as_type="evaluator",
                input={"answer_preview": (answer_text or "")[:300]},
            ) as val_span:
                try:
                    cited = CitedAnswer(answer=answer_text, sources=sources)
                    val_span.update(
                        output={
                            "citation_valid": True,
                            "sources_count": len(sources),
                        }
                    )
                except ValidationError as e:
                    val_span.update(
                        level="ERROR",
                        status_message=str(e),
                        output={"citation_valid": False},
                    )
                    raise

            total_ms = (monotonic() - t0) * 1000
            root.update(
                output={
                    "answer": cited.answer,
                    "sources_count": len(cited.sources),
                    "citation_valid": True,
                    "total_latency_ms": total_ms,
                }
            )
            return cited
    finally:
        flush_langfuse()


def _generate_untraced(
    query: str, bm25_index
) -> CitedAnswer:
    try:
        top_chunks = retrieval(query, bm25_index)
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
    return CitedAnswer(answer=answer_text, sources=sources)
