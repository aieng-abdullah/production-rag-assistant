"""Generate cited answers by combining retrieval, LLM inference, and citation validation."""
import logging
from time import monotonic
from typing import Any

from loguru import logger
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log,
)

from src.retrieval.pipeline import retrieval
from src.generation.Citation_system import build_citation_prompt, CitedAnswer, Source
from src.generation.providers import build_provider_chain, create_langchain_client, Provider
from src.config import Config
from src.db.chroma_client import count_chunks
from src.monitoring.langfuse_tracer import flush_langfuse, get_langfuse_client


def _usage_from_lc_response(response: Any) -> dict[str, int] | None:
    """Extract token usage from a LangChain response object."""
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


def _build_sources(chunks: list[dict]) -> list[Source]:
    """Convert raw retrieval chunks into Source objects."""
    return [
        Source(doc_id=c["doc_id"], page_num=c["page_num"], text=c["text"])
        for c in chunks
    ]


def _invoke_llm(
    prompt: str, callbacks: list | None = None
) -> tuple[str, dict[str, int] | None]:
    """Call LLM providers with retry + exponential backoff + failover.

    Tries each provider in order (Groq → Anthropic → OpenAI).
    Each provider gets up to 3 attempts with exponential backoff (1s, 2s, 4s).
    On failure, logs the error and moves to the next provider.
    """
    chain = build_provider_chain()
    config = {"callbacks": callbacks} if callbacks else {}
    last_error: Exception | None = None

    for provider in chain:
        try:
            logger.info(f"Trying provider: {provider.name} ({provider.model})")
            response = _call_provider_with_retry(provider, prompt, config)
            logger.info(f"Provider {provider.name} succeeded")
            return response.content, _usage_from_lc_response(response)
        except Exception as e:
            logger.warning(f"Provider {provider.name} failed after retries: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def _call_provider_with_retry(
    provider: Provider, prompt: str, config: dict
):
    """Call a single provider with retry + exponential backoff."""
    client = create_langchain_client(provider)
    return client.invoke(prompt, config=config)


def _run_pipeline(query: str, bm25_index) -> CitedAnswer:
    """Core RAG pipeline: retrieve → build prompt → call LLM → validate citations."""
    top_chunks = retrieval(query, bm25_index)
    logger.debug(f"Retrieved {len(top_chunks)} top chunks")

    citation_prompt = build_citation_prompt(query, top_chunks)
    logger.debug(f"Generated citation prompt: {citation_prompt}")

    answer_text, _ = _invoke_llm(citation_prompt)

    sources = _build_sources(top_chunks)
    return CitedAnswer(answer=answer_text, sources=sources)


def _generate_traced(query: str, bm25_index, lf) -> CitedAnswer:
    """Run the RAG pipeline with Langfuse tracing spans around each step."""
    from langfuse.langchain import CallbackHandler

    trace_id = lf.create_trace_id()
    trace_context: dict[str, str] = {"trace_id": trace_id}

    t0 = monotonic()

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
            # --- Retrieval ---
            with root.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input={"query": query, "corpus_chunk_count": count_chunks()},
            ) as retr:
                top_chunks = retrieval(query, bm25_index, lf_retrieval_parent=retr)
                logger.debug(f"Retrieved {len(top_chunks)} top chunks")
                retr.update(output={"chunks_retrieved": len(top_chunks)})

            # --- Prompt build ---
            with root.start_as_current_observation(
                name="prompt-build",
                as_type="span",
            ) as pb:
                citation_prompt = build_citation_prompt(query, top_chunks)
                logger.debug(f"Generated citation prompt: {citation_prompt}")
                pb.update(output={"prompt_chars": len(citation_prompt)})

            # --- LLM call ---
            handler = CallbackHandler(
                public_key=Config.LANGFUSE_PUBLIC_KEY,
                trace_context=trace_context,
            )

            with root.start_as_current_observation(
                name="llm-call",
                as_type="span",
                metadata={"model": Config.GROQ_MODEL},
            ) as llm_span:
                answer_text, usage = _invoke_llm(citation_prompt, callbacks=[handler])
                llm_span.update(
                    output={"answer_chars": len(answer_text) if answer_text else 0},
                    metadata={"token_usage": usage} if usage else None,
                )

            # --- Citation validation ---
            sources = _build_sources(top_chunks)

            with root.start_as_current_observation(
                name="citation-validation",
                as_type="evaluator",
                input={"answer_preview": (answer_text or "")[:300]},
            ) as val_span:
                try:
                    cited = CitedAnswer(answer=answer_text, sources=sources)
                    val_span.update(
                        output={"citation_valid": True, "sources_count": len(sources)}
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


def generate(query: str, bm25_index) -> CitedAnswer:
    """Generate a cited answer for the given query using the RAG pipeline."""
    lf = get_langfuse_client()
    if lf is None:
        return _run_pipeline(query, bm25_index)
    return _generate_traced(query, bm25_index, lf)
