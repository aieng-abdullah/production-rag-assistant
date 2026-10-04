"""Generate cited answers by combining retrieval, LLM inference, and citation validation."""
import logging
from time import monotonic
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log,
)

from src.retrieval.pipeline import retrieval
from src.generation.Citation_system import build_citation_prompt, CitedAnswer, Source
from src.generation.schema import (
    AnswerVerificationError,
    StructuredAnswer,
    parse_structured,
    structured_to_prose,
    verify_quotes,
)
from src.generation.verifier import (
    VERIFY_RETRY,
    ClaimVerification,
    build_verification,
    judge_claims,
)
from src.generation.profiles import get_prompt_version
from src.generation.providers import (
    Provider,
    ProviderOverrides,
    build_provider_chain,
    create_langchain_client,
)
from src.config import Config
from src.db.chroma_client import DEFAULT_TENANT, count_chunks
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


def _merge_usage(
    first: dict[str, int] | None, second: dict[str, int] | None
) -> dict[str, int] | None:
    """Sum token usage across the original call and its repair re-ask."""
    if not first:
        return second
    if not second:
        return first
    return {key: first.get(key, 0) + second.get(key, 0) for key in set(first) | set(second)}


def _collect_problems(raw: str, chunks: list[dict]) -> list[str]:
    """Parse + quote-verify one model output; [] means clean."""
    try:
        parsed = parse_structured(raw)
    except AnswerVerificationError as exc:
        return [str(exc)]
    return [
        f"claim {problem['claim']} source {problem['source_id']}: {problem['issue']}"
        for problem in verify_quotes(parsed, chunks)
    ]


def _generate_structured(
    prompt: str,
    chunks: list[dict],
    provider_overrides: ProviderOverrides | None = None,
    callbacks: list | None = None,
) -> tuple[StructuredAnswer, dict[str, int] | None]:
    """LLM → parse → deterministic quote verify, with ONE repair re-ask.

    The repair re-send appends the exact verification failures; a second
    failure raises `AnswerVerificationError` (never ships unverified prose).
    """
    raw, usage = _invoke_llm(
        prompt, callbacks=callbacks, provider_overrides=provider_overrides
    )
    problems = _collect_problems(raw, chunks)
    if not problems:
        return parse_structured(raw), usage

    logger.warning(f"Structured output needs repair: {problems}")
    repair_prompt = (
        f"{prompt}\n\nYour previous answer failed verification:\n"
        + "\n".join(f"- {problem}" for problem in problems)
        + "\nReturn the corrected JSON object only."
    )
    raw, usage2 = _invoke_llm(
        repair_prompt, callbacks=callbacks, provider_overrides=provider_overrides
    )
    problems = _collect_problems(raw, chunks)
    if problems:
        raise AnswerVerificationError(f"unrepairable after 1 attempt: {problems}")
    return parse_structured(raw), _merge_usage(usage, usage2)


def _needs_repair(verifications: list[ClaimVerification]) -> bool:
    """Retry generator for real rejections; a judge outage is not fixable
    by regenerating, so `judge unavailable` verdicts skip the loop."""
    return any(
        v.verdict != "SUPPORTED" and not v.reason.startswith("judge unavailable")
        for v in verifications
    )


def _generate_verified(
    prompt: str,
    chunks: list[dict],
    provider_overrides: ProviderOverrides | None = None,
    callbacks: list | None = None,
) -> tuple[StructuredAnswer, list[ClaimVerification], dict[str, int] | None]:
    """Generate → quote-verify → judge entailment, with bounded repair.

    Judge rejections are fed back to the generator at most `VERIFY_RETRY`
    times; persistent rejections ship as an `unverified`/`partial` badge —
    never silently dropped, never a hard failure (PLAN PR-4b-ii).
    """
    structured, usage = _generate_structured(
        prompt, chunks, provider_overrides=provider_overrides, callbacks=callbacks
    )
    verifications = judge_claims(structured, chunks, callbacks=callbacks)

    attempts = 0
    while attempts < VERIFY_RETRY and _needs_repair(verifications):
        feedback = "\n".join(
            f"- claim {v.claim_index} [{v.verdict}]: {v.reason}"
            for v in verifications
            if v.verdict != "SUPPORTED"
            and not v.reason.startswith("judge unavailable")
        )
        repair_prompt = (
            f"{prompt}\n\nYour previous claims were rejected by the verifier:\n"
            f"{feedback}\nFix the claims to match the evidence, or abstain "
            "entirely if they cannot be supported. Return the corrected JSON "
            "object only."
        )
        logger.warning(f"Judge rejected claims, re-generating (attempt {attempts + 1}): {feedback}")
        structured, round_usage = _generate_structured(
            repair_prompt, chunks, provider_overrides=provider_overrides, callbacks=callbacks
        )
        usage = _merge_usage(usage, round_usage)
        verifications = judge_claims(structured, chunks, callbacks=callbacks)
        attempts += 1

    return structured, verifications, usage


def _trace_payload(
    workspace: str,
    chunks: list[dict],
    structured: StructuredAnswer,
    verifications: list[ClaimVerification],
    usage: dict[str, int] | None,
) -> dict:
    """Provenance record persisted with the answer (PLAN PR-4b-iii)."""
    cited_ids = {
        citation.source_id for claim in structured.claims for citation in claim.citations
    }
    return {
        "workspace": workspace,
        "prompt_version": get_prompt_version(workspace),
        "model": Config.GROQ_MODEL,
        "verify_model": Config.VERIFY_MODEL,
        "token_usage": usage,
        "chunks": [
            {
                "source_id": index + 1,
                "doc_id": chunk.get("doc_id"),
                "page_num": chunk.get("page_num"),
                "cited": index + 1 in cited_ids,
            }
            for index, chunk in enumerate(chunks)
        ],
        "claims": [claim.model_dump() for claim in structured.claims],
        "abstained": structured.abstained,
        "abstain_reason": structured.abstain_reason,
        "verification": build_verification(structured, verifications),
    }


def _invoke_llm(
    prompt: str,
    callbacks: list | None = None,
    provider_overrides: ProviderOverrides | None = None,
) -> tuple[str, dict[str, int] | None]:
    """Call LLM providers with retry + exponential backoff + failover.

    Tries each provider in order (Groq → Anthropic → OpenAI).
    Each provider gets up to 3 attempts with exponential backoff (1s, 2s, 4s).
    On failure, logs the error and moves to the next provider.
    """
    chain = build_provider_chain(provider_overrides)
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


def _run_pipeline(
    query: str,
    bm25_index,
    provider_overrides: ProviderOverrides | None = None,
    tenant_id: str = DEFAULT_TENANT,
    workspace: str = Config.DEFAULT_WORKSPACE,
) -> CitedAnswer:
    """Core RAG pipeline: retrieve → build prompt → call LLM → validate citations."""
    top_chunks = retrieval(query, bm25_index, tenant_id=tenant_id, workspace=workspace)
    logger.debug(f"Retrieved {len(top_chunks)} top chunks")

    citation_prompt = build_citation_prompt(query, top_chunks, workspace=workspace)
    logger.debug(f"Generated citation prompt: {citation_prompt}")

    structured, verifications, _usage = _generate_verified(
        citation_prompt, top_chunks, provider_overrides=provider_overrides
    )
    answer_text = structured_to_prose(structured)

    sources = _build_sources(top_chunks)
    return CitedAnswer(
        answer=answer_text,
        sources=sources,
        verification=build_verification(structured, verifications),
        trace=_trace_payload(workspace, top_chunks, structured, verifications, _usage),
    )


def _generate_traced(
    query: str,
    bm25_index,
    lf,
    provider_overrides: ProviderOverrides | None = None,
    tenant_id: str = DEFAULT_TENANT,
    workspace: str = Config.DEFAULT_WORKSPACE,
) -> CitedAnswer:
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
                "corpus_chunk_count": count_chunks(
                    tenant_id=tenant_id, workspace=workspace
                ),
                "tenant_id": tenant_id,
                "workspace": workspace,
            },
            metadata={"groq_model": Config.GROQ_MODEL},
        ) as root:
            # --- Retrieval ---
            with root.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input={
                    "query": query,
                    "corpus_chunk_count": count_chunks(
                        tenant_id=tenant_id, workspace=workspace
                    ),
                },
            ) as retr:
                top_chunks = retrieval(
                    query,
                    bm25_index,
                    lf_retrieval_parent=retr,
                    tenant_id=tenant_id,
                    workspace=workspace,
                )
                logger.debug(f"Retrieved {len(top_chunks)} top chunks")
                retr.update(output={"chunks_retrieved": len(top_chunks)})

            # --- Prompt build ---
            with root.start_as_current_observation(
                name="prompt-build",
                as_type="span",
            ) as pb:
                citation_prompt = build_citation_prompt(
                    query, top_chunks, workspace=workspace
                )
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
                try:
                    structured, verifications, usage = _generate_verified(
                        citation_prompt,
                        top_chunks,
                        callbacks=[handler],
                        provider_overrides=provider_overrides,
                    )
                except AnswerVerificationError as exc:
                    llm_span.update(level="ERROR", status_message=str(exc))
                    raise
                answer_text = structured_to_prose(structured)
                llm_span.update(
                    output={
                        "claims": len(structured.claims),
                        "abstained": structured.abstained,
                        "verifications": [v.model_dump() for v in verifications],
                        "answer_chars": len(answer_text),
                    },
                    metadata={"token_usage": usage} if usage else None,
                )

            # --- Citation verification (deterministic, traced) ---
            sources = _build_sources(top_chunks)
            problems = verify_quotes(structured, top_chunks)
            verification = build_verification(structured, verifications)

            with root.start_as_current_observation(
                name="citation-validation",
                as_type="evaluator",
                input={"answer_preview": (answer_text or "")[:300]},
            ) as val_span:
                val_span.update(
                    output={
                        "quote_verified": not problems,
                        "verification_status": verification["status"],
                        "claims": len(structured.claims),
                        "abstained": structured.abstained,
                        "sources_count": len(sources),
                    }
                )
            cited = CitedAnswer(
                answer=answer_text,
                sources=sources,
                verification=verification,
                trace=_trace_payload(
                    workspace, top_chunks, structured, verifications, usage
                ),
            )

            total_ms = (monotonic() - t0) * 1000
            root.update(
                output={
                    "answer": cited.answer,
                    "sources_count": len(cited.sources),
                    "claims": len(structured.claims),
                    "verification_status": verification["status"],
                    "quote_verified": True,
                    "total_latency_ms": total_ms,
                }
            )
            return cited
    finally:
        flush_langfuse()


def generate(
    query: str,
    bm25_index,
    provider_overrides: ProviderOverrides | None = None,
    tenant_id: str = DEFAULT_TENANT,
    workspace: str = Config.DEFAULT_WORKSPACE,
) -> CitedAnswer:
    """Generate a cited answer for the given query using the RAG pipeline."""
    lf = get_langfuse_client()
    if lf is None:
        return _run_pipeline(
            query, bm25_index, provider_overrides, tenant_id, workspace=workspace
        )
    try:
        return _generate_traced(
            query, bm25_index, lf, provider_overrides, tenant_id, workspace=workspace
        )
    except ImportError as exc:
        # Tracing is non-critical: a broken/partial Langfuse integration must
        # never block core queries (AGENTS graceful-degradation contract).
        logger.warning(
            "Langfuse langchain integration unavailable ({}); "
            "falling back to untraced pipeline",
            exc,
        )
        return _run_pipeline(
            query, bm25_index, provider_overrides, tenant_id, workspace=workspace
        )
