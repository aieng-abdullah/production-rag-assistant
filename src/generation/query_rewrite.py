"""Standalone-query rewrite for multi-turn chat.

A follow-up like "what about the second one?" retrieves nothing useful on
its own: BM25 and vector search see pronouns, not nouns. `rewrite_query`
spends ONE cheap LLM call — reusing the `chain._invoke_llm` provider
failover machinery — to rewrite the latest message into a self-contained
search query with references resolved against the recent turns.

Contract: the rewrite NEVER fails a request. Any exception, empty output,
or output over `MAX_REWRITTEN_CHARS` logs a WARNING (tenant + reason) and
returns the original query untouched. Empty history skips the call
entirely, so stateless requests pay zero added latency.
"""

from loguru import logger

from src.config import Config
from src.db.qdrant_client import DEFAULT_TENANT
from src.generation.providers import ProviderOverrides
from src.generation.sanitize import sanitize_untrusted

__all__ = ["MAX_REWRITTEN_CHARS", "REWRITE_HISTORY_TURNS", "rewrite_query"]

MAX_REWRITTEN_CHARS = 500
REWRITE_HISTORY_TURNS = 6

_REWRITE_PROMPT = """Rewrite the latest user message as ONE standalone search query for a document-retrieval system.

Rules:
- Resolve pronouns and vague references ("it", "that", "the second one") using the conversation below.
- Keep the concrete nouns and terms of the latest message.
- Output ONLY the rewritten query: no quotes, no explanation, no newlines.
- If the latest message is already standalone, copy it verbatim.

Conversation (oldest first):
{conversation}

Latest user message:
{query}

Standalone search query:"""


def _cheap_overrides() -> ProviderOverrides:
    """Prefer the cheap verify tier for the rewrite when it is configured
    and distinct from the default generation model; else default model."""
    verify_model = getattr(Config, "VERIFY_MODEL", "") or ""
    if verify_model and verify_model != Config.GROQ_MODEL:
        return ProviderOverrides(groq_model=verify_model)
    return ProviderOverrides()


def _conversation(history: list[dict]) -> str:
    """Recent turns, one `role: content` line each, re-sanitized."""
    lines: list[str] = []
    for turn in history[-REWRITE_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        content = sanitize_untrusted(str(turn.get("content") or "")).strip()
        if content:
            lines.append(f"{turn.get('role', 'user')}: {content}")
    return "\n".join(lines)


def rewrite_query(
    query: str,
    history: list[dict] | None,
    tenant_id: str = DEFAULT_TENANT,
) -> str:
    """Rewrite `query` into a standalone search query using `history`.

    Returns `query` unchanged when history is empty or the rewrite fails —
    callers can treat the result as always usable.
    """
    if not history:
        return query

    # Imported here: `chain` imports this module for the pipeline wiring.
    from src.generation.chain import _invoke_llm

    try:
        prompt = _REWRITE_PROMPT.format(
            conversation=_conversation(history), query=query
        )
        raw, _usage = _invoke_llm(prompt, provider_overrides=_cheap_overrides())
        rewritten = sanitize_untrusted(str(raw or "").strip())
    except Exception as exc:
        logger.warning(f"Query rewrite failed tenant={tenant_id} reason=llm_error: {exc}")
        return query

    if not rewritten:
        logger.warning(f"Query rewrite failed tenant={tenant_id} reason=empty_output")
        return query
    if len(rewritten) > MAX_REWRITTEN_CHARS:
        logger.warning(
            f"Query rewrite failed tenant={tenant_id} reason=output_too_long "
            f"chars={len(rewritten)}"
        )
        return query
    logger.debug(
        f"Query rewritten tenant={tenant_id} from_chars={len(query)} "
        f"to_chars={len(rewritten)}"
    )
    return rewritten
