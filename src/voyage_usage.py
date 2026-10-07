"""Voyage free-grant budget tracking (PLAN-render-react risk #4).

Embeddings + rerank share one 200M-token grant. Every response reports
`usage.total_tokens`; we accumulate process-wide and warn once at 80% so
the burn is visible before the card is required.
"""

from threading import Lock

from loguru import logger

GRANT_TOKENS = 200_000_000
WARN_AT_TOKENS = int(GRANT_TOKENS * 0.8)

_lock = Lock()
_total_tokens = 0
_warned = False


def record(source: str, tokens: int) -> None:
    """Add one Voyage response's token usage to the process total."""
    global _total_tokens, _warned
    if tokens <= 0:
        return
    with _lock:
        _total_tokens += tokens
        total = _total_tokens
        warn = False
        if not _warned and total >= WARN_AT_TOKENS:
            _warned = True
            warn = True
    logger.debug(f"Voyage {source}: +{tokens} tokens (grant total: {total})")
    if warn:
        logger.warning(
            f"Voyage token grant at {total:,} / {GRANT_TOKENS:,} (80%) — "
            f"embeddings + rerank share this pool; budget more usage or add a card"
        )


def total_tokens() -> int:
    """Process-wide Voyage token total (tests / diagnostics)."""
    with _lock:
        return _total_tokens


def reset() -> None:
    """Reset the counters (tests)."""
    global _total_tokens, _warned
    with _lock:
        _total_tokens = 0
        _warned = False
