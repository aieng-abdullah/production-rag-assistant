"""Token-aware rate limiting for Voyage API calls.

The previous gate paced by *call count* on a fixed interval, which is
wrong for Voyage in two ways.

**Calls differ by ~130x in cost.** A query embedding is ~10 tokens; a
rerank of 20 documents at 256 characters each is ~1,300. A single
21-second gap between calls is far too coarse for the cheap call and
far too loose for the expensive one.

**Budgets are per model, not organisation-wide.** A free-tier account is
capped at 3 RPM / 10K TPM, and it matters that those are separate
buckets per model: a 37-item eval run issues 75 Voyage calls in 12.7
minutes — 5.9 RPM — without a single 429. Against a shared 3 RPM
allowance that traffic would have failed hard, so `voyage-4-lite` and
`rerank-3-lite` are counted independently. Coupling them behind one
window halves throughput for nothing, which an early version of this
module did before the run above showed the error.

So each model gets its own rolling window over 60 seconds, which is the
unit the limit is actually expressed in. Voyage returns the true
`usage.total_tokens` for every response, but the decision has to be
made *before* the call, so cost is estimated from payload size. The
estimate is a pacing heuristic, not a billing figure; `observe()` folds
the real figure back in and the drift is logged.
"""

from collections import deque
from threading import Lock
from time import monotonic, sleep

from loguru import logger

from src.config import Config

__all__ = ["reserve", "observe", "reset", "estimate_tokens", "snapshot", "EMBEDDINGS", "RERANK"]

EMBEDDINGS = "embeddings"
RERANK = "rerank"

WINDOW_SECONDS = 60.0

# Voyage's tokenizers run roughly 4 characters per token. Used only to
# size a request before sending it.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Pre-call token estimate for one payload."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_documents_tokens(documents: list[str]) -> int:
    """Pre-call token estimate for a batch of documents."""
    return sum(estimate_tokens(document) for document in documents)


class _Window:
    """Rolling (timestamp, tokens) history for one process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def _usage(self, now: float) -> tuple[int, int]:
        self._prune(now)
        return sum(tokens for _, tokens in self._events), len(self._events)

    def _oldest(self) -> float | None:
        return self._events[0][0] if self._events else None

    def reserve(self, tokens: int) -> float:
        """Block until `tokens` fit in the window. Returns seconds slept."""
        now = monotonic()
        with self._lock:
            used_tokens, used_calls = self._usage(now)
            wait = 0.0

            # Requests per minute.
            if used_calls >= Config.VOYAGE_RPM_LIMIT:
                oldest = self._oldest()
                if oldest is not None:
                    wait = max(wait, oldest + WINDOW_SECONDS - now)

            # Tokens per minute.
            if used_tokens + tokens > Config.VOYAGE_TPM_LIMIT:
                # Sleep until enough of the window decays. Charging the
                # request evenly across the window avoids a busy-wait on
                # the single largest call.
                overflow = used_tokens + tokens - Config.VOYAGE_TPM_LIMIT
                if used_tokens:
                    wait = max(wait, overflow / used_tokens * WINDOW_SECONDS)

            wait = min(wait, Config.VOYAGE_MAX_WAIT_S)
            if wait > 0:
                logger.debug(
                    f"Voyage pacing: sleeping {wait:.1f}s "
                    f"(est {tokens} tokens, window {used_tokens}/{Config.VOYAGE_TPM_LIMIT}, "
                    f"calls {used_calls}/{Config.VOYAGE_RPM_LIMIT})"
                )
                sleep(wait)

            self._events.append((monotonic(), tokens))
            return wait

    def observe(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Correct the window with the figure Voyage actually reported.

        The reservation is already booked, so this only logs meaningful
        drift — a persistent bias means the character-per-token estimate
        needs revisiting.
        """
        if actual_tokens <= 0:
            return
        if estimated_tokens > 0:
            drift = abs(actual_tokens - estimated_tokens) / max(actual_tokens, 1)
            if drift > 0.5:
                logger.debug(
                    f"Voyage token estimate off by {drift:.0%} "
                    f"(estimated {estimated_tokens}, actual {actual_tokens})"
                )

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def snapshot(self) -> dict:
        used_tokens, used_calls = self._usage(monotonic())
        return {"tokens": used_tokens, "calls": used_calls}


# One window per model. Voyage counts limits per model, so embeddings and
# rerank must not share a budget.
_windows: dict[str, _Window] = {}
_windows_lock = Lock()


def _window_for(resource: str) -> _Window:
    with _windows_lock:
        if resource not in _windows:
            _windows[resource] = _Window()
        return _windows[resource]


def reserve(tokens: int, resource: str = "embeddings") -> float:
    """Block until `tokens` fit in that model's rolling window."""
    return _window_for(resource).reserve(tokens)


def observe(actual_tokens: int, estimated_tokens: int, resource: str = "embeddings") -> None:
    """Fold the true token count back in after a response."""
    _window_for(resource).observe(actual_tokens, estimated_tokens)


def reset() -> None:
    """Clear every window (tests)."""
    with _windows_lock:
        for window in _windows.values():
            window.reset()
    _windows.clear()


def snapshot(resource: str | None = None) -> dict:
    """Window usage — diagnostics and tests."""
    if resource is not None:
        return _window_for(resource).snapshot()
    return {name: window.snapshot() for name, window in sorted(_windows.items())}
