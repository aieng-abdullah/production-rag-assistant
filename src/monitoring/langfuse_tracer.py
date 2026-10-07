"""Optional Langfuse client for RAG tracing (see `src/monitoring/langfuse_spec.md`)."""

from __future__ import annotations

import random
import threading
from typing import Any

from loguru import logger

from src.config import Config

_client: Any | None = None
_client_failed = False
_disabled_logged = False
_lock = threading.Lock()


def langfuse_configured() -> bool:
    """Check if Langfuse is configured with credentials."""
    return bool(Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY)


def langfuse_enabled() -> bool:
    """Check if Langfuse tracing is globally enabled via env var."""
    return Config.LANGFUSE_ENABLED and langfuse_configured()


def should_trace_request() -> bool:
    """Determine if a request should be traced based on sample rate."""
    if not langfuse_enabled():
        return False
    sample_rate = Config.LANGFUSE_SAMPLE_RATE
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    return random.random() < sample_rate


def langfuse_debug() -> bool:
    """Check if debug mode is enabled (includes full prompt/answer text)."""
    return Config.LANGFUSE_DEBUG and langfuse_enabled()


def get_langfuse_client() -> Any | None:
    """Return a Langfuse client, or None if tracing is disabled or unavailable."""
    global _client, _client_failed, _disabled_logged

    if not langfuse_configured():
        if not _disabled_logged:
            logger.debug(
                "Langfuse tracing off: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
            )
            _disabled_logged = True
        return None
    if _client_failed:
        return None

    with _lock:
        if _client is not None:
            return _client
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=Config.LANGFUSE_PUBLIC_KEY,
                secret_key=Config.LANGFUSE_SECRET_KEY,
                host=Config.LANGFUSE_HOST,
            )
            logger.info("Langfuse client initialized for tracing")
            return _client
        except ImportError:
            logger.warning("Langfuse package not installed; tracing disabled.")
            _client_failed = True
            return None
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            _client_failed = True
            return None


def flush_langfuse() -> None:
    if _client is None:
        return
    try:
        _client.flush()
    except Exception:
        pass


def _redact_for_debug(data: Any, debug: bool) -> Any:
    """Redact sensitive fields if debug mode is off."""
    if debug:
        return data
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            if k in ("query", "answer", "answer_text", "text", "chunk", "chunks", "prompt", "user_id", "tenant_id", "email"):
                redacted[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                redacted[k] = _redact_for_debug(v, debug)
            else:
                redacted[k] = v
        return redacted
    if isinstance(data, list):
        return [_redact_for_debug(item, debug) for item in data]
    return data