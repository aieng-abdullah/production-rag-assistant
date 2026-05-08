"""Optional Langfuse client for RAG tracing (see `src/monitoring/langfuse_spec.md`)."""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from src.config import Config

_client: Any | None = None
_client_failed = False
_disabled_logged = False
_lock = threading.Lock()


def langfuse_configured() -> bool:
    return bool(Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY)


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
