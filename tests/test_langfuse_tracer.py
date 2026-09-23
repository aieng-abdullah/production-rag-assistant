"""Tests for Langfuse tracer module."""

from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def reset_langfuse_state():
    """Reset module-level state before each test."""
    import src.monitoring.langfuse_tracer as mod
    mod._client = None
    mod._client_failed = False
    mod._disabled_logged = False
    yield
    mod._client = None
    mod._client_failed = False
    mod._disabled_logged = False


def test_langfuse_configured_true():
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = "pk"
        mock_cfg.LANGFUSE_SECRET_KEY = "sk"
        from src.monitoring.langfuse_tracer import langfuse_configured
        assert langfuse_configured() is True


def test_langfuse_configured_false_missing_keys():
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = ""
        mock_cfg.LANGFUSE_SECRET_KEY = ""
        from src.monitoring.langfuse_tracer import langfuse_configured
        assert langfuse_configured() is False


def test_get_langfuse_client_not_configured():
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = ""
        mock_cfg.LANGFUSE_SECRET_KEY = ""
        from src.monitoring.langfuse_tracer import get_langfuse_client
        result = get_langfuse_client()
        assert result is None


def test_get_langfuse_client_already_failed():
    import src.monitoring.langfuse_tracer as mod
    mod._client_failed = True
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = "pk"
        mock_cfg.LANGFUSE_SECRET_KEY = "sk"
        from src.monitoring.langfuse_tracer import get_langfuse_client
        result = get_langfuse_client()
        assert result is None


def test_get_langfuse_client_returns_cached():
    import src.monitoring.langfuse_tracer as mod
    mock_client = MagicMock()
    mod._client = mock_client
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = "pk"
        mock_cfg.LANGFUSE_SECRET_KEY = "sk"
        from src.monitoring.langfuse_tracer import get_langfuse_client
        result = get_langfuse_client()
        assert result is mock_client


@patch("langfuse.Langfuse", side_effect=ImportError("no langfuse"))
def test_get_langfuse_client_import_error(mock_cls):
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = "pk"
        mock_cfg.LANGFUSE_SECRET_KEY = "sk"
        mock_cfg.LANGFUSE_HOST = "https://example.com"
        from src.monitoring.langfuse_tracer import get_langfuse_client
        result = get_langfuse_client()
        assert result is None


@patch("langfuse.Langfuse", side_effect=RuntimeError("init failed"))
def test_get_langfuse_client_generic_error(mock_cls):
    with patch("src.monitoring.langfuse_tracer.Config") as mock_cfg:
        mock_cfg.LANGFUSE_PUBLIC_KEY = "pk"
        mock_cfg.LANGFUSE_SECRET_KEY = "sk"
        mock_cfg.LANGFUSE_HOST = "https://example.com"
        from src.monitoring.langfuse_tracer import get_langfuse_client
        result = get_langfuse_client()
        assert result is None


def test_flush_langfuse_no_client():
    import src.monitoring.langfuse_tracer as mod
    mod._client = None
    from src.monitoring.langfuse_tracer import flush_langfuse
    flush_langfuse()


def test_flush_langfuse_calls_flush():
    import src.monitoring.langfuse_tracer as mod
    mock_client = MagicMock()
    mod._client = mock_client
    from src.monitoring.langfuse_tracer import flush_langfuse
    flush_langfuse()
    mock_client.flush.assert_called_once()


def test_flush_langfuse_exception_silenced():
    import src.monitoring.langfuse_tracer as mod
    mock_client = MagicMock()
    mock_client.flush.side_effect = RuntimeError("flush error")
    mod._client = mock_client
    from src.monitoring.langfuse_tracer import flush_langfuse
    flush_langfuse()
