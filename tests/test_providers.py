"""Tests for LLM provider chain, client factory, and retry/failover logic."""

import pytest
from unittest.mock import patch, MagicMock

from src.generation.providers import Provider, build_provider_chain, create_langchain_client


class TestBuildProviderChain:
    """Test that the provider chain is built correctly from Config."""

    @patch("src.config.Config.GROQ_API_KEY", "gsk_test")
    @patch("src.config.Config.GROQ_MODEL", "llama-3.3-70b-versatile")
    @patch("src.config.Config.ANTHROPIC_API_KEY", "")
    @patch("src.config.Config.OPENAI_API_KEY", "")
    def test_groq_only(self):
        chain = build_provider_chain()
        assert len(chain) == 1
        assert chain[0].name == "groq"
        assert chain[0].api_key == "gsk_test"

    @patch("src.config.Config.GROQ_API_KEY", "gsk_test")
    @patch("src.config.Config.GROQ_MODEL", "llama-3.3-70b-versatile")
    @patch("src.config.Config.ANTHROPIC_API_KEY", "sk-ant_test")
    @patch("src.config.Config.ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    @patch("src.config.Config.OPENAI_API_KEY", "sk-test")
    @patch("src.config.Config.OPENAI_MODEL", "gpt-4o")
    def test_all_three_providers(self):
        chain = build_provider_chain()
        assert len(chain) == 3
        assert [p.name for p in chain] == ["groq", "anthropic", "openai"]

    @patch("src.config.Config.GROQ_API_KEY", "")
    @patch("src.config.Config.ANTHROPIC_API_KEY", "sk-ant_test")
    @patch("src.config.Config.ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    @patch("src.config.Config.OPENAI_API_KEY", "sk-test")
    @patch("src.config.Config.OPENAI_MODEL", "gpt-4o")
    def test_anthropic_and_openai_only(self):
        chain = build_provider_chain()
        assert len(chain) == 2
        assert [p.name for p in chain] == ["anthropic", "openai"]

    @patch("src.config.Config.GROQ_API_KEY", "")
    @patch("src.config.Config.ANTHROPIC_API_KEY", "")
    @patch("src.config.Config.OPENAI_API_KEY", "")
    def test_no_keys_raises(self):
        with pytest.raises(RuntimeError, match="No LLM providers configured"):
            build_provider_chain()

    @patch("src.config.Config.GROQ_API_KEY", "gsk")
    @patch("src.config.Config.GROQ_MODEL", "model-groq")
    @patch("src.config.Config.ANTHROPIC_API_KEY", "ant")
    @patch("src.config.Config.ANTHROPIC_MODEL", "model-ant")
    @patch("src.config.Config.OPENAI_API_KEY", "oai")
    @patch("src.config.Config.OPENAI_MODEL", "model-oai")
    def test_preserves_order(self):
        chain = build_provider_chain()
        assert chain[0].name == "groq"
        assert chain[1].name == "anthropic"
        assert chain[2].name == "openai"


class TestCreateLangchainClient:
    """Test that the client factory returns the correct LangChain class."""

    @patch("langchain_groq.ChatGroq")
    def test_groq_client(self, mock_chat_groq):
        provider = Provider("groq", "gsk_test", "llama-3.3-70b-versatile")
        create_langchain_client(provider)
        mock_chat_groq.assert_called_once_with(api_key="gsk_test", model="llama-3.3-70b-versatile")

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("langchain_anthropic") is None,
        reason="langchain-anthropic not installed",
    )
    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_client(self, mock_chat_anthropic):
        provider = Provider("anthropic", "sk-ant_test", "claude-sonnet-4-20250514")
        create_langchain_client(provider)
        mock_chat_anthropic.assert_called_once_with(api_key="sk-ant_test", model="claude-sonnet-4-20250514")

    @patch("langchain_openai.ChatOpenAI")
    def test_openai_client(self, mock_chat_openai):
        provider = Provider("openai", "sk-test", "gpt-4o")
        create_langchain_client(provider)
        mock_chat_openai.assert_called_once_with(api_key="sk-test", model="gpt-4o")

    def test_unknown_provider_raises(self):
        provider = Provider("unknown", "key", "model")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_langchain_client(provider)


class TestProviderDataclass:
    def test_fields(self):
        p = Provider(name="groq", api_key="key", model="model")
        assert p.name == "groq"
        assert p.api_key == "key"
        assert p.model == "model"
