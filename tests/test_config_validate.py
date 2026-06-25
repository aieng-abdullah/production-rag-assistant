"""Tests for Config.validate()."""

import os
import pytest
from unittest.mock import patch


def test_validate_missing_groq_key():
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
        from src.config import Config
        original = Config.GROQ_API_KEY
        Config.GROQ_API_KEY = ""
        try:
            with pytest.raises(EnvironmentError, match="No LLM provider API key found"):
                Config.validate()
        finally:
            Config.GROQ_API_KEY = original


def test_validate_passes_with_key():
    from src.config import Config
    original = Config.GROQ_API_KEY
    Config.GROQ_API_KEY = "test_key_123"
    try:
        Config.validate()
    finally:
        Config.GROQ_API_KEY = original
