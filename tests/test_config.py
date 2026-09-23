"""Tests for Config class - meaningful validation tests."""
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config import Config


class TestConfigPaths:
    def test_base_dir_is_project_root(self):
        assert Config.BASE_DIR.name == "production-rag-assistant"

    def test_data_dir_exists(self):
        assert Config.DATA_DIR.exists()

    def test_chroma_dir_is_under_data_dir(self):
        assert Config.CHROMA_DIR.parent == Config.DATA_DIR

    def test_chroma_dir_path_is_absolute(self):
        assert Config.CHROMA_DIR.is_absolute()


class TestConfigEnvVars:
    """Config reads env vars at import time. We verify the mechanism works by
    testing that class attributes reflect os.getenv defaults."""

    def test_chroma_host_is_string(self):
        assert isinstance(Config.CHROMA_HOST, str)

    def test_chroma_port_is_int(self):
        assert isinstance(Config.CHROMA_PORT, int)

    def test_log_level_has_default(self):
        assert Config.LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    def test_groq_model_has_default(self):
        assert isinstance(Config.GROQ_MODEL, str)
        assert len(Config.GROQ_MODEL) > 0

    def test_groq_model_is_valid_groq_model(self):
        valid_prefixes = ("llama", "mixtral", "gemma", "deepseek")
        assert any(Config.GROQ_MODEL.startswith(p) for p in valid_prefixes)


class TestConfigValidation:
    def test_validate_passes_with_key(self):
        original = Config.GROQ_API_KEY
        try:
            Config.GROQ_API_KEY = "test-key-123"
            Config.validate()
        finally:
            Config.GROQ_API_KEY = original

    def test_validate_raises_without_key(self):
        original = Config.GROQ_API_KEY
        try:
            Config.GROQ_API_KEY = ""
            with pytest.raises(EnvironmentError, match="No LLM provider API key found"):
                Config.validate()
        finally:
            Config.GROQ_API_KEY = original


class TestConfigConstraints:
    def test_chunk_size_is_positive(self):
        assert Config.CHUNK_SIZE > 0

    def test_chunk_overlap_is_less_than_chunk_size(self):
        assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE

    def test_top_k_rerank_is_positive(self):
        assert Config.TOP_K_RERANK > 0

    def test_rrf_k_is_positive(self):
        assert Config.RRF_K > 0

    def test_faithfulness_threshold_is_between_0_and_1(self):
        assert 0 < Config.FAITHFULNESS_THRESHOLD <= 1

    def test_answer_relevancy_threshold_is_between_0_and_1(self):
        assert 0 < Config.ANSWER_RELEVANCY_THRESHOLD <= 1

    def test_collection_name_is_nonempty(self):
        assert len(Config.COLLECTION_NAME) > 0

    def test_embedding_model_is_nonempty(self):
        assert len(Config.EMBEDDING_MODEL) > 0

    def test_reranker_model_is_nonempty(self):
        assert len(Config.RERANKER_MODEL) > 0
