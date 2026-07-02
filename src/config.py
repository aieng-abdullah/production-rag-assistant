"""Centralized configuration for RAG Research Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma"

    # --- ChromaDB ---
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    COLLECTION_NAME = "research_docs"

    # --- Embeddings ---
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranker ---
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Groq LLM (primary, free) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # --- Anthropic (optional failover) ---
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # --- OpenAI (optional failover) ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- Retrieval Params ---
    CHUNK_SIZE = 350
    CHUNK_OVERLAP = 75
    TOP_K_RERANK = 5
    RRF_K = 60

    # --- Evaluation ---
    FAITHFULNESS_THRESHOLD = 0.80
    ANSWER_RELEVANCY_THRESHOLD = 0.80

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Langfuse (optional tracing) ---
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    @classmethod
    def validate(cls):
        """
        Call this once at app startup.
        Crashes immediately with a clear message if critical vars are missing.
        Much better than crashing mid-request with a cryptic API error.
        """
        has_any_key = bool(cls.GROQ_API_KEY or cls.ANTHROPIC_API_KEY or cls.OPENAI_API_KEY)
        if not has_any_key:
            raise EnvironmentError(
                "No LLM provider API key found.\n"
                "Set at least one of GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env,\n"
                "or add one via the sidebar in the app."
            )