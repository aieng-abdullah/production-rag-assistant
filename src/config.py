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

    # --- Relational DB (users, auth, quotas — PLAN PR-2a) ---
    # SQLite by default (local dev + tests); set DATABASE_URL to Postgres in prod.
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}")

    # --- Embeddings ---
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranker ---
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Groq LLM (primary, free) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    # Judge model for claim entailment verification (PLAN PR-4b-ii).
    VERIFY_MODEL = os.getenv("VERIFY_MODEL", "openai/gpt-oss-20b")

    # --- Anthropic (optional failover) ---
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # --- OpenAI (optional failover) ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- Google OAuth + JWT (PLAN PR-2b, env-gated) ---
    # Absent → /auth/* returns 501 and the app runs without sign-in.
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI = os.getenv(
        "GOOGLE_REDIRECT_URI", "http://localhost:8001/auth/google/callback"
    )
    # HS256 signing key — env-only, never logged, required once Google creds are set.
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    JWT_TTL_DAYS = int(os.getenv("JWT_TTL_DAYS", "7"))
    # Browser redirect target after OAuth callback (Streamlit UI).
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
    # FastAPI base for browser links (Google OAuth entrypoint, PR-6 client).
    # 8001 — chromadb owns 8000 in docker-compose.
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

    # --- Retrieval Params ---
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 100
    TOP_K_RERANK = 5
    RRF_K = 60

    # --- Quotas (PLAN PR-3b, env-tunable) ---
    DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "20"))
    DOCUMENT_LIMIT = int(os.getenv("DOCUMENT_LIMIT", "5"))
    STORAGE_LIMIT_MB = int(os.getenv("STORAGE_LIMIT_MB", "100"))
    # Progressive auth wall: free anonymous queries before the login prompt.
    ANON_QUERY_LIMIT = int(os.getenv("ANON_QUERY_LIMIT", "5"))

    # --- Workspaces (PLAN PR-4): two niches, one engine ---
    WORKSPACES = ("legal", "academic")
    DEFAULT_WORKSPACE = "academic"

    # --- Evaluation ---
    FAITHFULNESS_THRESHOLD = 0.80
    ANSWER_RELEVANCY_THRESHOLD = 0.80

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Langfuse (optional tracing) ---
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Stripe billing (PLAN PR-5) — routes register only when the secret key
    # is present; webhook additionally needs its signing secret.
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRO_PRICE = os.getenv("STRIPE_PRO_PRICE", "")
    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")

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


def _hf_models_cached() -> bool:
    """Both RAG models already on disk (no download needed)?"""
    hub = Path(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ) / "hub"
    repos = (Config.EMBEDDING_MODEL, Config.RERANKER_MODEL)
    return all(
        (hub / f"models--{repo.replace('/', '--')}").exists() for repo in repos
    )


# Must run before `huggingface_hub` imports (it reads this env at import time) —
# every heavy module imports src.config first. Saves the ~5s hub roundtrip on
# each model init. Only when both models are cached: first run still downloads.
# Explicit user setting (e.g. HF_HUB_OFFLINE=0) always wins (setdefault).
if _hf_models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")