"""Centralized configuration for RAG Research Assistant."""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Streamlit secrets.toml fallback (gitignored; populated on Streamlit Cloud).
# Local .env wins — load_dotenv already set those keys, setdefault fills gaps.
# No streamlit import: src/ stays framework-free (AGENTS.md).
_SECRETS_TOML = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
if _SECRETS_TOML.is_file():
    import tomllib

    with _SECRETS_TOML.open("rb") as _f:
        for _key, _val in tomllib.load(_f).items():
            os.environ.setdefault(_key, str(_val))


class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

    # --- Qdrant Cloud (vector store — Phase 2) ---
    COLLECTION_NAME = "research_docs"
    # Cloud: full URL, e.g. https://xxx.cloud.qdrant.io:6333
    # Local tests/demo: ":memory:" or a filesystem path for local persistence.
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

    # --- Admin ---
    ADMIN_EMAILS: list[str] = [
        e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
    ]

    # --- Relational DB (users, auth, quotas — PLAN PR-2a) ---
    # SQLite by default (local dev + tests); set DATABASE_URL to Postgres in prod.
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}")

    # --- Voyage AI (embeddings + rerank, API-only — PLAN-render-react Phase 1) ---
    # Optional at Config level: GROQ stays the only required key.
    # Missing key degrades loudly (WARNING at validate()), never crashes boot.
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
    VOYAGE_BASE_URL = os.getenv("VOYAGE_BASE_URL", "https://api.voyageai.com/v1")
    VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite")
    VOYAGE_RERANKER_MODEL = os.getenv("VOYAGE_RERANKER_MODEL", "rerank-3-lite")
    # Trial accounts (no payment method) are capped at 3 RPM / 10K TPM:
    # 64 inputs ≈ 3K tokens → 3 paced batches/min stays inside both caps.
    VOYAGE_EMBED_BATCH_SIZE = int(os.getenv("VOYAGE_EMBED_BATCH_SIZE", "64"))
    # Minimum spacing between batched embedding calls during ingestion.
    VOYAGE_EMBED_PACE_S = float(os.getenv("VOYAGE_EMBED_PACE_S", "21"))
    # Token budget, enforced as a rolling 60s window per model by
    # src/voyage_pacer.py rather than as a fixed interval between calls:
    # a rerank costs ~130x a query embedding, so one interval is too slow
    # for the cheap call and too loose for the expensive one.
    VOYAGE_TPM_LIMIT = int(os.getenv("VOYAGE_TPM_LIMIT", "10000"))
    VOYAGE_RPM_LIMIT = int(os.getenv("VOYAGE_RPM_LIMIT", "3"))
    # Ceiling on a single pacing sleep, so one fat rerank cannot stall a
    # query for a minute without the log saying so.
    VOYAGE_MAX_WAIT_S = float(os.getenv("VOYAGE_MAX_WAIT_S", "45"))
    # TTL for the query-embedding cache (src/ingestion/embed_cache.py).
    VOYAGE_CACHE_TTL_S = int(os.getenv("VOYAGE_CACHE_TTL_S", "3600"))
    EMBED_TIMEOUT_S = float(os.getenv("EMBED_TIMEOUT_S", "30"))
    RERANK_TIMEOUT_S = float(os.getenv("RERANK_TIMEOUT_S", "10"))

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

    # --- Retrieval Params ---
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 100
    # 8: reranker drops good chunks at 5 (measured — intro chunk pushed out,
    # junk claimed a slot). Wider prompt costs ~nothing; rerank scans top-20 anyway.
    TOP_K_RERANK = 8
    RRF_K = 60

    # --- Quotas (PLAN PR-3b, env-tunable) ---
    DAILY_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "20"))
    DOCUMENT_LIMIT = int(os.getenv("DOCUMENT_LIMIT", "5"))
    STORAGE_LIMIT_MB = int(os.getenv("STORAGE_LIMIT_MB", "100"))
    # Guest tier: free try-out before the login wall.
    GUEST_QUERY_LIMIT = int(os.getenv("GUEST_QUERY_LIMIT", "3"))
    # API guest tier (restored FastAPI layer — PLAN PR-6): ANON_* names kept
    # because src/services/quotas and its tests pin them. Same defaults.
    ANON_QUERY_LIMIT = int(os.getenv("ANON_QUERY_LIMIT", "3"))
    ANON_DOCUMENT_LIMIT = int(os.getenv("ANON_DOCUMENT_LIMIT", "1"))

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
    # Production: off by default. Set LANGFUSE_ENABLED=on to enable.
    LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "off") == "on"
    # Sample rate 0.0-1.0. 0.0 = disabled, 1.0 = all. Production default 0.0.
    LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "0.0"))
    # Debug mode: if on, includes full prompt/answer/chunk text in traces (dev only).
    LANGFUSE_DEBUG = os.getenv("LANGFUSE_DEBUG", "off") == "on"

    # Stripe billing (PLAN PR-5) — routes register only when the secret key
    # is present; webhook additionally needs its signing secret.
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRO_PRICE = os.getenv("STRIPE_PRO_PRICE", "")
    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")

    # Google OAuth (PLAN PR-2b)
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI = os.getenv(
        "GOOGLE_REDIRECT_URI", "http://localhost:8001/auth/google/callback"
    )
    # Browser redirect target after the OAuth callback (SPA origin).
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # API auth (restored FastAPI layer): HS256 signing key — env-only,
    # never logged, required once Google creds are set.
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    JWT_TTL_DAYS = int(os.getenv("JWT_TTL_DAYS", "7"))
    # Demo login (PR-6): "on" | "off" | "" = auto (on exactly when no
    # Google creds — local clones get demo access, deployments with real
    # auth do not). Explicit env always wins.
    ENABLE_DEMO_LOGIN = os.getenv("ENABLE_DEMO_LOGIN", "")

    # Auth mode
    APP_AUTH_ENABLED = os.getenv("APP_AUTH", "off") == "on"

    # Guest tier (replaces ANON_QUERY_LIMIT)
    GUEST_QUERY_LIMIT = int(os.getenv("GUEST_QUERY_LIMIT", "3"))

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
        if not cls.VOYAGE_API_KEY:
            from loguru import logger

            logger.warning(
                "VOYAGE_API_KEY missing — embeddings and rerank are API-only "
                "(PLAN-render-react Phase 1): ingestion and retrieval queries "
                "will fail until it is set in .env"
            )


# Demo flag resolution (PR-6): auto = demo on exactly when Google auth is absent.
if Config.ENABLE_DEMO_LOGIN not in ("on", "off"):
    Config.ENABLE_DEMO_LOGIN = "off" if Config.GOOGLE_CLIENT_ID else "on"

# Local demo needs signable tokens without forcing every clone to invent a
# secret: generate an in-memory one ONLY when no Google creds and no env
# secret (dies with the process — tokens never survive restart, never logged).
# Google creds without JWT_SECRET fail loudly at request time (src/api/deps).
if not Config.JWT_SECRET and not Config.GOOGLE_CLIENT_ID:
    Config.JWT_SECRET = secrets.token_urlsafe(32)
