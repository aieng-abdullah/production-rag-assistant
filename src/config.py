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
    CHROMA_MODE = os.getenv("CHROMA_MODE", "local")  
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    COLLECTION_NAME = "research_docs"

    # --- Embeddings ---
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranker ---
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Groq LLM ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  

    # --- Retrieval Params ---
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    TOP_K_VECTOR = 20
    TOP_K_BM25 = 20
    TOP_K_RERANK = 5
    RRF_K = 60  # 

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
        required = {
            "GROQ_API_KEY": cls.GROQ_API_KEY,
        }
        missing = [key for key, val in required.items() if not val]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}\n"
                f"Check your .env file."
            )