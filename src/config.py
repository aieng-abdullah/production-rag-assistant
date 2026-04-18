"""Centralized configuration management."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration settings."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma"
    
    # ChromaDB settings
    CHROMA_HOST = os.getenv("CHROMA_HOST", "")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    COLLECTION_NAME = "research_docs"
    
    # Groq LLM settings (fast, free tier available)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-oss-120b")  # or: llama-3.3-70b-versatile, mixtral-8x7b-32768
    
    # Active LLM model
    LLM_MODEL = GROQ_MODEL
    
    # LangSmith settings
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-research-assistant")
    
    # Langfuse settings (open-source alternative to LangSmith)
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # or self-hosted URL
    
    # Retrieval settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    TOP_K_VECTOR = 20
    TOP_K_BM25 = 20
    TOP_K_RERANK = 5
    RRF_K = 60  # Reciprocal Rank Fusion constant
    
    # Evaluation thresholds
    FAITHFULNESS_THRESHOLD = 0.80
    ANSWER_RELEVANCY_THRESHOLD = 0.80
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
