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
    
    # LLM settings (OpenAI for generation)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # LangSmith settings
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-research-assistant")
    
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
