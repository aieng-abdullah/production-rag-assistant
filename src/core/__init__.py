"""Core infrastructure classes for RAG Research Assistant.

This module provides class-based implementations of core functionality
with backward compatibility wrappers for existing functional code.
"""

from src.core.embedding_model import EmbeddingModel
from src.core.vector_store import VectorStore
from src.core.bm25_retriever import BM25Retriever
from src.core.cross_encoder import CrossEncoderReranker

__all__ = [
    "EmbeddingModel",
    "VectorStore",
    "BM25Retriever",
    "CrossEncoderReranker",
]
