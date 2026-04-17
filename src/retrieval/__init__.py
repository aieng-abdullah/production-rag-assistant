"""Retrieval pipeline - hybrid search with BM25, vector, and reranking."""

from .pipeline import hybrid_search

__all__ = ["hybrid_search"]
