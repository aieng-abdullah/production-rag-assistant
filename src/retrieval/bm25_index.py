"""BM25 search using pure LangChain BM25Retriever.

Backward compatibility wrapper around BM25Retriever class.
"""

from typing import List, Dict

from langchain_community.retrievers import BM25Retriever as BM25RetrieverLC
from loguru import logger
from src.core.bm25_retriever import BM25Retriever


def build_bm25_index(chunks: List[Dict]) -> BM25RetrieverLC:
    """
    Build a BM25 index from chunks using LangChain BM25Retriever.
    
    Args:
        chunks: List of chunk dictionaries.
        
    Returns:
        LangChain BM25Retriever instance.
    """
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    return retriever._retriever


def bm25_search(bm25: BM25RetrieverLC, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
    """
    Search the BM25 index using LangChain retriever.
    
    Args:
        bm25: LangChain BM25Retriever instance.
        query: Search query string.
        chunks: List of chunks to search.
        top_k: Number of results to return.
        
    Returns:
        List of chunk dictionaries with search results.
    """
    try:
        retriever = BM25Retriever(top_k=top_k)
        retriever._retriever = bm25
        return retriever.search(query, chunks)
    except Exception as e:
        raise RuntimeError(f"Error while ranking documents: {e}")


def get_bm25_retriever(chunks: List[Dict], top_k: int = 20) -> BM25RetrieverLC:
    """Get a BM25Retriever for use in chains.
    
    Args:
        chunks: List of chunk dictionaries.
        top_k: Default number of results to return.
        
    Returns:
        LangChain BM25Retriever instance.
    """
    retriever = BM25Retriever(top_k=top_k)
    retriever.build_index(chunks)
    return retriever._retriever
