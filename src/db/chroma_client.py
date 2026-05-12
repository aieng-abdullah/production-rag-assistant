"""
ChromaDB client using pure LangChain Chroma integration.

Backward compatibility wrapper around VectorStore class.
"""

from typing import Optional, List, Dict

from langchain_chroma import Chroma
from loguru import logger
from src.config import Config
from src.core.vector_store import VectorStore

# Lazy-loaded vectorstore instance (for backward compatibility)
_vectorstore_instance: Optional[VectorStore] = None


def _get_vectorstore() -> VectorStore:
    """Get or initialize the VectorStore (backward compatibility wrapper)."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStore()
    return _vectorstore_instance


def get_collection():
    """
    Returns the underlying Chroma collection for compatibility.
    Accesses the internal _collection from LangChain Chroma.
    """
    vectorstore = _get_vectorstore()
    return vectorstore.vectorstore._collection


def upsert_chunks(chunks: List[Dict]) -> int:
    """
    Add or update chunks in the vectorstore.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and metadata.
        
    Returns:
        Number of chunks upserted.
    """
    return _get_vectorstore().upsert_chunks(chunks)


def load_all_chunks() -> List[Dict]:
    """
    Load all chunks from the vectorstore.
    
    Returns:
        List of all chunk dictionaries with metadata.
    """
    return _get_vectorstore().load_all()


def reset_client():
    """
    Resets the singleton vectorstore (useful for tests).
    """
    global _vectorstore_instance
    _vectorstore_instance = None
    logger.debug("VectorStore reset")


def get_vectorstore() -> Chroma:
    """Get the LangChain Chroma vectorstore for use in chains.

    Returns:
        Chroma vectorstore instance.
    """
    return _get_vectorstore().vectorstore
