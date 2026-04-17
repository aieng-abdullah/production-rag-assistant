"""BM25 keyword search index builder."""

from typing import List, Dict
from rank_bm25 import BM25Okapi

from src.db.chroma_client import get_chroma_client, get_or_create_collection
from src.monitoring.logger import get_logger

logger = get_logger("bm25_index")


class BM25Index:
    """In-memory BM25 index for keyword search."""
    
    def __init__(self):
        """Initialize empty index."""
        self.bm25 = None
        self.chunk_ids = []
        self.corpus = []
    
    def build(self, doc_id: str = None):
        """Build BM25 index from ChromaDB corpus."""
        collection = get_or_create_collection(get_chroma_client())
        
        # Get all chunks (or filter by doc)
        where_filter = {"doc_id": doc_id} if doc_id else None
        results = collection.get(where=where_filter)
        
        # Store chunk IDs and texts
        self.chunk_ids = results["ids"]
        documents = results["documents"]
        
        # Tokenize for BM25
        self.corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        
        logger.info(f"BM25 index built: {len(self.chunk_ids)} chunks")
    
    def search(self, query: str, top_k: int = 20) -> List[tuple]:
        """Search BM25 index, return (chunk_id, score) pairs."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Tokenize query
        tokenized = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized)
        
        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(self.chunk_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        
        logger.debug(f"BM25 search: {len(results)} results")
        return results
