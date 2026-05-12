"""RetrievalPipeline class for hybrid retrieval orchestration."""

from typing import List, Dict, Optional

from loguru import logger

from src.config import Config
from src.core.vector_store import VectorStore
from src.core.bm25_retriever import BM25Retriever
from src.core.cross_encoder import CrossEncoderReranker


class RetrievalPipeline:
    """Hybrid retrieval orchestration: BM25 + Vector + RRF + Rerank.
    
    This class orchestrates the full retrieval pipeline:
    1. BM25 keyword search
    2. Vector semantic search
    3. RRF (Reciprocal Rank Fusion) to combine results
    4. Cross-encoder reranking for final refinement
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        cross_encoder: CrossEncoderReranker,
        top_k: int = 5,
    ):
        """Initialize the RetrievalPipeline.
        
        Args:
            vector_store: VectorStore instance for semantic search.
            bm25_retriever: BM25Retriever instance for keyword search.
            cross_encoder: CrossEncoderReranker instance for reranking.
            top_k: Number of final results to return.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder
        self.top_k = top_k
    
    def retrieve(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Execute full retrieval pipeline.
        
        Args:
            query: Search query string.
            chunks: List of all chunks to search from.
            
        Returns:
            List of reranked chunk dictionaries.
        """
        # BM25 search
        bm25_results = self.bm25_retriever.search(query, chunks)
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        
        # Vector search
        query_embedding = self.vector_store.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=20)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # RRF fusion
        fused = self._rrf_fusion(bm25_results, vector_results, top_k=20)
        logger.info(f"RRF fusion returned {len(fused)} results")
        
        # Rerank
        reranked = self.cross_encoder.rerank(query, fused, self.top_k)
        logger.info(f"Rerank returned {len(reranked)} results")
        
        return reranked
    
    @staticmethod
    def _rrf_fusion(bm25_results: List[Dict], vector_results: List[Dict], top_k: int) -> List[Dict]:
        """Reciprocal Rank Fusion to combine BM25 and vector results.
        
        Args:
            bm25_results: Results from BM25 search.
            vector_results: Results from vector search.
            top_k: Number of results to return.
            
        Returns:
            List of fused chunk dictionaries with RRF scores.
        """
        scores: Dict[str, float] = {}
        all_chunks: Dict[str, Dict] = {}
        
        # Score BM25 results
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)
            all_chunks[chunk_id] = chunk
        
        # Score vector results
        for rank, chunk in enumerate(vector_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (Config.RRF_K + rank)
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk
        
        # Sort by RRF score
        ranked_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k with RRF scores
        fused_results = []
        for chunk_id, score in ranked_chunks[:top_k]:
            chunk = all_chunks[chunk_id].copy()
            chunk["rrf_score"] = score
            fused_results.append(chunk)
        
        return fused_results
