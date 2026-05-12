"""CrossEncoderReranker class for cross-encoder reranking."""

from typing import Optional, List, Dict

from sentence_transformers import CrossEncoder
from loguru import logger

from src.config import Config


class CrossEncoderReranker:
    """Cross-encoder reranking with lazy loading.
    
    This class encapsulates cross-encoder model loading and reranking
    operations, providing a clean interface for result reranking.
    """
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """Initialize the CrossEncoderReranker.
        
        Args:
            model_name: Name of the cross-encoder model.
                       Defaults to Config.RERANKER_MODEL.
            device: Device to run model on ('cpu' or 'cuda').
        """
        self._model_name = model_name or Config.RERANKER_MODEL
        self._device = device
        self._model: Optional[CrossEncoder] = None
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy-load model on first access.
        
        Returns:
            The CrossEncoder instance.
        """
        if self._model is None:
            self._model = CrossEncoder(self._model_name, device=self._device)
            logger.info(f"Loaded cross encoder model: {self._model_name}")
        return self._model
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[Dict]:
        """Rerank chunks against a query using the cross-encoder.
        
        Args:
            query: The query string.
            chunks: List of chunk dictionaries to rerank.
            top_k: Number of top results to return.
            score_threshold: Optional minimum score threshold.
                           Chunks below this threshold are filtered out.
            
        Returns:
            List of reranked chunk dictionaries with 'rerank_score' added.
            
        Raises:
            ValueError: If top_k is not a positive integer.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        if not chunks:
            logger.warning("rerank() called with empty chunks list — returning []")
            return []
        
        # Create query-chunk pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        # Apply score threshold if provided
        if score_threshold is not None:
            before = len(reranked)
            reranked = [(c, s) for c, s in reranked if s >= score_threshold]
            logger.debug(f"Score threshold {score_threshold} filtered {before - len(reranked)} chunks")
        
        # Return top_k results
        top = reranked[:top_k]
        logger.debug(f"Reranked {len(chunks)} chunks → returning {len(top)} (top score: {top[0][1]:.4f})" if top else "No chunks passed reranking")
        
        return [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in top
        ]
    
    def reset(self):
        """Clear model from memory (useful for testing)."""
        self._model = None
        logger.debug("CrossEncoderReranker reset")
