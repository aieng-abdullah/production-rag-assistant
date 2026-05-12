"""EmbeddingModel class for encapsulating embedding generation."""

from typing import Optional, List

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import Config


class EmbeddingModel:
    """Encapsulates embedding generation with lazy loading.
    
    This class manages the lifecycle of the HuggingFace embedding model,
    providing lazy initialization to avoid loading the model until needed.
    """
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use.
                       Defaults to Config.EMBEDDING_MODEL.
            device: Device to run model on ('cpu' or 'cuda').
        """
        self._model_name = model_name or Config.EMBEDDING_MODEL
        self._device = device
        self._model: Optional[HuggingFaceEmbeddings] = None
    
    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Lazy-load model on first access.
        
        Returns:
            The HuggingFaceEmbeddings instance.
        """
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": self._device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Loaded embedding model: {self._model_name}")
        return self._model
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: The text to embed.
            
        Returns:
            List of float embedding values.
        """
        return self.model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding lists.
        """
        return self.model.embed_documents(texts)
    
    def reset(self):
        """Clear model from memory (useful for testing)."""
        self._model = None
        logger.debug("Embedding model reset")
