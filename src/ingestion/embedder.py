"""Free local embedding generation with Sentence Transformers."""

import asyncio
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from src.monitoring.logger import get_logger

logger = get_logger("embedder")

# Load free local model (384 dimensions, runs on CPU)
# Downloaded automatically on first run (~80MB)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _model


async def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Generate embeddings for chunks in batches using local model."""
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    
    logger.info(f"Embedding {len(texts)} chunks with local model (batch size: {batch_size})")
    
    model = _get_model()
    
    # Process in batches using asyncio to not block
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Run CPU-intensive encoding in thread pool
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, lambda: model.encode(batch, convert_to_list=True)
            )
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
            raise
    
    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding
    
    logger.info(f"Successfully embedded {len(chunks)} chunks")
    return chunks


def embed_query(text: str) -> List[float]:
    """Generate embedding for a single query text using local model."""
    model = _get_model()
    embedding = model.encode(text, convert_to_list=True)
    return embedding
