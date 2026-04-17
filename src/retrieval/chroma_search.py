"""Vector similarity search with ChromaDB."""

from typing import List, Tuple

from src.db.chroma_client import get_chroma_client, get_or_create_collection
from src.ingestion.embedder import embed_query
from src.monitoring.logger import get_logger

logger = get_logger("chroma_search")


def vector_search(query: str, top_k: int = 20, doc_id: str = None) -> List[Tuple[str, float]]:
    """Search ChromaDB with vector similarity."""
    # Embed query
    query_embedding = embed_query(query)
    
    # Get collection
    collection = get_or_create_collection(get_chroma_client())
    
    # Build filter
    where_filter = {"doc_id": doc_id} if doc_id else None
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["distances"]
    )
    
    # Extract IDs and convert distance to similarity score
    chunk_ids = results["ids"][0] if results["ids"] else []
    distances = results["distances"][0] if results["distances"] else []
    
    # Convert cosine distance to similarity (1 - distance)
    scores = [1 - d for d in distances]
    
    logger.debug(f"Vector search: {len(chunk_ids)} results")
    return list(zip(chunk_ids, scores))
