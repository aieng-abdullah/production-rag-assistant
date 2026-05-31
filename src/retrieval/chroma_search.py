"""
Description:
         Semantic similarity search using top-k method to return top-k most similar results.
"""

from loguru import logger
from src.ingestion.embedder import embed_query
from src.db.chroma_client import get_collection


def vector_search(query: str, top_k: int) -> list[dict]:
    """
    Semantic similarity search in vector store. Uses embed_query to embed the query.
    """
    embeddings = embed_query(query)

    collection = get_collection()
    try:
        results = collection.query(query_embeddings=[embeddings], n_results=top_k)
        logger.info("Vector search completed")
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise RuntimeError("Error while vector search")

    chunks = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
        "text": text,
        "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
        **metadata
    })
    
        
    return chunks
