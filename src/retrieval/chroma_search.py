"""
Description:
         semantic similarity search using top=k methoad to return top-k most similar results.
"""

from typing import List
from loguru import logger
from src.ingestion.embedder import embed_query, _get_model
from src.db.chroma_client import get_collection


def vector_search(query: str, top_k: int) -> list[dict]:
    """
    semantic similrity search in vector stroe .use embed_query function to do embeddin
    """
    embeddings = embed_query(query)

    # search in tha vector stre
    collection = get_collection()
    try:

        results = collection.query(query_embeddings=[embeddings], n_results=top_k)
        logger.info("vector search is done ")

    except Exception as e:
        logger.error("there is an errro while vextor search")
        raise RuntimeError("Error while vector search")

    chunks = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, **metadata})
    return chunks
