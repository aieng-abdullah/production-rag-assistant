"""Vector search using LangChain Chroma integration.

Provides semantic search over document chunks using vector similarity.
"""

from typing import List, Dict, Optional

from loguru import logger

from src.db.chroma_client import get_vectorstore


def vector_search(query: str, top_k: int = 20) -> List[Dict]:
    """Search for similar chunks using vector similarity.

    Args:
        query: The search query text.
        top_k: Number of top results to return.

    Returns:
        List of chunk dictionaries with similarity scores.
        Each dict contains: text, metadata, score
    """
    logger.debug(f"Vector search: '{query[:50]}...' (top_k={top_k})")

    vectorstore = get_vectorstore()

    try:
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
        )

        # Format results to match existing chunk structure
        chunks = []
        for doc, score in results:
            chunks.append({
                "text": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "page_num": doc.metadata.get("page_num", -1),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "score": float(score),
                "source": "vector",
            })

        logger.debug(f"Vector search returned {len(chunks)} results")
        return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise RuntimeError(f"Vector search error: {e}")


def get_retriever(top_k: int = 20):
    """Get a LangChain retriever for use in chains.

    Args:
        top_k: Number of documents to retrieve.

    Returns:
        A LangChain retriever instance.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
