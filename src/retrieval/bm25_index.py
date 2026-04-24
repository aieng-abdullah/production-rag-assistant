"""BM25 search using pure LangChain BM25Retriever."""

from typing import List, Dict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger


def _chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Convert chunk dictionaries to LangChain Documents."""
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "chunk_index": chunk.get("chunk_index", -1),
                "source": "bm25",
            }
        )
        for chunk in chunks
    ]


def _documents_to_chunks(documents: List[Document]) -> List[Dict]:
    """Convert LangChain Documents back to chunk dictionaries."""
    return [
        {
            "text": doc.page_content,
            "doc_id": doc.metadata.get("doc_id", "unknown"),
            "page_num": doc.metadata.get("page_num", -1),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "source": "bm25",
        }
        for doc in documents
    ]


def build_bm25_index(chunks: List[Dict]) -> BM25Retriever:
    """
    Build a BM25 index from chunks using LangChain BM25Retriever.

    Args:
        chunks: List of chunk dictionaries with 'text' and metadata.

    Returns:
        BM25Retriever instance ready for searching.
    """
    documents = _chunks_to_documents(chunks)
    retriever = BM25Retriever.from_documents(documents=documents)
    logger.info(f"Built BM25 index with {len(documents)} documents")
    return retriever


def bm25_search(bm25: BM25Retriever, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
    """
    Search the BM25 index using LangChain retriever.

    Args:
        bm25: BM25Retriever instance (from build_bm25_index).
        query: Search query string.
        chunks: Original chunks (kept for interface compatibility).
        top_k: Number of results to return.

    Returns:
        List of chunk dictionaries matching the query.
    """
    try:
        # Configure k for this search
        bm25.k = top_k

        # Search using LangChain retriever
        documents = bm25.invoke(query)

        # Convert back to chunk format
        results = _documents_to_chunks(documents)

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    except Exception as e:
        raise RuntimeError(f"Error while ranking documents: {e}")


def get_bm25_retriever(chunks: List[Dict], top_k: int = 20) -> BM25Retriever:
    """Get a BM25Retriever for use in chains.

    Args:
        chunks: List of chunk dictionaries with 'text' and metadata.
        top_k: Number of documents to retrieve.

    Returns:
        BM25Retriever instance.
    """
    documents = _chunks_to_documents(chunks)
    retriever = BM25Retriever.from_documents(
        documents=documents,
        k=top_k,
    )
    logger.info(f"Created BM25Retriever with {len(documents)} documents")
    return retriever
