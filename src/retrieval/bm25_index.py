"""BM25 search backed by rank_bm25 (Okapi BM25)."""

from collections.abc import Callable

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi


def _default_preprocess(text: str) -> list[str]:
    """Whitespace tokenisation (same default the LangChain retriever used)."""
    return text.split()


class BM25Retriever:
    """Minimal Okapi BM25 retriever over LangChain Documents.

    Drop-in for the slice of the LangChain community BM25Retriever
    this app used: `from_documents(...)`, a mutable `k`, and `invoke(query)`.
    Ranking is identical: BM25Okapi scores, top-k by descending score.
    """

    def __init__(
        self,
        vectorizer: BM25Okapi,
        docs: list[Document],
        k: int = 4,
        preprocess_func: Callable[[str], list[str]] = _default_preprocess,
    ) -> None:
        self.vectorizer = vectorizer
        self.docs = docs
        self.k = k
        self.preprocess_func = preprocess_func

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        *,
        k: int = 4,
        bm25_params: dict | None = None,
        preprocess_func: Callable[[str], list[str]] = _default_preprocess,
    ) -> "BM25Retriever":
        docs = list(documents)
        tokenised = [preprocess_func(doc.page_content) for doc in docs]
        vectorizer = BM25Okapi(tokenised, **(bm25_params or {}))
        return cls(vectorizer, docs, k=k, preprocess_func=preprocess_func)

    def invoke(self, query: str) -> list[Document]:
        scores = self.vectorizer.get_scores(self.preprocess_func(query))
        top = np.argsort(scores)[::-1][: self.k]
        return [self.docs[i] for i in top]


def _chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Convert chunk dictionaries to LangChain Documents.

    Full metadata passes through (provenance keys included -- PR-4b-iv);
    `source: bm25` marks the retrieval channel."""
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "chunk_index": chunk.get("chunk_index", -1),
                "chunk_id": chunk.get(
                    "chunk_id",
                    f"{chunk.get('doc_id', 'unknown')}_chunk_{chunk.get('chunk_index', -1)}",
                ),
                **{key: value for key, value in chunk.items() if key != "text"},
                "source": "bm25",
            },
        )
        for chunk in chunks
    ]


def _documents_to_chunks(documents: list[Document]) -> list[dict]:
    """Convert LangChain documents back to chunk dictionaries."""
    return [
        {
            "text": doc.page_content,
            "doc_id": doc.metadata.get("doc_id", "unknown"),
            "page_num": doc.metadata.get("page_num", -1),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "chunk_id": f"{doc.metadata.get('doc_id', 'unknown')}_chunk_{doc.metadata.get('chunk_index', -1)}",
            **{
                key: value
                for key, value in doc.metadata.items()
                if key != "source"
            },
            "source": "bm25",
        }
        for doc in documents
    ]


def build_bm25_index(chunks: list[dict]) -> BM25Retriever:
    """
    Build a BM25 index from chunks using rank_bm25.

    """
    documents = _chunks_to_documents(chunks)
    retriever = BM25Retriever.from_documents(documents=documents)
    logger.info(f"Built BM25 index with {len(documents)} documents")
    return retriever


def bm25_search(bm25: BM25Retriever, query: str, top_k: int) -> list[dict]:
    """
    Search the BM25 index using the rank_bm25-backed retriever.

    """
    try:
        # Configure k for this search
        bm25.k = top_k

        # Search using our retriever
        documents = bm25.invoke(query)

        # Convert back to chunk format
        results = _documents_to_chunks(documents)

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    except Exception as e:
        raise RuntimeError(f"Error while ranking documents: {e}")


def get_bm25_retriever(chunks: list[dict], top_k: int = 20) -> BM25Retriever:
    """Get a BM25Retriever for use in chains.

    """
    documents = _chunks_to_documents(chunks)
    retriever = BM25Retriever.from_documents(
        documents=documents,
        k=top_k,
    )
    logger.info(f"Created BM25Retriever with {len(documents)} documents")
    return retriever
