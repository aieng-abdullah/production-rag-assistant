"""
ChromaDB client for vector storage and retrieval.
"""

from typing import Optional

import chromadb
from loguru import logger
from src.config import Config

_client: Optional[chromadb.Client] = None


def get_chroma_client() -> chromadb.Client:
    """
    Returns a cached ChromaDB client.
    Creates it on first call, reuses on subsequent calls.
    """
    global _client

    if _client is not None:
        return _client

    if Config.CHROMA_MODE == "local":
        Config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"ChromaDB: local mode → {Config.CHROMA_DIR}")
        _client = chromadb.PersistentClient(path=str(Config.CHROMA_DIR))

    elif Config.CHROMA_MODE == "prod":
        logger.info(f"ChromaDB: prod mode → {Config.CHROMA_HOST}:{Config.CHROMA_PORT}")
        _client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
        )

    else:
        raise ValueError(
            f"Invalid CHROMA_MODE: '{Config.CHROMA_MODE}'. "
            f"Must be 'local' or 'prod'."
        )

    return _client


def get_collection() -> chromadb.Collection:
    """
    Returns the main document collection.
    """
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
        },
    )

    logger.debug(
        f"Collection '{Config.COLLECTION_NAME}' ready — {collection.count()} docs"
    )
    return collection


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Extracts IDs, embeddings, and text from a list of chunks and upserts them
    into the specified collection.
    """
    collection = get_collection()

    # extracting data from
    ids = [chunk["chunk_id"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(chunks)


def reset_client():
    """
    Resets the singleton client (useful for tests).
    """
    global _client
    _client = None
