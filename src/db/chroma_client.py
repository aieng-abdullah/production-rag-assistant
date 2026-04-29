"""
ChromaDB client using pure LangChain Chroma integration.
"""

from typing import Optional, List, Dict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from src.config import Config
from src.ingestion.embedder import _get_model as get_embedding_model

# Lazy-loaded vectorstore instance
_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    """Get or initialize the LangChain Chroma vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        embedding_model = get_embedding_model()
        _vectorstore = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(Config.CHROMA_DIR),
        )
        logger.info(f"LangChain Chroma initialized: {Config.COLLECTION_NAME}")
    return _vectorstore


def get_collection():
    """
    Returns the underlying Chroma collection for compatibility.
    Accesses the internal _collection from LangChain Chroma.
    """
    vectorstore = _get_vectorstore()
    # Access internal collection for backward compatibility
    return vectorstore._collection


def upsert_chunks(chunks: List[Dict]) -> int:
    """
    Add or update chunks in the vectorstore using LangChain add_documents.
    """
    vectorstore = _get_vectorstore()

    # Convert chunks to LangChain Documents
    documents = []
    ids = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "chunk_index": chunk.get("chunk_index", -1),
            }
        )
        documents.append(doc)
        ids.append(chunk.get("chunk_id", f"chunk_{len(ids)}"))

    vectorstore.add_documents(documents=documents, ids=ids)
    logger.info(f"Upserted {len(chunks)} chunks to vectorstore")
    return len(chunks)


def load_all_chunks() -> List[Dict]:
    """
    Load all chunks from the vectorstore.
    Uses the underlying collection for efficient retrieval.
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection

    results = collection.get()
    chunks = []
    for text, metadata in zip(results["documents"], results["metadatas"]):
        chunks.append({
        "text": text,
        "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
        **metadata
    })
    return chunks


def reset_client():
    """
    Resets the singleton vectorstore (useful for tests).
    """
    global _vectorstore
    _vectorstore = None
    logger.debug("Vectorstore reset")


def get_vectorstore() -> Chroma:
    """Get the LangChain Chroma vectorstore for use in chains.

    Returns:
        Chroma vectorstore instance.
    """
    return _get_vectorstore()
