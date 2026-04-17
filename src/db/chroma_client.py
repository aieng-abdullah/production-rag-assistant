"""ChromaDB client with environment-based switching."""

import chromadb
from chromadb.config import Settings

from src.config import Config


def get_chroma_client():
    """Get ChromaDB client - PersistentClient locally, HttpClient in prod."""
    # Production mode - use HTTP client
    if Config.CHROMA_HOST:
        return chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False)
        )
    
    # Local development - use persistent client
    Config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(Config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(client, name: str = None):
    """Get or create a ChromaDB collection."""
    name = name or Config.COLLECTION_NAME
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
