"""VectorStore class for managing ChromaDB operations."""

from typing import Optional, List, Dict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from src.config import Config
from src.core.embedding_model import EmbeddingModel


class VectorStore:
    """Manages ChromaDB operations with proper lifecycle management.
    
    This class encapsulates all ChromaDB operations including:
    - Adding/upserting documents
    - Semantic similarity search
    - Loading all chunks
    - Proper resource cleanup
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: EmbeddingModel = None,
    ):
        """Initialize the VectorStore.
        
        Args:
            collection_name: Name of the ChromaDB collection.
                          Defaults to Config.COLLECTION_NAME.
            persist_directory: Directory to persist ChromaDB data.
                            Defaults to Config.CHROMA_DIR.
            embedding_model: EmbeddingModel instance for embeddings.
                           If None, creates a new instance.
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = persist_directory or str(Config.CHROMA_DIR)
        self.embedding_model = embedding_model or EmbeddingModel()
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load vectorstore on first access.
        
        Returns:
            The Chroma vectorstore instance.
        """
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model.model,
                persist_directory=self.persist_directory,
            )
            logger.info(f"ChromaDB initialized: {self.collection_name}")
        return self._vectorstore
    
    def upsert_chunks(self, chunks: List[Dict]) -> int:
        """Add or update chunks in the vectorstore.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata.
            
        Returns:
            Number of chunks upserted.
        """
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
        
        self.vectorstore.add_documents(documents=documents, ids=ids)
        logger.info(f"Upserted {len(chunks)} chunks to vectorstore")
        return len(chunks)
    
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Semantic similarity search using embeddings.
        
        Args:
            query_embedding: Embedding vector for the query.
            top_k: Number of results to return.
            
        Returns:
            List of chunk dictionaries with metadata.
        """
        results = self.vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        chunks = []
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text": text,
                "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
                **metadata
            })
        
        logger.debug(f"Vector search returned {len(chunks)} results")
        return chunks
    
    def load_all(self) -> List[Dict]:
        """Load all chunks from the vectorstore.
        
        Returns:
            List of all chunk dictionaries with metadata.
        """
        results = self.vectorstore._collection.get()
        chunks = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text": text,
                "chunk_id": f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}",
                **metadata
            })
        return chunks
    
    def reset(self):
        """Clear vectorstore from memory (useful for testing)."""
        self._vectorstore = None
        logger.debug("VectorStore reset")
