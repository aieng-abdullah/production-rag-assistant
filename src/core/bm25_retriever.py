"""BM25Retriever class for BM25 keyword search."""

from typing import List, Dict

from langchain_community.retrievers import BM25Retriever as BM25RetrieverLC
from langchain_core.documents import Document
from loguru import logger


class BM25Retriever:
    """BM25 keyword search with LangChain integration.
    
    This class encapsulates BM25 index building and search operations,
    providing a clean interface for keyword-based retrieval.
    """
    
    def __init__(self, top_k: int = 20):
        """Initialize the BM25Retriever.
        
        Args:
            top_k: Default number of results to return.
        """
        self.top_k = top_k
        self._retriever: Optional[BM25RetrieverLC] = None
    
    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata.
        """
        documents = self._chunks_to_documents(chunks)
        self._retriever = BM25RetrieverLC.from_documents(
            documents=documents,
            k=self.top_k
        )
        logger.info(f"Built BM25 index with {len(documents)} documents")
    
    def search(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Search the BM25 index.
        
        Args:
            query: Search query string.
            chunks: List of chunks to search (used if index not built).
            
        Returns:
            List of chunk dictionaries with search results.
        """
        if self._retriever is None:
            self.build_index(chunks)
        
        self._retriever.k = self.top_k
        documents = self._retriever.invoke(query)
        results = self._documents_to_chunks(documents)
        
        logger.debug(f"BM25 search returned {len(results)} results")
        return results
    
    @staticmethod
    def _chunks_to_documents(chunks: List[Dict]) -> List[Document]:
        """Convert chunk dictionaries to LangChain Documents.
        
        Args:
            chunks: List of chunk dictionaries.
            
        Returns:
            List of LangChain Document objects.
        """
        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "page_num": chunk.get("page_num", -1),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "chunk_id": chunk.get("chunk_id", f"{chunk.get('doc_id')}_chunk_{chunk.get('chunk_index')}"),
                    "source": "bm25",
                }
            )
            for chunk in chunks
        ]
    
    @staticmethod
    def _documents_to_chunks(documents: List[Document]) -> List[Dict]:
        """Convert LangChain Documents back to chunk dictionaries.
        
        Args:
            documents: List of LangChain Document objects.
            
        Returns:
            List of chunk dictionaries.
        """
        return [
            {
                "text": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "page_num": doc.metadata.get("page_num", -1),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "chunk_id": f"{doc.metadata.get('doc_id', 'unknown')}_chunk_{doc.metadata.get('chunk_index', -1)}",
                "source": "bm25",
            }
            for doc in documents
        ]
