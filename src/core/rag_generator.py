"""RAGGenerator class for RAG answer generation with citations."""

from typing import List, Dict

from langchain_groq import ChatGroq
from loguru import logger

from src.config import Config
from src.generation.Citation_system import build_citation_prompt, CitedAnswer, Source
from src.core.retrieval_pipeline import RetrievalPipeline


class RAGGenerator:
    """RAG answer generation with citations.
    
    This class orchestrates the full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build citation prompt
    3. Generate answer with LLM
    4. Validate and return cited answer
    """
    
    def __init__(self, llm: ChatGroq, retrieval_pipeline: RetrievalPipeline):
        """Initialize the RAGGenerator.
        
        Args:
            llm: ChatGroq LLM instance.
            retrieval_pipeline: RetrievalPipeline instance for chunk retrieval.
        """
        self.llm = llm
        self.retrieval_pipeline = retrieval_pipeline
    
    def generate(self, query: str, chunks: List[Dict]) -> CitedAnswer:
        """Generate answer with citations.
        
        Args:
            query: User query string.
            chunks: List of all chunks to search from.
            
        Returns:
            CitedAnswer with answer text and sources.
        """
        # Retrieve top chunks
        top_chunks = self.retrieval_pipeline.retrieve(query, chunks)
        logger.info(f"Retrieved {len(top_chunks)} chunks for query")
        
        # Build citation prompt
        citation_prompt = build_citation_prompt(query, top_chunks)
        
        # Generate answer
        response = self.llm.invoke(citation_prompt)
        answer_text = response.content
        logger.info(f"Generated answer: {len(answer_text)} characters")
        
        # Build sources
        sources = [
            Source(
                doc_id=chunk["doc_id"],
                page_num=chunk["page_num"],
                text=chunk["text"],
            )
            for chunk in top_chunks
        ]
        
        return CitedAnswer(answer=answer_text, sources=sources)
