"""RAG pipeline metrics tracking."""

import time
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from src.monitoring.logger import get_logger

logger = get_logger("rag_metrics")


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval phase."""
    query: str
    latency_ms: float
    vector_hits: int
    bm25_hits: int
    fused_count: int
    reranked_count: int
    doc_filter: str = None


@dataclass
class GenerationMetrics:
    """Metrics for generation phase."""
    query: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    citation_count: int
    chunk_count: int


@dataclass
class RAGMetrics:
    """Complete RAG pipeline metrics."""
    query_id: str
    query: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    total_latency_ms: float
    timestamp: str


class RAGMetricsCollector:
    """Collect and log RAG pipeline metrics."""
    
    def __init__(self):
        """Initialize collector."""
        self._query_count = 0
    
    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def collect_retrieval(
        self,
        query: str,
        start_time: float,
        vector_hits: int,
        bm25_hits: int,
        fused_count: int,
        reranked_count: int,
        doc_filter: str = None
    ) -> RetrievalMetrics:
        """Collect retrieval metrics."""
        latency = (time.time() - start_time) * 1000
        
        metrics = RetrievalMetrics(
            query=query[:100],
            latency_ms=round(latency, 2),
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            fused_count=fused_count,
            reranked_count=reranked_count,
            doc_filter=doc_filter
        )
        
        logger.info(f"Retrieval: {metrics.latency_ms}ms | Vector: {vector_hits} | BM25: {bm25_hits} | Rerank: {reranked_count}")
        
        return metrics
    
    def collect_generation(
        self,
        query: str,
        start_time: float,
        prompt_tokens: int,
        completion_tokens: int,
        citation_count: int,
        chunk_count: int
    ) -> GenerationMetrics:
        """Collect generation metrics."""
        latency = (time.time() - start_time) * 1000
        
        metrics = GenerationMetrics(
            query=query[:100],
            latency_ms=round(latency, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            citation_count=citation_count,
            chunk_count=chunk_count
        )
        
        logger.info(f"Generation: {metrics.latency_ms}ms | Tokens: {metrics.total_tokens} | Citations: {citation_count}")
        
        return metrics
    
    def log_full_pipeline(
        self,
        query: str,
        retrieval_metrics: RetrievalMetrics,
        generation_metrics: GenerationMetrics
    ) -> Dict[str, Any]:
        """Log complete RAG pipeline metrics."""
        import datetime
        
        query_id = self._generate_query_id()
        self._query_count += 1
        
        total_latency = retrieval_metrics.latency_ms + generation_metrics.latency_ms
        
        full_metrics = {
            "query_id": query_id,
            "query": query[:100],
            "retrieval_latency_ms": retrieval_metrics.latency_ms,
            "generation_latency_ms": generation_metrics.latency_ms,
            "total_latency_ms": round(total_latency, 2),
            "vector_hits": retrieval_metrics.vector_hits,
            "bm25_hits": retrieval_metrics.bm25_hits,
            "chunks_used": generation_metrics.chunk_count,
            "tokens_total": generation_metrics.total_tokens,
            "citations": generation_metrics.citation_count,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        logger.info(f"RAG Pipeline [{query_id}]: {round(total_latency, 2)}ms total")
        
        return full_metrics


# Global collector instance
metrics_collector = RAGMetricsCollector()
