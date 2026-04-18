"""Langfuse integration for open-source observability."""

from typing import Optional, Dict, Any
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

from src.config import Config
from src.monitoring.logger import get_logger

logger = get_logger("langfuse")

# Initialize Langfuse client
langfuse = None
if Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY:
    langfuse = Langfuse(
        public_key=Config.LANGFUSE_PUBLIC_KEY,
        secret_key=Config.LANGFUSE_SECRET_KEY,
        host=Config.LANGFUSE_HOST
    )
    logger.info("Langfuse initialized successfully")
else:
    logger.info("Langfuse not configured - skipping tracing")


def is_enabled() -> bool:
    """Check if Langfuse is properly configured."""
    return langfuse is not None


def trace_retrieval(query: str, doc_filter: Optional[str] = None):
    """Decorator to trace retrieval pipeline."""
    def wrapper(func):
        if not is_enabled():
            return func
        
        @observe(name="retrieval", as_type="retrieval")
        def traced(*args, **kwargs):
            # Update context with retrieval params
            langfuse_context.update_current_observation(
                input=query,
                metadata={
                    "doc_filter": doc_filter,
                    "vector_k": Config.TOP_K_VECTOR,
                    "bm25_k": Config.TOP_K_BM25,
                    "rerank_k": Config.TOP_K_RERANK
                }
            )
            
            result = func(*args, **kwargs)
            
            # Log output metrics
            if isinstance(result, tuple) and len(result) >= 1:
                chunks = result[0]
                langfuse_context.update_current_observation(
                    output={"retrieved_chunks": len(chunks)},
                    usage={"total": len(chunks)}
                )
            
            return result
        
        return traced
    return wrapper


def trace_generation(question: str, chunk_count: int):
    """Decorator to trace generation pipeline."""
    def wrapper(func):
        if not is_enabled():
            return func
        
        @observe(name="generation", as_type="generation")
        def traced(*args, **kwargs):
            # Update context with generation params
            langfuse_context.update_current_observation(
                input=question,
                metadata={
                    "chunk_count": chunk_count,
                    "model": Config.LLM_MODEL,
                    "temperature": 0.3
                }
            )
            
            result = func(*args, **kwargs)
            
            # Log output and token usage
            if isinstance(result, dict):
                langfuse_context.update_current_observation(
                    output=result.get("answer", "")[:500],
                    metadata={
                        "citations": result.get("citations", []),
                        "citation_count": len(result.get("citations", []))
                    }
                )
                
                # Token usage if available
                usage = result.get("usage", {})
                if usage:
                    langfuse_context.update_current_observation(
                        usage={
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
                    )
            
            return result
        
        return traced
    return wrapper


def trace_ingestion(filename: str):
    """Decorator to trace ingestion pipeline."""
    def wrapper(func):
        if not is_enabled():
            return func
        
        @observe(name="ingestion", as_type="span")
        def traced(*args, **kwargs):
            langfuse_context.update_current_observation(
                input=filename,
                metadata={
                    "chunk_size": Config.CHUNK_SIZE,
                    "chunk_overlap": Config.CHUNK_OVERLAP
                }
            )
            
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                langfuse_context.update_current_observation(
                    output=result,
                    metadata={
                        "doc_id": result.get("doc_id"),
                        "chunk_count": result.get("chunk_count"),
                        "page_count": result.get("page_count")
                    }
                )
            
            return result
        
        return traced
    return wrapper


def score_metric(name: str, value: float, comment: Optional[str] = None):
    """Log a metric score to Langfuse."""
    if not is_enabled():
        return
    
    try:
        langfuse.score(
            name=name,
            value=value,
            comment=comment
        )
        logger.debug(f"Score logged: {name}={value}")
    except Exception as e:
        logger.error(f"Failed to log score: {e}")


class LangfuseSpan:
    """Context manager for manual span tracing."""
    
    def __init__(self, name: str, metadata: Dict[str, Any] = None):
        self.name = name
        self.metadata = metadata or {}
        self.span = None
    
    def __enter__(self):
        if is_enabled():
            self.span = langfuse.span(name=self.name, metadata=self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.update(level="ERROR", status_message=str(exc_val))
            else:
                self.span.update(level="DEFAULT")
        return False
    
    def update(self, metadata: Dict[str, Any]):
        """Update span metadata."""
        if self.span:
            self.span.update(metadata=metadata)
