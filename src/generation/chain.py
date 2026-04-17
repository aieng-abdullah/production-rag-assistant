"""Generation chain with LangSmith tracing."""

import time
from typing import List, Dict
from openai import OpenAI
from langsmith import traceable

from src.config import Config
from src.generation.prompts import SYSTEM_PROMPT, format_context
from src.generation.output_models import CitedAnswer
from src.generation.validator import validate_citations, extract_citations
from src.monitoring.logger import get_logger
from src.monitoring.rag_metrics import metrics_collector
from src.eval.live_ragas import live_evaluator

logger = get_logger("generation")

# Initialize OpenAI client
client = OpenAI(api_key=Config.OPENAI_API_KEY)


@traceable(run_type="chain", name="rag_generate")
def generate(question: str, chunks: List[Dict], retrieval_metrics=None, run_ragas: bool = False) -> Dict:
    """Generate cited answer using LLM."""
    gen_start = time.time()
    
    if not chunks:
        return {
            "answer": "I cannot answer this based on the provided documents.",
            "citations": [],
            "sources": [],
            "metrics": None
        }
    
    # Format context
    context = format_context(chunks)
    
    # Build prompt
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    
    logger.info(f"Generating answer for: '{question[:50]}...' with {len(chunks)} chunks")
    
    try:
        # Call LLM
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        raw_answer = response.choices[0].message.content
        
        # Validate citations
        max_index = len(chunks) - 1
        is_valid, _, citation_indices = validate_citations(raw_answer, max_index)
        
        # Collect generation metrics
        gen_metrics = metrics_collector.collect_generation(
            query=question,
            start_time=gen_start,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            citation_count=len(citation_indices),
            chunk_count=len(chunks)
        )
        
        # Log full pipeline metrics if retrieval metrics provided
        full_metrics = None
        if retrieval_metrics:
            full_metrics = metrics_collector.log_full_pipeline(
                query=question,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=gen_metrics
            )
        
        # Run RAGAS evaluation if requested
        ragas_scores = None
        if run_ragas:
            contexts = [c["text"] for c in chunks]
            ragas_scores = live_evaluator.evaluate_query(
                question=question,
                answer=raw_answer,
                contexts=contexts
            )
            quality_pass = live_evaluator.check_quality(ragas_scores)
            ragas_scores["quality_passed"] = quality_pass
        
        # Build result
        result = {
            "answer": raw_answer,
            "citations": citation_indices,
            "sources": [chunks[i] for i in citation_indices if i < len(chunks)],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metrics": {
                "generation": gen_metrics.__dict__ if gen_metrics else None,
                "pipeline": full_metrics,
                "ragas": ragas_scores
            }
        }
        
        logger.info(f"Generated answer with {len(citation_indices)} citations")
        return result
        
    except ValueError as e:
        # Citation validation failed - return error info
        logger.error(f"Citation validation failed: {e}")
        return {
            "answer": raw_answer,
            "citations": extract_citations(raw_answer),
            "sources": [],
            "error": str(e),
            "usage": {},
            "metrics": None
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "answer": "Error generating answer. Please try again.",
            "citations": [],
            "sources": [],
            "error": str(e),
            "usage": {},
            "metrics": None
        }
