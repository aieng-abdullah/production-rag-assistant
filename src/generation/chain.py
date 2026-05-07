"""
file: chain.py
This file do retrive top k chunk and generate citation,call llm and genarate citeted answer
"""
from loguru import logger
from langchain_groq import ChatGroq
from src.retrieval.pipeline import retrieval
from src.generation.Citation_system import build_citation_prompt
from src.generation.Citation_system import CitedAnswer
from src.generation.Citation_system import Source
from src.config import Config


def generate(query: str, chunks: list[dict], bm25_index) -> CitedAnswer:
    """
    genarate citation and answer with llm
    """
    #retrive top k chunks
    try:
        top_chunks=retrieval(query, chunks, bm25_index)
        logger.debug(f"Retrieved {len(top_chunks)} top chunks")
    except Exception as e:
        logger.error(f"Error while retrieving top chunks: {e}")
        raise RuntimeError(f"Failed to retrieve top chunks: {e}")
    
    # Generate citation prompt
    try:
        citation_prompt = build_citation_prompt(query, top_chunks)
        logger.debug(f"Generated citation prompt: {citation_prompt}")
    except Exception as e:
        logger.error(f"Error while generating citation prompt: {e}")
        raise RuntimeError(f"Failed to generate citation prompt: {e}")
    
    #LLm calling
    
    client=ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
    )
    response=client.invoke(citation_prompt)
    answer_text = response.content
    
    sources = [
    Source(
        doc_id=chunk["doc_id"],
        page_num=chunk["page_num"],
        text=chunk["text"]
    )
    for chunk in top_chunks
        ]
    return CitedAnswer(answer=answer_text, sources=sources)
    
    
    
    
    

