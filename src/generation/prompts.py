"""RAG prompts using LangChain ChatPromptTemplate.

Provides structured prompts for the generation phase of the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


# RAG answer generation prompt with citation enforcement
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant that answers questions based on provided documents.

Instructions:
- Answer using ONLY the information from the provided context
- Cite your sources using [SOURCE N] format where N corresponds to the document number
- If the context doesn't contain enough information, say "I don't have enough information to answer this"
- Be concise but thorough
- Always include at least one citation for every claim you make

Context:
{context}"""),
    ("human", "{question}"),
])


# Prompt for validating citations
CITATION_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a citation validator. Your job is to check if all claims in an answer are properly supported by the provided context.

Check:
1. Every claim has a corresponding [SOURCE N] citation
2. The cited source actually supports the claim
3. No hallucinated information outside the context

Respond with:
- VALID if all citations are correct
- INVALID: [explanation] if there are issues"""),
    ("human", """Context: {context}

Answer to validate: {answer}"""),
])


# Prompt for re-ranking context chunks
RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a relevance scorer. Rate how relevant each document is to the question on a scale of 0-10."),
    ("human", """Question: {question}

Documents:
{documents}

Provide scores as a comma-separated list."""),
])


def format_context(chunks: list[dict]) -> str:
    """Format chunks into context string for prompts.

    Args:
        chunks: List of chunk dictionaries with 'text' and metadata.

    Returns:
        Formatted context string with numbered sources.
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[SOURCE {i}]"
        page_info = f" (Page {chunk.get('page_num', '?')})" if "page_num" in chunk else ""
        doc_id = f" [{chunk.get('doc_id', 'unknown')}]"
        text = chunk.get("text", "")

        formatted.append(f"{source}{page_info}{doc_id}\n{text}\n")

    return "\n---\n\n".join(formatted)
