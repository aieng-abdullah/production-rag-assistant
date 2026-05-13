<div align="center">

<img width="300" height="300" alt="artificial-intelligence" src="https://github.com/user-attachments/assets/b92417d0-a09f-4353-883b-d6f545e727e8" />

  
# Production RAG Research Assistant

[![CI](https://github.com/aieng-abdullah/production-rag-assistant/actions/workflows/eval.yml/badge.svg)](https://github.com/aieng-abdullah/production-rag-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red?logo=streamlit)](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com)
[![Langfuse](https://img.shields.io/badge/Observability-Langfuse-purple)](https://langfuse.com)
[![Ragas](https://img.shields.io/badge/Evaluated-Ragas-blue)](https://ragas.io)

A production-grade Retrieval-Augmented Generation system for research papers with hybrid retrieval, cross-encoder reranking, citation enforcement, and automated evaluation.

---

## 🚀 Live Link

[Try the live app](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)

Upload any research paper PDF and ask questions. Every answer is cited with the exact source and page number.

---
</div>
## The Problem It Solves

Reading research papers is slow. Finding specific answers across multiple papers is slower. And when you ask an LLM without grounding, it hallucinates confidently.

This system solves three real problems.

**Problem 1**: LLMs hallucinate facts from research papers.
**Solution**: Every answer is grounded in retrieved chunks with mandatory [SOURCE N] citations enforced by Pydantic schema validation. If the LLM skips citations, the response is rejected at the validation layer.

**Problem 2**: Keyword search misses semantic meaning. Vector search misses exact terms.
**Solution**: Hybrid BM25 + Vector search with Reciprocal Rank Fusion combines both retrieval methods. The system always runs both and fuses the ranked results using the RRF formula: score = 1/(k + rank).

**Problem 3**: No way to know if your RAG system is actually working.
**Solution**: Automated Ragas evaluation pipeline with faithfulness, answer relevancy, and context recall metrics. GitHub Actions CI blocks merges when evaluation scores drop below threshold.

---

## Architecture

```
PDF Upload
    |
PyMuPDF Parser        -- page-aware text extraction
    |
RecursiveCharacterTextSplitter -- 350 token chunks, 75 overlap
    |
HuggingFace Embeddings -- sentence-transformers/all-MiniLM-L6-v2
    |
ChromaDB              -- vector storage with cosine similarity

User Query
    |
BM25 Search (top 20) --|
                        |-- RRF Fusion (k=60) -- Cross-Encoder Reranker
Vector Search (top 20)-|                         ms-marco-MiniLM-L-6-v2
                                                        |
                                                  Top 5 Chunks
                                                        |
                                            Citation Prompt Builder
                                                        |
                                            Groq LLM (Llama 3.3 70B)
                                                        |
                                            Pydantic CitedAnswer Validator
                                                        |
                                            Answer with [SOURCE N] Citations
```

---

## Key Technical Decisions

**Why hybrid retrieval over vector-only?**
BM25 excels at exact term matching, critical for technical papers with specific terminology like "scaled dot-product attention" or "BLEU score". Vector search handles semantic similarity. RRF fusion combines both without needing to normalize different scoring scales.

**Why cross-encoder reranking?**
Bi-encoders embed query and chunks independently. Cross-encoders read both together, producing more accurate relevance scores. Running the cross-encoder on all 1000+ chunks would be too slow, so it runs only on the top 20 results from RRF fusion.

**Why citation enforcement with Pydantic?**
Prompt instructions alone are not reliable. A Pydantic validator hard-fails if the LLM response contains no [SOURCE N] pattern. This is not a warning — it is a hard rejection that forces the system to be honest about its sources.

**Why Langfuse observability?**
Once deployed, you cannot debug a live AI system by reading logs. Langfuse traces every query end-to-end: retrieval latency, prompt content, LLM response, token usage, and citation validation status. The cross-encoder currently accounts for 72% of total latency — identified through traces, not guessing.

---

## Evaluation Results

Evaluated on 15 question-answer pairs from the "Attention Is All You Need" paper using Ragas metrics with Groq LLM as the judge.

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.83 | 0.75 | PASS |
| Answer Relevancy | 0.90 | 0.75 | PASS |
| Context Recall | 1.00 | 0.70 | PASS |

**Faithfulness 0.83** — 83% of claims in generated answers are grounded in retrieved sources. The system is not hallucinating.

**Answer Relevancy 0.90** — answers directly address the questions asked. Highest scoring metric.

**Context Recall 1.00** — the retrieval pipeline found all information needed to answer every question. No relevant chunks were missed.

Known limitation: context precision is lower due to dense academic text producing overlapping chunks. Planned fix: reduce chunk size from 350 to 256 tokens with section-aware metadata filtering.

---

## Stack

| Component | Technology |
|-----------|------------|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| BM25 | LangChain BM25Retriever |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (Llama 3.3 70B) |
| Orchestration | LangChain |
| UI | Streamlit |
| Observability | Langfuse |
| Evaluation | Ragas |
| CI | GitHub Actions |

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/aieng-abdullah/production-rag-assistant.git
cd production-rag-assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 5. Run the app
streamlit run app.py
```

---

## Running Evaluation

```bash
python3 eval/eval_runner.py
```

Results are saved to `results.json`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Performance Notes

Current end-to-end latency is approximately 14 seconds. The cross-encoder reranker accounts for 72% of this running on CPU. GPU deployment or a lighter reranker model would reduce latency to under 2 seconds. This is a deliberate trade-off — accuracy over speed for the portfolio demo.

---

## Author

Abdullah Al Arif — AI Engineer

[GitHub](https://github.com/aieng-abdullah) | [LinkedIn](https://linkedin.com/in/aieng-abdullah)
