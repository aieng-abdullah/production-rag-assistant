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

### Production-grade Retrieval-Augmented Generation system for research papers

Hybrid Retrieval • Cross-Encoder Reranking • Citation Enforcement • Automated Evaluation

---

## Live Link⚡

[Try the live app](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)

Upload research paper PDFs and ask questions with grounded citations and page references.

</div>

---

# The Problem It Solves

Reading research papers is slow.

Finding exact information across multiple papers is slower.

And standard LLMs hallucinate confidently when responses are not grounded in source material.

This system solves three core RAG problems.

---

> ## Problem 1: Hallucinated Research Facts
>
> LLMs generate unsupported claims from research papers.
>
> ### Solution
>
> Every response is grounded in retrieved chunks with mandatory `[SOURCE N]` citations enforced through Pydantic schema validation.
>
> If citations are missing, the response is rejected at the validation layer.

---

> ## Problem 2: Weak Retrieval Quality
>
> Keyword search misses semantic meaning.
>
> Vector search misses exact terminology.
>
> ### Solution
>
> Hybrid BM25 + Vector Retrieval with Reciprocal Rank Fusion (RRF).
>
> Both retrieval systems run in parallel and ranked results are fused using:
>
> ```text
> score = 1 / (k + rank)
> ```

---

> ## Problem 3: No Reliable Evaluation
>
> Most RAG systems have no measurable quality validation.
>
> ### Solution
>
> Automated Ragas evaluation pipeline with:
>
> - Faithfulness
> - Answer Relevancy
> - Context Recall
>
> GitHub Actions CI blocks merges when evaluation scores drop below threshold.

---

# Architecture

## Document Processing Pipeline

```text
PDF Upload
    ↓
PyMuPDF Parser
    • Page-aware text extraction

    ↓
RecursiveCharacterTextSplitter
    • 350 token chunks
    • 75 overlap

    ↓
HuggingFace Embeddings
    • sentence-transformers/all-MiniLM-L6-v2

    ↓
ChromaDB
    • Vector storage with cosine similarity
```

---

## Retrieval & Generation Pipeline

```text
User Query
     ├── BM25 Search (Top 20)
     ├── Vector Search (Top 20)
     ↓
Reciprocal Rank Fusion (RRF)
     ↓
Cross-Encoder Reranker
     • ms-marco-MiniLM-L-6-v2

     ↓
Top 5 Chunks
     ↓
Citation Prompt Builder
     ↓
Groq LLM (Llama 3.3 70B)
     ↓
Pydantic Citation Validator
     ↓
Final Response with [SOURCE N] Citations
```

---

# Key Technical Decisions

<details>
<summary><b>Why hybrid retrieval instead of vector-only retrieval?</b></summary>

<br>

BM25 excels at exact keyword matching.

This is critical for technical research terminology such as:

- "scaled dot-product attention"
- "BLEU score"
- "LoRA adapters"

Vector retrieval handles semantic similarity.

Reciprocal Rank Fusion combines both retrieval systems without requiring score normalization across retrieval methods.

</details>

---

<details>
<summary><b>Why use cross-encoder reranking?</b></summary>

<br>

Bi-encoders embed queries and chunks independently.

Cross-encoders evaluate the query and chunk together, producing significantly more accurate relevance scoring.

Running cross-encoder inference across all chunks would be computationally expensive.

Instead, reranking is applied only to the top retrieval candidates after RRF fusion.

</details>

---

<details>
<summary><b>Why enforce citations with Pydantic validation?</b></summary>

<br>

Prompt instructions alone are unreliable.

The validator hard-fails if the response does not contain valid `[SOURCE N]` patterns.

This forces grounded responses instead of relying entirely on prompt compliance.

</details>

---

<details>
<summary><b>Why Langfuse observability?</b></summary>

<br>

Production AI systems cannot be debugged effectively using logs alone.

Langfuse traces:

- Retrieval latency
- Prompt construction
- Token usage
- LLM outputs
- Citation validation

Tracing identified that the cross-encoder reranker accounts for approximately 72% of total latency.

</details>

---

# Evaluation Results

Evaluated on 15 question-answer pairs from the *Attention Is All You Need* paper using Ragas metrics with Groq LLM as the judge.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | **0.83** | 0.75 | ![PASS](https://img.shields.io/badge/PASS-success) |
| Answer Relevancy | **0.90** | 0.75 | ![PASS](https://img.shields.io/badge/PASS-success) |
| Context Recall | **1.00** | 0.70 | ![PASS](https://img.shields.io/badge/PASS-success) |

---

### Faithfulness — 0.83

83% of generated claims are grounded in retrieved context.

The system is not fabricating unsupported answers at a significant rate.

---

### Answer Relevancy — 0.90

Responses directly address the user query with minimal irrelevant output.

Highest scoring evaluation metric.

---

### Context Recall — 1.00

The retrieval pipeline successfully retrieved all required information for every evaluation query.

No relevant chunks were missed.

---

### Known Limitation

Context precision is lower due to overlapping academic chunks.

Planned optimizations:

- Reduce chunk size from 350 → 256
- Add section-aware metadata filtering

---

# Observability & Monitoring

Every query is traced end-to-end with Langfuse.

Each trace captures:

| Span | What It Tracks |
|------|---------------|
| retrieval | BM25 + vector + RRF + rerank latency |
| prompt-build | prompt length and construction time |
| llm-call | token usage, model, response time |
| citation-validation | pass/fail status |

### Latency Breakdown (from live traces)

| Component | Latency | % of Total |
|-----------|---------|------------|
| Cross-Encoder Reranker | ~10s | 72% |
| Vector Search | ~0.4s | 3% |
| BM25 Search | ~0.17s | 1% |
| Groq LLM | ~1.2s | 9% |
| Other | ~2s | 15% |

Bottleneck identified through Langfuse traces — not guessing.

# Technology Stack

| Layer | Technology |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25Retriever |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (Llama 3.3 70B) |
| Orchestration | LangChain |
| UI | Streamlit |
| Observability | Langfuse |
| Evaluation | Ragas |
| CI/CD | GitHub Actions |

---

# Local Setup

<details>
<summary><b>Setup Instructions</b></summary>

<br>

```bash
# Clone repository
git clone https://github.com/aieng-abdullah/production-rag-assistant.git

# Move into project
cd production-rag-assistant

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Add GROQ_API_KEY inside .env

# Start application
streamlit run app.py
```

</details>

---

# Running Evaluation

```bash
python3 eval/eval_runner.py
```

Evaluation results are saved to `results.json`.

---

# Running Tests

```bash
pytest tests/ -v
```

---

# Performance Notes

Current end-to-end latency is approximately 14 seconds.

The cross-encoder reranker accounts for roughly 72% of total runtime when executed on CPU.

Potential optimizations:

- GPU deployment
- Lightweight reranker model
- Smaller reranking candidate set

Current implementation prioritizes retrieval quality and grounded answers over raw latency.

---

# Author

## Abdullah Al Arif

JR. AI Engineer

[GitHub](https://github.com/aieng-abdullah) • [LinkedIn](www.linkedin.com/in/abdullah-al-arif-8b58542a7)
