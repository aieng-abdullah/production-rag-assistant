<div align="center">

<img width="300" height="300" alt="artificial-intelligence" src="https://github.com/user-attachments/assets/b92417d0-a09f-4353-883b-d6f545e727e8" />

# Production RAG Research Assistant

[![CI](https://github.com/aieng-abdullah/production-rag-assistant/actions/workflows/eval.yml/badge.svg)](https://github.com/aieng-abdullah/production-rag-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red?logo=streamlit)](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Anthropic%20%7C%20OpenAI-orange)](https://groq.com)
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

Academic researchers, engineers, and students read dozens of papers
to find specific answers — and still leave unsure if they understood
correctly.

Asking a standard LLM is worse. It gives confident, well-written
answers that may have nothing to do with what the paper actually says.
You can't cite a hallucination.

**This system lets you query research papers and trust the answers.**

Upload one or more PDFs. Ask a specific question. Every response
comes with exact `[SOURCE N]` citations and page references — grounded
in what the document actually contains, not what the model assumes.

If the source doesn't support the answer, the response is rejected
before it reaches you.

---

### Who it's built for

| If you are... | This solves... |
|---|---|
| A researcher | Cross-paper synthesis without manual skimming |
| An engineer | Extracting implementation details from technical papers |
| A student | Citeable answers you can actually reference in writing |

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
LLM Provider Chain (with retry + exponential backoff)
     ├── Groq (Llama 3.3 70B) — free
     ├── Anthropic (Claude) — optional, user-provided key
     └── OpenAI (GPT-4o) — optional, user-provided key
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

Evaluated on 5 question-answer pairs from the *Attention Is All You Need* paper using Ragas metrics with Groq LLM as the judge.

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

All requests are traced end-to-end using Langfuse.

141 complete request traces collected across real usage.

---

### Latency Profile

Based on 141 traced requests.

| Metric | Latency | What It Means |
|--------|---------|---------------|
| p50 | 1.54s | Most users experience this |
| p90 | 7.32s | 1 in 10 users waits this long |
| p95 | 11.09s | 1 in 20 users waits this long |
| p99 | 14.06s | Worst case observed |

Median experience is 1.54 seconds.

Variance is driven by the cross-encoder reranker, not the LLM.

---

### Component Latency Breakdown

| Component | p50 | p90 | p95 | p99 | Role |
|-----------|-----|-----|-----|-----|------|
| Full Request | 1.54s | 7.32s | 11.09s | 14.06s | End-to-end |
| Retrieval | 0.77s | 6.11s | 10.01s | 12.19s | BM25 + vector + RRF |
| Rerank | 0.74s | 6.03s | 9.23s | 11.68s | Cross-encoder scoring |
| ChatGroq (LLM) | 0.57s | 0.93s | 1.19s | 2.52s | Generation |
| Vector Search | 0.03s | 0.06s | 0.22s | 0.38s | Embedding lookup |

---

### Key Finding: LLM Is Not the Bottleneck

Common assumption: LLM generation drives latency.

What the data shows: LLM contributes only 0.57s at median.

The cross-encoder reranker is the actual bottleneck.

| Component | p50 | p95 | Variance Ratio |
|-----------|-----|-----|----------------|
| Rerank | 0.74s | 9.23s | 12x |
| Retrieval | 0.77s | 10.01s | 13x |
| ChatGroq | 0.57s | 1.19s | 2x |
| Vector Search | 0.03s | 0.22s | 7x |

Groq inference is fast and stable — 2x variance between p50 and p95.

The reranker shows 12x variance because the cross-encoder scores every (query, chunk) pair individually on CPU.

Latency scales with document density and query length.

This finding would have been invisible without instrumentation.

Optimizing the LLM — the intuitive target — would have had near zero impact.

---

### Why the Reranker Has High Variance

The cross-encoder reranker scores each (query, chunk) pair individually.

With 20 candidate chunks per query, total compute scales with:

- Query token length
- Chunk token length
- Number of candidates

Short query + sparse PDF  → ~0.74s

Long query + dense PDF    → ~9.23s

This is a CPU-bound operation with no batching optimization in the current implementation.

---

### Model Usage

| Model | Tokens Used | Purpose |
|-------|-------------|---------|
| llama-3.3-70b-versatile | 96,490 | Primary generation |

Average tokens per trace: ~684 tokens

---

### Trace Structure

Every request is broken into spans:

| Span | What It Tracks |
|------|---------------|
| retrieval | BM25 + vector + RRF + rerank latency |
| prompt-build | prompt length and construction time |
| llm-call | token usage, model, response time |
| citation-validation | pass/fail status |

Bottleneck identified through Langfuse traces — not guessing.

---

# LLM Provider Failover

The system supports **3 LLM providers** with automatic retry, exponential backoff, and failover:

| Provider | Model | Cost | Setup |
|----------|-------|------|-------|
| **Groq** | Llama 3.3 70B Versatile | Free | Set `GROQ_API_KEY` in `.env` |
| **Anthropic** | Claude Sonnet 4 | Pay-per-use | Add key via sidebar or set `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o | Pay-per-use | Add key via sidebar or set `OPENAI_API_KEY` |

### How It Works

1. **Retry**: Each provider gets up to 3 attempts with exponential backoff (1s → 2s → 4s)
2. **Failover**: If a provider fails after retries, the next provider in the chain is tried
3. **Chain order**: Groq → Anthropic → OpenAI (only providers with API keys are included)

### Adding Your Own Provider

No code changes needed. Two options:

**Option A — Sidebar UI (recommended):**
Open the app → expand "LLM Providers" in the sidebar → enter your API key and model name.

**Option B — Environment variables:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

If only Groq is configured (default), the system works exactly as before — just with retry resilience.

---

# Technology Stack

| Layer | Technology |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25Retriever |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq / Anthropic / OpenAI (with retry + failover) |
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

End-to-end latency ranges from 1.54s (p50) to 14.06s (p99) across 141 traced requests.

The cross-encoder reranker accounts for 72% of total runtime on CPU with 12x variance between p50 and p95.

Current implementation prioritizes retrieval quality and grounded answers over raw latency.

---

# Author

## Abdullah Al Arif

JR. AI Engineer

[GitHub](https://github.com/aieng-abdullah) • [LinkedIn](www.linkedin.com/in/abdullah-al-arif-8b58542a7)
