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

### Ask research papers questions. Every sentence cites its source. Zero hallucination by design  enforced by schema validation, not prompts.

Built for grad students and researchers drowning in arXiv PDFs who need trustworthy answers with page-level provenance  not confident-sounding guesses.

**[Try the live demo →](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)** · Upload a research paper PDF and ask questions with grounded citations.

---

### Why this isn't another RAG demo

| Differentiator | What most demos do | What this does |
|---|---|---|
| **Citation enforcement** | Prompt says "cite sources" | Pydantic validates **every sentence** for `[SOURCE N]`; violations rejected at the validation layer |
| **Retrieval** | Vector-only | BM25 ∥ vector search → Reciprocal Rank Fusion → cross-encoder rerank |
| **Latency forensics** | Guess at slowness | Langfuse traces (n=141): reranker is the bottleneck — measured, not guessed |
| **Provider resilience** | Single LLM, dies on rate limit | Groq → Anthropic → OpenAI failover with retry + exponential backoff |

</div>

---

### Who it's built for

| If you are... | This solves... |
|---|---|
| A researcher | Cross-paper synthesis without manual skimming |
| An engineer | Extracting implementation details from technical papers |
| A student | Citeable answers you can actually reference in writing |

---

## Results (Ragas, n=5 golden set)

Evaluated on 5 question–answer pairs from *Attention Is All You Need* with Groq LLM-as-judge. Source: `results.json`.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | **1.00** | 0.75 | PASS |
| Answer relevancy | **0.88** | 0.75 | PASS |
| Context recall | **1.00** | 0.70 | PASS |
| Context precision | **0.375** | 0.70 | **FAIL** |

**Faithfulness 1.00** — every claim grounded in retrieved context; system abstains rather than fabricates.

**Answer relevancy 0.88** — grounding prompt sometimes returns "I don't have enough information" instead of guessing. Desired tradeoff.

**Context precision 0.375 (FAIL)** — overlapping academic chunks pull partially relevant context. Planned fix: section-aware metadata filtering. See [Known limitations](#known-limitations).

Golden set is intentionally small (n=5) and honest about it. Target: expand to 30–50 verified pairs before tuning further.

---

## Architecture

### Document processing

```text
PDF Upload
    ↓
PyMuPDF Parser — page-aware extraction
    ↓
RecursiveCharacterTextSplitter — 256 character chunks, 100 overlap
    ↓
sentence-transformers/all-MiniLM-L6-v2 embeddings
    ↓
ChromaDB — cosine similarity vector store
```

### Retrieval and generation

```text
User Query
     ├── BM25 Search (Top 20)
     ├── Vector Search (Top 20)
     ↓
Reciprocal Rank Fusion (score = 1 / (k + rank), k=60)
     ↓
Cross-Encoder Reranker — ms-marco-MiniLM-L-6-v2
     ↓
Top 5 Chunks → Citation Prompt Builder
     ↓
LLM Provider Chain (retry + exponential backoff + failover)
     ├── Groq (default llama-3.1-8b-instant) — free
     ├── Anthropic (Claude) — optional, user-provided key
     └── OpenAI (GPT-4o) — optional, user-provided key
     ↓
Pydantic Citation Validator — per-sentence [SOURCE N] check
     ↓
Final Response with page-level citations
```

---

## Key technical decisions

<details>
<summary><b>Why hybrid retrieval instead of vector-only?</b></summary>

<br>

BM25 excels at exact keyword matching — critical for technical terminology like "scaled dot-product attention", "BLEU score", "LoRA adapters".

Vector retrieval handles semantic similarity.

Reciprocal Rank Fusion combines both without score normalization across heterogeneous retrievers.

</details>

---

<details>
<summary><b>Why use cross-encoder reranking?</b></summary>

<br>

Bi-encoders embed query and chunk independently. Cross-encoders score them together — significantly more accurate relevance.

Full-corpus cross-encoder inference is too expensive. Rerank only the top RRF candidates instead.

</details>

---

<details>
<summary><b>Why enforce citations with Pydantic validation?</b></summary>

<br>

Prompt instructions alone are unreliable. The validator checks **every sentence** for valid `[SOURCE N]` patterns. Any uncited sentence rejects the response at the validation layer.

This blocks partial hallucination where some sentences are cited and others are not.

</details>

---

<details>
<summary><b>Why Langfuse observability?</b></summary>

<br>

Production AI systems cannot be debugged with logs alone. Langfuse traces retrieval latency, prompt construction, token usage, LLM outputs, and citation validation.

See [How I found the bottleneck](#how-i-found-the-bottleneck).

</details>

---

## Anti-hallucination design (5 layers)

1. **Tighter chunking** — 350 → 256 characters, 100 overlap. Tighter context = less noise to fabricate from.
2. **Grounding prompt** — "ONLY use information from the provided sources." / "If sources don't contain enough information, say so." / cite every factual claim with `[SOURCE N]`.
3. **Chain-of-thought source identification** — model identifies relevant sources before answering. Explicit source reasoning first; generation second.
4. **Per-sentence citation validation** — Pydantic rejects any sentence missing `[SOURCE N]`. Previous version only required one citation total — partial hallucination slipped through.
5. **Graceful abstention** — insufficient sources → "I don't have enough information" instead of guessing. Completeness traded for accuracy.

---

## How I found the bottleneck

Every request traced end-to-end with Langfuse. **141 complete traces** collected across real usage.

### Latency profile (n=141)

| Metric | Latency | What it means |
|--------|---------|---------------|
| p50 | 1.54s | Most users experience this |
| p90 | 7.32s | 1 in 10 users waits this long |
| p95 | 11.09s | 1 in 20 users waits this long |
| p99 | 14.06s | Worst case observed |

### Component breakdown

| Component | p50 | p90 | p95 | p99 | Role |
|-----------|-----|-----|-----|-----|------|
| Full request | 1.54s | 7.32s | 11.09s | 14.06s | End-to-end |
| Retrieval | 0.77s | 6.11s | 10.01s | 12.19s | BM25 + vector + RRF |
| Rerank | 0.74s | 6.03s | 9.23s | 11.68s | Cross-encoder scoring |
| ChatGroq (LLM) | 0.57s | 0.93s | 1.19s | 2.52s | Generation |
| Vector search | 0.03s | 0.06s | 0.22s | 0.38s | Embedding lookup |

### Key finding: the LLM is not the bottleneck

Common assumption: LLM generation drives latency.

Data: LLM contributes only **0.57s at median**. Cross-encoder reranker is the actual bottleneck.

| Component | p50 | p95 | Variance ratio |
|-----------|-----|-----|----------------|
| Rerank | 0.74s | 9.23s | 12× |
| Retrieval | 0.77s | 10.01s | 13× |
| ChatGroq | 0.57s | 1.19s | 2× |
| Vector search | 0.03s | 0.22s | 7× |

Groq inference is fast and stable (2× p50→p95). Reranker shows 12× variance because the cross-encoder scores every (query, chunk) pair individually on CPU — no batching.

Optimizing the LLM — the intuitive target — would have had near-zero impact. This finding would have been invisible without instrumentation.

Trace spans: `retrieval` · `prompt-build` · `llm-call` · `citation-validation`.

---

## LLM provider failover

| Provider | Model (default) | Cost | Setup |
|----------|-----------------|------|-------|
| **Groq** | `llama-3.1-8b-instant` | Free | Set `GROQ_API_KEY` |
| **Anthropic** | Claude Sonnet 4 | Pay-per-use | Sidebar or `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o | Pay-per-use | Sidebar or `OPENAI_API_KEY` |

1. **Retry**: up to 3 attempts per provider, exponential backoff (1s → 2s → 4s)
2. **Failover**: provider fails after retries → next in chain
3. **Chain order**: Groq → Anthropic → OpenAI (only providers with keys)

**Add a key without code changes:** app sidebar → "LLM Providers" → enter key, or set env vars. Only Groq configured (default) works exactly as before — with retry resilience.

---

## Known limitations

| Limitation | Status | Planned fix |
|---|---|---|
| Context precision 0.375 (below 0.70 gate) | Open | Section-aware metadata filtering |
| Eval gate not wired into CI | Open | Run Ragas in `eval.yml`; block merge on threshold breach |
| Golden set n=5 | Open | Expand to 30–50 verified question–answer pairs |
| p95 latency 11.09s (CPU rerank) | Accepted for now | Reranker optimization (see bottleneck section) |
| Single shared Chroma collection (no multi-tenant isolation) | Open | Per-user collections |

CI currently runs unit tests with coverage (≥70%). Ragas evaluation runs locally via `python3 eval/eval_runner.py`. The CI badge reflects tests, not an automated eval gate.

---

## What I would change

Retrospective if rebuilding with what I know now:

1. **Eval harness first, pipeline second.** Golden set and gates before tuning retrieval. Would have caught context-precision issues earlier.
2. **Baseline before optimization.** Ship vector-only baseline, score it, then layer hybrid + rerank with measured deltas — not all at once.
3. **Fix precision before adding features.** Context precision 0.375 was known; section-aware metadata filter should precede any new capability.
4. **Wire the CI gate on day one.** Claiming an eval gate that doesn't block merges was a documentation-drift bug — this rewrite corrects the claim; next step is making it true.
5. **Smaller reranker, or GPU, earlier.** Reranker dominates p95 variance (12×) — obvious first target when faithfulness is already 1.00.
6. **Character vs token wording.** Chunk size is characters (`RecursiveCharacterTextSplitter`), not tokens — earlier README said tokens. Precision in claims matters.

---

## Technology stack

| Layer | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Sparse retrieval | BM25Retriever |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq / Anthropic / OpenAI (retry + failover) |
| Orchestration | LangChain |
| UI | Streamlit |
| Observability | Langfuse |
| Evaluation | Ragas |
| CI | GitHub Actions |

---

## Local setup

```bash
git clone https://github.com/aieng-abdullah/production-rag-assistant.git
cd production-rag-assistant

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env   # add GROQ_API_KEY (see note below)

streamlit run app.py
```

> **Note:** `.env.example` does not exist yet (tracked gap). Until it ships, create `.env` manually with at least `GROQ_API_KEY=...`. Optional: `GROQ_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CHROMA_MODE=local`, `LOG_LEVEL`.

Required: at least one of `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (env or sidebar).

---

## Running evaluation

```bash
python3 eval/eval_runner.py
```

Writes `results.json` (metric scores, thresholds, per-sample outputs).

---

## Running tests

```bash
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70
```

CI runs the same command on every push/PR.

---

## Performance notes

End-to-end latency: **1.54s p50 → 14.06s p99** across 141 traced requests.

Cross-encoder reranker accounts for the dominant share of tail latency on CPU (12× p50→p95 variance).

Current implementation prioritizes retrieval quality and grounded answers over raw latency.

---

## Author

### Abdullah Al Arif

 AI Engineer

[GitHub](https://github.com/aieng-abdullah) · [LinkedIn](https://www.linkedin.com/in/abdullah-al-arif-8b58542a7)
