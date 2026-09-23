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

### Ask research papers questions. Every sentence cites its source. Zero hallucination by design  enforced by schema validation, not prompts.

Built for grad students and researchers drowning in arXiv PDFs who need trustworthy answers with page-level provenance  not confident-sounding guesses.

**[Try the live demo →](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)** · Upload a research paper PDF and ask questions with grounded citations.

---

### Why this isn't another RAG demo

| Differentiator | What most demos do | What this does |
|---|---|---|
| **Citation enforcement** | Prompt says "cite sources" | Pydantic validates **every sentence** for `[SOURCE N]`; violations rejected at the validation layer |
| **Retrieval** | Vector-only | BM25 ∥ vector search → Reciprocal Rank Fusion → cross-encoder rerank |
| **Latency forensics** | Guess at slowness | Langfuse traces: reranker = **72% of ~14s end-to-end** — measured, not guessed |

</div>

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
Groq LLM — llama-3.1-8b-instant (Config default; override via GROQ_MODEL)
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

Every query traced end-to-end with Langfuse: retrieval span, prompt-build, llm-call, citation-validation.

| Component | Latency | % of total |
|-----------|---------|------------|
| Cross-encoder reranker | ~10s | 72% |
| Vector search | ~0.4s | 3% |
| BM25 search | ~0.17s | 1% |
| Groq LLM | ~1.2s | 9% |
| Other | ~2s | 15% |

Measured from live traces — not guessed. Current priority: retrieval quality and grounded answers over raw latency. Optimizations on deck: GPU deployment, lighter reranker, smaller rerank candidate set.

---

## Known limitations

| Limitation | Status | Planned fix |
|---|---|---|
| Context precision 0.375 (below 0.70 gate) | Open | Section-aware metadata filtering |
| Eval gate not wired into CI | Open | Run Ragas in `eval.yml`; block merge on threshold breach |
| Golden set n=5 | Open | Expand to 30–50 verified question–answer pairs |
| End-to-end latency ~14s (CPU) | Accepted for now | Reranker optimization (see bottleneck section) |
| Single shared Chroma collection (no multi-tenant isolation) | Open | Per-user collections |

CI currently runs unit tests only. Ragas evaluation runs locally via `python3 eval/eval_runner.py`. The README badge reflects tests, not an automated eval gate.

---

## What I would change

Retrospective if rebuilding with what I know now:

1. **Eval harness first, pipeline second.** Golden set and gates before tuning retrieval. Would have caught context-precision issues earlier.
2. **Baseline before optimization.** Ship vector-only baseline, score it, then layer hybrid + rerank with measured deltas — not all at once.
3. **Fix precision before adding features.** Context precision 0.375 was known; section-aware metadata filter should precede any new capability.
4. **Wire the CI gate on day one.** Claiming an eval gate that doesn't block merges was a documentation-drift bug — this rewrite corrects the claim; next step is making it true.
5. **Smaller reranker, or GPU, earlier.** 72% of latency in one component is an obvious first target when quality is already at faithfulness 1.00.
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
| LLM | Groq (default `llama-3.1-8b-instant`) |
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

> **Note:** `.env.example` does not exist yet (tracked gap). Until it ships, create `.env` manually with at least `GROQ_API_KEY=...`. Optional: `GROQ_MODEL`, `CHROMA_MODE=local`, `LOG_LEVEL`.

Required env: at least one of `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

---

## Running evaluation

```bash
python3 eval/eval_runner.py
```

Writes `results.json` (metric scores, thresholds, per-sample outputs).

---

## Running tests

```bash
pytest tests/ -v
```

CI runs `tests/test_config.py`, `tests/test_rrf.py`, `tests/test_citations.py` on every push/PR.

> Known: `test_config.py` currently asserts stale chunk values (350/75 vs actual 256/100). Tracked fix.

---

## Author

### Abdullah Al Arif

JR. AI Engineer

[GitHub](https://github.com/aieng-abdullah) · [LinkedIn](https://www.linkedin.com/in/abdullah-al-arif-8b58542a7)
