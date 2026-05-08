# Langfuse Observability Specification

## Why

Once an LLM system becomes even slightly real, debugging turns into 
digital archaeology. A user says "the answer was wrong" and you're 
reconstructing which prompt, retrieval chunk, model call, latency spike, 
or hallucination caused the disaster.

Langfuse solves this by providing end-to-end observability for the full 
RAG pipeline — traces every query, inspects prompts and responses, 
monitors token usage and latency, tracks retrieval quality, and enables 
systematic debugging instead of guessing.

---

## What We Observe Per Query

| Data Point | Where It Comes From |
|------------|-------------------|
| Raw query | User input |
| Retrieved chunks | `pipeline.py` |
| Full prompt | `build_citation_prompt()` |
| Model response | Groq LLM |
| Citations used | `CitedAnswer` |
| Step latencies | Each span |
| Token usage | LLM call |
| Success/failure | `CitedAnswer` validation |

---

## What We Track When Things Go Wrong

- Retrieval returning irrelevant chunks
- Prompt formatting errors
- Model hallucinations
- Timeout and latency spikes
- Citation validation failures
- Parsing errors in `CitedAnswer`
- Cost spikes

---

## Trace Structure

### Trace: `rag-generate`
One trace per user query.
rag-generate
├── span: retrieval
│   ├── bm25-search
│   ├── vector-search
│   ├── rrf-fusion
│   └── rerank
├── span: prompt-build
├── span: llm-call
└── span: citation-validation

### Inputs Recorded
```python
{
    "query": str,
    "chunks_retrieved": int,
    "top_k": int
}
```

### Outputs Recorded
```python
{
    "answer": str,
    "sources_count": int,
    "citation_valid": bool,
    "total_latency_ms": float
}
```

---

## Priority Functions to Monitor

chain.py      ← orchestration, LLM calls, citation validation
pipeline.py   ← retrieval quality, step latencies


---

## Implementation

This repository uses **Langfuse Python SDK 4.x** (`start_as_current_observation`, nested spans). Live wiring lives in:

- `src/monitoring/langfuse_tracer.py` — optional client + flush
- `src/generation/chain.py` — trace `rag-generate`, spans `retrieval` → `prompt-build` → `llm-call` → `citation-validation`
- `src/retrieval/pipeline.py` — nested retrieval steps: `bm25-search`, `vector-search`, `rrf-fusion`, `rerank`

Tracing runs only when **`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`** are set.

### Legacy-style snippet (conceptual; SDK 2.x)

Older docs used `langfuse.trace` / `langfuse.callback.CallbackHandler`. Equivalent ideas in v4:

```python
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    public_key=Config.LANGFUSE_PUBLIC_KEY,
    secret_key=Config.LANGFUSE_SECRET_KEY,
    host=Config.LANGFUSE_HOST,
)

trace_id = langfuse.create_trace_id()

with langfuse.start_as_current_observation(
    name="rag-generate",
    as_type="chain",
    trace_context={"trace_id": trace_id},
    input={"query": query},
) as root:
    with root.start_as_current_observation(name="retrieval", as_type="retriever") as retr:
        top_chunks = retrieval(query, chunks, bm25_index, lf_retrieval_parent=retr)

    handler = CallbackHandler(
        public_key=Config.LANGFUSE_PUBLIC_KEY,
        trace_context={"trace_id": trace_id},
    )
    response = client.invoke(citation_prompt, config={"callbacks": [handler]})
```

---

## Monitoring is Working When We Can Answer

- Why did the model hallucinate?
- Which chunk caused the bad citation?
- Why did latency jump from 2s to 12s?
- Which model version increased cost?
- Which queries fail most often?

---

## Success Criteria

Problems are identified and fixed from traces — not from blind guessing.