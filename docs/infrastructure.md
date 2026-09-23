# Infrastructure & Cloud Architecture

This document explains the infrastructure decisions behind the RAG Research Assistant — why each component was chosen, how they connect, and what tradeoffs were made.

---

## System Overview

```text
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Cloud                       │
│                    (app.py :8501)                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │
│  │   PDF    │  │   Chat   │  │   LLM Provider Chain  │ │
│  │ Uploader │  │   UI     │  │  Groq → Anthropic →   │ │
│  │          │  │          │  │       OpenAI           │ │
│  └────┬─────┘  └────┬─────┘  └───────────┬───────────┘ │
│       │              │                    │             │
│  ┌────▼──────────────▼────────────────────▼───────────┐ │
│  │              Generation Pipeline                   │ │
│  │  Retrieval → Prompt Build → LLM → Citation Check   │ │
│  └────┬──────────────────────────────┬───────────────┘ │
│       │                              │                 │
│  ┌────▼──────┐              ┌────────▼────────┐       │
│  │ ChromaDB  │              │   Langfuse      │       │
│  │ (persist) │              │   (tracing)     │       │
│  └───────────┘              └─────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Container Architecture

### Why Docker

The system has two runtime dependencies that benefit from containerization:

1. **ChromaDB** — vector database that needs persistent storage
2. **Streamlit app** — the application server

Docker Compose orchestrates both with a single command.

### Dockerfile Decisions

**Base image: `python:3.12-slim`**

Chosen over `python:3.12` (full) to reduce image size. The app only needs Python and a few system libraries for building native extensions (NumPy, sentence-transformers).

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
```

`build-essential` is required because `sentence-transformers` and `numpy` compile native code during `pip install`. `curl` is needed for the health check.

**No multi-stage build**

The embedding model (`all-MiniLM-L6-v2`, ~90MB) and reranker (`ms-marco-MiniLM-L-6-v2`, ~80MB) are downloaded at first runtime, not build time. A multi-stage build would not reduce the final image size meaningfully since these models are cached in the container's filesystem.

**Health check**

```dockerfile
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
```

Streamlit exposes a built-in health endpoint at `/_stcore/health`. This is used by orchestrators (Docker Compose, Kubernetes, cloud platforms) to determine if the container is ready to accept traffic.

**Entrypoint**

```dockerfile
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

`0.0.0.0` binds to all network interfaces inside the container. The port mapping (`8501:8501` in docker-compose) handles external access.

---

## ChromaDB Deployment

### Why ChromaDB

| Consideration | ChromaDB | Alternatives |
|---------------|----------|--------------|
| Embedding storage | Native support | Requires custom code |
| Metadata filtering | Built-in | Manual implementation |
| Persistence | File-based, zero config | Requires server setup |
| Deployment | Embedded or client-server | Separate infrastructure |

ChromaDB runs in embedded mode by default (file-based persistence in `data/chroma/`). The Docker Compose file overrides this to use client-server mode with the official ChromaDB Docker image.

### Local vs Docker Mode

**Local mode** (default, `CHROMA_MODE=local`):
- ChromaDB embedded in the Python process
- Data stored in `data/chroma/`
- No external dependency
- Used for development and Streamlit Cloud deployment

**Docker mode** (`docker-compose.yml`):
- ChromaDB runs as a separate container on port 8000
- Connected via `CHROMA_HOST=chromadb` environment variable
- Persistent volume (`chroma_data`) survives container restarts
- Used for production self-hosted deployments

### Data Persistence

```yaml
volumes:
  - chroma_data:/chroma/chroma
```

The named volume `chroma_data` persists vector embeddings across container restarts. Without this, re-uploading PDFs after a restart would be required.

The `./data` directory is also bind-mounted into the app container for raw PDF storage.

---

## LLM Provider Architecture

### Why Multi-Provider Failover

Single-provider architectures have a single point of failure. When Groq returns a 429 (rate limit) or 500 (server error), the entire system fails. Provider failover adds resilience without sacrificing the free tier.

### Provider Chain

```python
# src/generation/providers.py
chain = build_provider_chain()  # [Groq, Anthropic, OpenAI]
for provider in chain:
    try:
        return _call_with_retry(provider, prompt)
    except Exception:
        continue  # failover to next provider
```

**Chain order: Groq → Anthropic → OpenAI**

- Groq is first because it is free and fast
- Anthropic and OpenAI are fallbacks for when Groq is unavailable
- Only providers with valid API keys are included in the chain

### Retry + Exponential Backoff

Each provider gets up to 3 attempts with exponential backoff:

| Attempt | Delay | Rationale |
|---------|-------|-----------|
| 1 | 0s | Immediate retry |
| 2 | 1s | Brief wait for transient errors |
| 3 | 2s | Longer wait for persistent issues |

Backoff is implemented with `tenacity`:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_provider_with_retry(provider, prompt, config):
    client = create_langchain_client(provider)
    return client.invoke(prompt, config=config)
```

### Lazy Provider Imports

```python
def create_langchain_client(provider):
    if provider.name == "groq":
        from langchain_groq import ChatGroq  # lazy import
        return ChatGroq(api_key=provider.api_key, model=provider.model)
```

Imports are inside the function, not at module level. This means:

- Users only need to install the package for providers they actually use
- `langchain-anthropic` and `langchain-openai` are optional dependencies
- The base install stays lightweight (only `langchain-groq` is required)

### Runtime Key Injection

Users can add provider API keys at runtime through the Streamlit sidebar without modifying `.env` or restarting the app:

```python
# app.py
def _apply_provider_overrides():
    if st.session_state.get("anthropic_key"):
        Config.ANTHROPIC_API_KEY = st.session_state.anthropic_key
```

This is possible because `Config` is a class with mutable attributes, not a frozen dataclass. The tradeoff is that keys are session-scoped (not persisted across browser sessions).

---

## Observability Architecture

### Why Langfuse

Logs answer "what happened." Traces answer "why it happened."

Langfuse provides end-to-end request tracing that breaks each query into spans:

| Span | What It Captures |
|------|-----------------|
| `retrieval` | BM25 + vector search + RRF + rerank latency |
| `prompt-build` | Prompt length and construction time |
| `llm-call` | Token usage, model, response time |
| `citation-validation` | Pass/fail status |

This granularity identified that the cross-encoder reranker accounts for 72% of total latency — a finding that would be invisible in aggregate logs.

### Fail-Safe Design

```python
# src/monitoring/langfuse_tracer.py
def get_langfuse_client():
    if not langfuse_configured():
        return None  # tracing disabled, app continues
    try:
        _client = Langfuse(...)
    except ImportError:
        return None  # package not installed, app continues
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        return None  # connection failed, app continues
```

Langfuse is completely optional. If the package is not installed, keys are not set, or the server is unreachable, the app continues without tracing. This prevents observability infrastructure from becoming a reliability dependency.

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/eval.yml
- name: Run tests with coverage
  run: pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70
```

**Key decisions:**

| Decision | Rationale |
|----------|-----------|
| `ubuntu-latest` | Cheapest CI runner, sufficient for Python tests |
| `python-version: "3.12"` | Matches production runtime |
| `cache: "pip"` | Caches installed packages across runs, faster CI |
| `-m "not slow"` | Skips embedder tests (~60s each), keeps CI under 2 minutes |
| `--cov-fail-under=70` | Blocks merges if coverage drops below 70% |

**What CI does NOT do:**

- No deployment — this is a test-only pipeline
- No Docker build — container builds happen at deploy time
- No evaluation run — Ragas eval requires a live Groq key and ingested documents

### Coverage Gate

The 70% coverage threshold is enforced at the PR level. If a PR reduces coverage below 70%, the CI job fails and the PR cannot be merged.

Coverage is measured against `src/` only, excluding test files and evaluation scripts.

---

## Deployment Modes

### 1. Streamlit Cloud (Current Production)

```text
GitHub push → Streamlit Cloud auto-deploys → app.streamlit.app
```

- Zero infrastructure management
- ChromaDB in embedded mode (file-based)
- Environment variables set in Streamlit Cloud dashboard
- Limitation: no Docker, no persistent volumes across deploys

### 2. Docker Compose (Self-Hosted)

```bash
docker-compose up -d
```

- ChromaDB as a separate container with persistent volume
- App container with health checks
- All environment variables via `.env` file
- Suitable for single-server deployments

### 3. Kubernetes (Scalable)

Not currently implemented, but the architecture supports it:

- ChromaDB → managed service (e.g., Chroma Cloud, or self-hosted StatefulSet)
- App → Deployment with horizontal pod autoscaling
- Langfuse → separate namespace or managed service
- Secrets → Kubernetes Secrets or external vault

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GROQ_API_KEY` | Yes* | — | Groq API authentication |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model name |
| `ANTHROPIC_API_KEY` | No | — | Anthropic failover |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Anthropic model name |
| `OPENAI_API_KEY` | No | — | OpenAI failover |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `CHROMA_HOST` | No | `localhost` | ChromaDB host |
| `CHROMA_PORT` | No | `8000` | ChromaDB port |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse tracing |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse endpoint |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |

*At least one provider API key is required. It can be set via `.env` or the sidebar UI.

---

## Security Decisions

| Decision | Rationale |
|----------|-----------|
| `.env` in `.gitignore` | Prevents accidental API key commits |
| Sidebar keys in `st.session_state` | Session-scoped, not persisted to disk |
| `type="password"` for key inputs | Masks API keys in the sidebar |
| No hardcoded secrets | All keys read from environment variables |
| `validate()` at startup | Fails fast if critical keys are missing |

---

## Tradeoffs & Limitations

### Accepted Tradeoffs

| Tradeoff | Why |
|----------|-----|
| ChromaDB embedded mode on Streamlit Cloud | No persistent volumes available; data is lost on redeploy |
| BM25 index rebuilt on every app start | In-memory only; avoids index serialization complexity |
| Cross-encoder reranker on CPU | Accuracy over speed; accounts for 72% of latency |
| Session-scoped provider keys | No database for key storage; keys lost on browser close |

### Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No horizontal scaling | Single-user or small team only | Docker Compose for self-hosting |
| ChromaDB data lost on Streamlit Cloud redeploy | Users must re-upload PDFs | Use self-hosted ChromaDB for production |
| Embedding model downloaded at first run | ~90MB download on cold start | Pre-bake into Docker image |
| BM25 index not persisted | Rebuilt from ChromaDB on each startup | Acceptable for current corpus size |
