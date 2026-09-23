# AGENTS.md

## Quick Commands

```bash
# Run app
streamlit run app.py

# Run CI tests (same as GitHub Actions: skips slow embedder tests, enforces 70% coverage)
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70

# Run all tests (including slow embedder tests, no coverage gate)
pytest tests/ -v

# Run evaluation (requires GROQ_API_KEY + pre-ingested docs in ChromaDB)
python3 eval/eval_runner.py
```

## Architecture

Single-app Streamlit project. No monorepo, no packages.

```
app.py                        # Streamlit UI entrypoint
src/
  config.py                   # Centralized config, reads .env, validates at startup
  ingestion/                  # PDF → chunks → embeddings → ChromaDB
  retrieval/                  # BM25 + vector search → RRF fusion → cross-encoder rerank
  generation/                 # Citation prompt builder + Groq LLM call + Pydantic validation
  db/                         # ChromaDB client (LangChain Chroma wrapper)
  monitoring/                 # Langfuse tracing (optional, fails silently if unconfigured)
eval/                         # Ragas evaluation runner
tests/                        # Pytest suite
```

## Key Facts

- **Python 3.12** required. `asyncio_mode = auto` in pytest.ini.
- **GROQ_API_KEY** is the only required env var. App crashes at startup without it.
- **ChromaDB** stores vectors in `data/chroma/`. Data dirs are gitignored.
- **BM25 index** is rebuilt in-memory from ChromaDB chunks on each app start or PDF upload. Not persisted separately.
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) runs on CPU. Accounts for ~72% of query latency (~10s of ~14s total).
- **Citation validation** is Pydantic-enforced: every answer must contain `[SOURCE N]` patterns or it raises `ValidationError`.
- **Langfuse** is optional. Traces are skipped silently when keys are absent.
- **Eval dataset** lives at `data/eval_dataset.json`. Results saved to `results.json`.

## CI Behavior

GitHub Actions (`.github/workflows/eval.yml`) runs on push/PR to `main`:
1. `pip install -r requirements.txt`
2. `pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70`

CI runs **all tests** except the slow embedder tests (marked `@pytest.mark.slow`). Coverage must stay above **70%** or the build fails. Coverage report is printed to the CI log with missing lines highlighted.

## Docker

`docker-compose.yml` runs ChromaDB + the Streamlit app. The app container reads `CHROMA_HOST=chromadb` to connect to the compose service. Locally, `CHROMA_MODE=local` uses persistent file storage.

## Gotchas

- `src/ingestion/embedder.py` loads `all-MiniLM-L6-v2` on CPU. First run downloads the model (~90MB).
- `src/retrieval/cross_encoder.py` lazily loads the reranker model. First query is slow.
- The Streamlit app stores uploaded PDFs in `data/raw/` and rebuilds BM25 on every upload.
- `load_all_chunks()` reads every document from ChromaDB. With large corpora this is expensive.
- Test files use inconsistent naming: some `test_*.py`, some `*_test.py`. Only `test_*` pattern files are auto-discovered by pytest.
