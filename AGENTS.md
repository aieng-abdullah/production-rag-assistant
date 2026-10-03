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

## Git Hygiene

- **Never push directly to `main`** — branch protection enforces this.
  Every change goes through a PR branch, even one-line doc fixes.

- Branch names: `type/short-slug` — `feat/`, `fix/`, `chore/`, `ci/`, `docs/`, `refactor/`, `test/`.
- Commits: Conventional Commits — `type: subject`, lowercase imperative, ≤50 chars.
  Body only when the "why" isn't obvious from the subject (see existing `ci:` commits).
- One concern per branch. No mixed refactor + feature work.
- Never commit: `.env`, `data/` outputs, `venv/`, keys/tokens.
  Pre-push secret scan must be empty:
  `git log -p | grep -iE "sk-[a-zA-Z0-9]{20,}|pk_live_[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"`
- Resolve conflicts inside the feature branch. No `merge:` conflict-fix commits on main.
- Squash-merge PRs, then delete the head branch. Clean up stale branches after merge.
- Run the CI test command green locally before push (coverage ≥70 gate).

## PR Structure

- PR title = Conventional Commit (becomes the squash subject on main).
- Body sections: **What/Why** · **Changes** (bullets) · **Test plan**
  (commands run + results) · **Screenshots** if UI · **Risk/rollback** if breaking.
- Keep diffs under ~400 lines; split larger work. PLAN.md breakdown = one PR = one phase.
- Link the PLAN.md ticket (PR-0…PR-7) when applicable.
- **No PR before local code review.** Before opening: read your own
  `git diff main...HEAD` line by line — kill debug prints, dead code,
  secrets, stray files. CI and PR-Agent review come *after* local
  self-review, never instead of it.
- Draft PR = WIP; mark ready before requesting review.

## Docker

`docker-compose.yml` runs ChromaDB + the Streamlit app. The app container reads `CHROMA_HOST=chromadb` to connect to the compose service. Locally, `CHROMA_MODE=local` uses persistent file storage.

## Gotchas

- `src/ingestion/embedder.py` loads `all-MiniLM-L6-v2` on CPU. First run downloads the model (~90MB).
- `src/retrieval/cross_encoder.py` lazily loads the reranker model. First query is slow.
- The Streamlit app stores uploaded PDFs in `data/raw/` and rebuilds BM25 on every upload.
- `load_all_chunks()` reads every document from ChromaDB. With large corpora this is expensive.
- Test files use inconsistent naming: some `test_*.py`, some `*_test.py`. Only `test_*` pattern files are auto-discovered by pytest.
