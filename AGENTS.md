# AGENTS.md

## Build & Test Entrypoints

```bash
# Run app
streamlit run app.py

# Lint (same as CI)
ruff check src tests app.py eval alembic api_client.py

# Local test gate (CI runs the same tests WITHOUT the coverage gate)
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70

# Run all tests (including slow embedder tests, no coverage gate)
pytest tests/ -v

# Run evaluation (requires GROQ_API_KEY + pre-ingested docs in ChromaDB)
python3 eval/eval_runner.py
```

## System Architecture

Single-app Streamlit project. No monorepo, no packages.

```
app.py                        # Streamlit UI entrypoint
api_client.py                 # httpx client pages use to reach the FastAPI backend (PR-6)
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

## Runtime Invariants

- **Python 3.12** required. `asyncio_mode = auto` in pytest.ini.
- **GROQ_API_KEY** is the only required env var. App crashes at startup without it.
- **ChromaDB** stores vectors in `data/chroma/`. Data dirs are gitignored.
- **BM25 index** is rebuilt in-memory from ChromaDB chunks on each app start or PDF upload. Not persisted separately.
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) runs on CPU. Accounts for ~72% of query latency (~10s of ~14s total).
- **Citation validation** is Pydantic-enforced: every answer must contain `[SOURCE N]` patterns or it raises `ValidationError`.
- **Langfuse** is optional. Traces are skipped silently when keys are absent.
- **Eval dataset** lives at `data/eval_dataset.json`. Results saved to `results.json`.

## Continuous Integration Contract

GitHub Actions (`.github/workflows/ci.yml`) runs on PRs targeting `main`:

1. **lint** — `pip install ruff==0.15.10` → `ruff check src tests app.py eval alembic api_client.py`
2. **test** — `pip install -r requirements.txt` → `pytest tests/ -v -m "not slow" --cov-fail-under=0`

CI runs fast tests only (slow embedder tests marked `@pytest.mark.slow` are skipped).
No coverage gate on PRs — keep ≥70% locally with the command above before pushing.

## Version-Control Hygiene & Branch Discipline

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
- Run `ruff check src tests app.py eval alembic api_client.py` + the fast test command green locally before push (keep coverage ≥70%).

## Change Review Protocol (PR Contract)

- PR title = Conventional Commit (becomes the squash subject on main).
- Body sections: **What/Why** · **Changes** (bullets) · **Test plan**
  (commands run + results) · **Screenshots** if UI · **Risk/rollback** if breaking.
- Keep diffs under ~400 lines; split larger work. PLAN.md breakdown = one PR = one phase.
- Link the PLAN.md ticket (PR-0…PR-7) when applicable.
- **No PR before local code review.** Before opening: read your own
  `git diff main...HEAD` line by line — kill debug prints, dead code,
  secrets, stray files. CI review comes *after* local self-review, never
  instead of it.
- Draft PR = WIP; mark ready before requesting review.

## Software Quality Attributes

- **Scalability:** keep code and architecture modular and scalable.
- **Modular architecture:** each component is a separate unit with a clear interface.
- **Clean, readable code:** prioritize clarity over cleverness — code is read more than written.
- **Production-focused:** handle errors, log meaningfully, fail loudly; every line should be production-ready.
- **Simple over complex:** if a solution needs more than one abstraction layer, you're overcomplicating it.

## Logging & Error-Handling Contract

- **Structured logging:** `loguru` with context fields — no raw `print` statements.
- **Log levels:** `ERROR` (needs attention), `WARNING` (degraded state), `INFO` (business events), `DEBUG` (dev detail).
- **Never swallow exceptions:** every caught error must be logged or re-raised. Silent failures are bugs.
- **Fail fast on startup:** crash immediately when required env vars are missing (`Config.validate()`).
- **Graceful degradation:** if a non-critical dependency (Langfuse) is down, log and continue — never block core queries.
- **Error responses:** return structured error objects to callers — never stack traces or raw exceptions.
- **Retry with backoff:** exponential backoff for external API calls (Groq failover chain). Don't hammer a failing service.

## Container Orchestration

`docker-compose.yml` runs ChromaDB + the Streamlit app. The app container reads `CHROMA_HOST=chromadb` to connect to the compose service. Locally, `CHROMA_MODE=local` uses persistent file storage.

## Operational Pitfalls

- `src/ingestion/embedder.py` loads `all-MiniLM-L6-v2` on CPU. First run downloads the model (~90MB).
- `src/retrieval/cross_encoder.py` lazily loads the reranker model. First query is slow.
- The Streamlit app stores uploaded PDFs in `data/raw/` and rebuilds BM25 on every upload.
- `load_all_chunks()` reads every document from ChromaDB. With large corpora this is expensive.
- Test files use inconsistent naming: some `test_*.py`, some `*_test.py`. Only `test_*` pattern files are auto-discovered by pytest.

## Engineering Guardrails (Prohibited Patterns)

- Don't swap the stack (Groq / LangChain / Chroma / Streamlit) without updating the
  PLAN.md **Locked decisions** table first.
- Don't hardcode workspace-specific logic (legal vs academic) into the pipeline —
  workspace profiles live in prompt config (PLAN PR-4); the engine stays shared.
- Don't run irreversible actions (force-push, `git push --delete`, wiping `data/`,
  resetting ChromaDB, pushing to `main`) without explicit confirmation.
- Don't polish the Streamlit UI before the underlying service/pipeline is proven by
  tests — service layer first (PLAN PR-0 → PR-6 order).
- Don't hardcode API keys or endpoints — env vars via `src/config.py` only.
- Don't keep long-lived feature branches — rebase on `main` frequently.
- Don't leave TODO comments without a linked PLAN.md ticket (PR-0…PR-7).
- Don't write clever one-liners that sacrifice readability.
- Don't over-engineer — start simple, iterate when constraints demand it.
- Don't write tests just to "make it pass" — write tests to verify correctness.
  If the code is good, tests pass automatically. Focus on edge cases, error paths,
  and failure modes — not happy-path-only green checks.
- **Never loosen test assertions to pass.** If a test fails, fix the code — not the
  test. Never change `assert X in result` to `assert X in result or Y in result`.
  Never skip tests without explicit user request.
- **Never edit `.env` / `.env.*` without explanation first.** Before touching one you
  MUST tell the user: (1) which variable, (2) current value, (3) new value,
  (4) why. Wait for explicit approval.
- **Never make cosmetic changes to pass time.** No import reordering, no refactoring
  working code, no `try/except` → `contextlib.suppress` swaps unless the user asks.
  Every change must have a functional purpose.
- Don't add new direct `src.*` imports to `app.py` — go through `RAGService`;
  `src/` stays framework-free (no Streamlit imports below `app.py`).
