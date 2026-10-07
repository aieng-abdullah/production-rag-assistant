# AGENTS.md

## Build & Test Entrypoints

```bash
# Run API (FastAPI)
uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# Run React frontend (Vite dev server)
cd frontend && npm run dev

# Lint (same as CI)
ruff check src tests app.py eval alembic
cd frontend && npm run lint

# Local test gate (CI runs the same tests WITHOUT the coverage gate)
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70

# Run all tests (including slow embedder tests, no coverage gate)
pytest tests/ -v

# Run evaluation (requires GROQ_API_KEY + Voyage API + pre-ingested docs)
python3 eval/eval_runner.py

# Frontend build
cd frontend && npm run build
```

## System Architecture

Two-tier: FastAPI backend + React SPA frontend. No monorepo.

```
app.py                        # Legacy Streamlit UI (being phased out)
src/
  config.py                   # Centralized config, reads .env, validates at startup
  ingestion/                  # PDF → chunks → embeddings (Voyage) → ChromaDB
  retrieval/                  # BM25 + vector search → RRF fusion → cross-encoder rerank (Voyage)
  generation/                 # Citation prompt builder + Groq LLM call + Pydantic validation
  services/                   # RAGService facade + BM25 cache (pages call this)
  db/                         # ChromaDB client (LangChain Chroma wrapper)
  monitoring/                 # Langfuse tracing (optional, fails silently if unconfigured)
  api/                        # FastAPI routes: auth, documents, chat, answers, usage, billing
frontend/                     # React 19 + TS + Vite 6 SPA
  src/
    pages/                    # Landing, Chat, Documents, Dashboard, Settings, Billing, Auth
    components/               # Reusable UI (Toast, TraceModal, WorkspaceSwitch, etc.)
    api/client.ts             # Typed fetch (Bearer, 401→login, 429 detail, 60s timeout)
    auth/AuthContext.tsx      # JWT storage, Google/guest/demo login, fragment token
  public/                     # favicon, static assets
eval/                         # Ragas evaluation runner
tests/                        # Pytest suite (backend)
```

## Runtime Invariants

- **Python 3.12** required. `asyncio_mode = auto` in pytest.ini.
- **GROQ_API_KEY** + **VOYAGE_API_KEY** + **JWT_SECRET** required. App crashes at startup without them.
- **ChromaDB** stores vectors in `data/chroma/`. Data dirs are gitignored.
- **BM25 index** rebuilt in-memory from ChromaDB chunks on each API start or PDF upload. Not persisted separately.
- **Cross-encoder reranker** (`voyage-rerank-3-lite`) runs via Voyage API. Accounts for major query latency.
- **Citation validation** is Pydantic-enforced: every answer must contain `[SOURCE N]` patterns or it raises `ValidationError`.
- **Langfuse** optional. Traces skipped silently when keys absent.
- **Eval dataset** at `data/eval_dataset.json`. Results to `results.json`.
- **Frontend dev** runs on `:5173`, proxies API to `:8001` via Vite config.
- **CORS**: single origin `Config.FRONTEND_URL` (default `http://localhost:5173`), no cookies.

## Continuous Integration Contract

GitHub Actions (`.github/workflows/ci.yml`) runs on PRs targeting `main`:

1. **lint** — `pip install ruff==0.15.10` → `ruff check src tests app.py eval alembic` + `cd frontend && npm run lint`
2. **test** — `pip install -r requirements.txt` → `pytest tests/ -v -m "not slow" --cov-fail-under=0` + `cd frontend && npm run build`

CI runs fast tests only (slow embedder tests marked `@pytest.mark.slow` skipped).
No coverage gate on PRs — keep ≥70% locally with the command above before pushing.

## Version-Control Hygiene & Branch Discipline

- **Never push directly to `main`** — branch protection enforces this. Every change goes through a PR branch.
- Branch names: `type/short-slug` — `feat/`, `fix/`, `chore/`, `ci/`, `docs/`, `refactor/`, `test/`, `security/`.
- Commits: Conventional Commits — `type: subject`, lowercase imperative, ≤50 chars.
- One concern per branch. No mixed refactor + feature work.
- Never commit: `.env`, `data/` outputs, `venv/`, keys/tokens.
- Pre-push secret scan must be empty:
  `git log -p | grep -iE "sk-[a-zA-Z0-9]{20,}|pk_live_[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"`
- Resolve conflicts inside the feature branch. No `merge:` conflict-fix commits on main.
- Branch policy: delete merged feature/fix/chore/docs branches after squash-merge; keep long-lived branches (`main`, `develop`, `release/*`, `gh-pages`) and any unmerged branch.
- Run `ruff check src tests app.py eval alembic` + `pytest -m "not slow" --cov-fail-under=70` + `cd frontend && npm run build && npm run lint` green locally before push (keep coverage ≥70%).

## Change Review Protocol (PR Contract)

- PR title = Conventional Commit (becomes the squash subject on main).
- Body sections: **What/Why** · **Changes** (bullets) · **Test plan** (commands + results) · **Screenshots** if UI · **Risk/rollback** if breaking.
- Keep diffs under ~400 lines; split larger work.
- **No PR before local code review.** Read your own `git diff main...HEAD` line by line — kill debug prints, dead code, secrets, stray files.
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
- **Retry with backoff:** exponential backoff for external API calls (Groq failover chain, Voyage 429 pacing). Don't hammer a failing service.

## Container Orchestration

`docker-compose.yml` runs ChromaDB + FastAPI + (optional) Streamlit. App container reads `CHROMA_HOST=chromadb`. Locally, `CHROMA_MODE=local` uses persistent file storage.

## Operational Pitfalls

- `src/ingestion/embedder.py` calls Voyage API (`voyage-4-lite` dim 1024). Trial: 3 RPM / 10K TPM — client-side pacing implemented.
- `src/retrieval/reranker.py` calls Voyage `rerank-3-lite`. Same trial limits.
- Frontend: Vite dev server on `:5173`, `VITE_API_URL` defaults to `http://localhost:8001`.
- `load_all_chunks()` reads every document from ChromaDB — expensive at scale.
- Test naming: some `test_*.py`, some `*_test.py`. Only `test_*` pattern auto-discovered by pytest.
- `docs/CHAT_FIX_PLAN.md` untracked — do not commit.

## Engineering Guardrails (Prohibited Patterns)

- Don't swap the stack (Groq / LangChain / Chroma / Voyage / FastAPI / React) without updating PLAN.md Locked decisions first.
- Don't hardcode workspace-specific logic (legal vs academic) into the pipeline — workspace profiles live in prompt config; the engine stays shared.
- Don't run irreversible actions (force-push, `git push --delete`, wiping `data/`, resetting ChromaDB, pushing to `main`) without explicit confirmation.
- Don't polish the Streamlit UI before the underlying service/pipeline is proven by tests — service layer first.
- Don't hardcode API keys or endpoints — env vars via `src/config.py` only.
- Don't keep long-lived feature branches — rebase on `main` frequently.
- Don't leave TODO comments without a linked PLAN.md ticket.
- Don't write clever one-liners that sacrifice readability.
- Don't over-engineer — start simple, iterate when constraints demand it.
- Don't write tests just to "make it pass" — write tests to verify correctness (edge cases, error paths, failure modes).
- **Never loosen test assertions to pass.** If a test fails, fix the code — not the test.
- **Never edit `.env` / `.env.*` without explanation first.** Before touching one you MUST tell the user: (1) which variable, (2) current value, (3) new value, (4) why. Wait for explicit approval.
- **Never make cosmetic changes to pass time.** No import reordering, no refactoring working code, no `try/except` → `contextlib.suppress` swaps unless asked.
- Pages call `src.services.RAGService` directly (single-process mode). `app.py` keeps no `src.*` imports; `src/` stays framework-free (no FastAPI/Streamlit imports below `app.py`).
