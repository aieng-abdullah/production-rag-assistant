# PLAN: API-only ML → Render (Docker API + React Static Site)

**Status:** OPEN — today (2026-10-07): Phases 1, 3, 5 in progress · deferred: 2, 4, 6
**Date opened:** 2026-10-07
**Owner:** Abdullah
**Hosting decision:** Render for BOTH (backend Web Service + frontend Static Site) — confirmed 2026-10-07
**Supersedes:** docs/PLAN.md PR-6.1 (Streamlit-only revert), Stack locked decision
**Closes:** docs/case-studies/001-deployment-hosting-strategy.md (blocker removed)
**Related:** docs/PLAN.md PR-6 (restorable API split), refactor/drop-jwt (partially reverses)

---

## Thesis

The PR-6.1 revert rationale — *"no free host runs the API + ML backend"* — dies
when the ML leaves the backend. API-only embeddings + rerank remove torch
(~450MB RSS, ~90MB weights, 30–100s CPU rerank) from the container. The backend
becomes a plain FastAPI process that fits Render Free (512MB / 0.1 CPU).

| Change | Effect |
|---|---|
| Local cross-encoder → Voyage `rerank-3-lite` | Kills ~10s of ~11s query latency (72%); English-only → multilingual |
| Local `all-MiniLM-L6-v2` → Voyage `voyage-4-lite` | torch out of image (~450MB → ~200MB); Render Free becomes viable |
| Chroma embedded → Qdrant Cloud free | Render free filesystem wipes on every spin-down; Qdrant persists |
| SQLite `app.db` → Neon Postgres | Same wipe problem; Render free Postgres expires after 30 days (rejected) |
| Streamlit single process → FastAPI (Render Docker) + React (Render Static Site) | Public demo at $0; one vendor, one blueprint |

**Cost: $0/mo.** Budget ceiling ($5/mo) untouched.

---

## Today's scope (2026-10-07) — Phases 1, 3, 5

Backend/frontend split + local models → API. Infra/deploy work waits for
Qdrant/Neon keys (Phases 2, 4) and parity proof (Phase 6).

| Order | Phase | Branch | Work |
|---|---|---|---|
| 1 | Phase 1 | `feat/voyage-api-ml` | Remove local models — Voyage embeddings + rerank (torch dies) |
| 2 | Phase 3 | `feat/restore-api` | Restore FastAPI layer (backend split — CORS, `/health`, OAuth SPA redirect) |
| 3 | Phase 5 | `feat/react-frontend` (5a–5d) | React SPA frontend (frontend split) |

**Ready:** `VOYAGE_API_KEY` in `.env` ✓

**Deferred:** Phase 2 (Qdrant — needs cluster key), Phase 4 (Render deploy —
blocked on Phase 2: Render free disk is ephemeral, Chroma embedded cannot
persist there), Phase 6 (delete Streamlit — blocked on Phase 5 parity).

---

## Locked decisions to update in docs/PLAN.md FIRST — done (2026-10-07)

Guardrail: never swap the stack without updating the Locked decisions table.

| Axis | Old | New |
|---|---|---|
| Stack | Single Streamlit process (direct service calls) | **FastAPI on Render (Docker) + React SPA on Render Static Site** |
| Embeddings | local `sentence-transformers/all-MiniLM-L6-v2` | **Voyage `voyage-4-lite` (API-only, no local fallback)** |
| Reranker | local `cross-encoder/ms-marco-MiniLM-L-6-v2` | **Voyage `rerank-3-lite` (API-only, no local fallback)** |
| Vector store | Chroma embedded (`data/chroma/`) | **Qdrant Cloud free tier** |
| App DB | SQLite local (Postgres optional) | **Neon Postgres (prod), SQLite (tests)** |
| Frontend | Streamlit pages (delete after React ships) | **React SPA, full parity, then delete Streamlit** |

---

## Hosting architecture (chosen: Option A)

```
                    ┌── Render Static Site (free, CDN, NEVER sleeps) ── React SPA
 git push ──┬────────┤                                                      │
            │        └── Render Web Service (free, Docker) ── FastAPI API ◄─┘  CORS (one origin env)
            │                     │
            │                     ├── Neon Postgres (free forever) — users, answers, quotas
            │                     ├── Qdrant Cloud free — vectors (1GB RAM / 4GB disk)
            │                     └── Groq + Voyage APIs — LLM, embeddings, rerank
            └── render.yaml Blueprint: defines BOTH services, one push deploys both
```

| | Frontend | Backend |
|---|---|---|
| Render type | **Static Site** | **Web Service (custom Docker)** |
| Build | `npm ci && npm run build` → publish `dist` | Dockerfile → `uvicorn src.api.app:app` |
| Free spec | CDN, no spin-down, unlimited sites, 5GB bw/mo, **custom domain free** | 512MB / 0.1 CPU, 750h/mo workspace, spins down after 15min idle (~60s wake) |
| SPA routing | `routes: rewrite /* → /index.html` in render.yaml | n/a |

**Rejected alternative — Option B (single container, FastAPI serves React `dist/`):**
landing page would sleep with the API (60s cold start on first visitor kills the
portfolio demo), every UI tweak rebuilds the Python image (burns 500 build-min/mo),
coupled deploys. No CORS savings worth those costs.

**Consequences:**
- CORS: one env `FRONTEND_URL` on the API (static origin → API origin).
- API sleep: chat first message pays ~60s wake — React client shows a
  "waking backend" state, retries within its 60s timeout.
- No cron/workers on Render free → Qdrant keep-alive runs as a GitHub Action.

---

## Free-tier stack (verified 2026-10-07)

| Piece | Provider | Free spec | Gotcha |
|---|---|---|---|
| Backend | Render Web Service | 512MB, 0.1 CPU, 750h/mo, 15min spin-down | ephemeral disk — all state external |
| Frontend | Render Static Site | CDN, never sleeps, 5GB bw/mo, free custom domain | build minutes shared (500/mo) |
| Vectors | Qdrant Cloud | free forever: 1GB RAM / 4GB disk / 0.5 vCPU (~1M vec @768d), no card | **auto-suspend 1wk idle, delete 4wk** — keep-alive required |
| App DB | Neon Postgres | free forever, 0.5GB | (Render free Postgres expires 30 days — rejected) |
| Embeddings | Voyage `voyage-4-lite` | first 200M tokens free, then $0.02/1M | new dim → full re-ingest |
| Rerank | Voyage `rerank-3-lite` | same grant pool, $0.02/1M | ~5.3k tokens/query → ~37k queries |
| LLM | Groq | unchanged | — |

---

## Restorable from git — do not rebuild

- `34fa49d` (revert commit) deleted the whole API layer; restore it:
  - `src/api/{app,auth,chat,documents,answers,usage,billing,deps,security,uploads}.py`
  - `src/services/quotas.py`
  - `tests/test_api_*.py`, `tests/test_auth.py`, `tests/test_billing.py`, `tests/test_guest_tier.py`
- `refactor/frontend-api-client:api_client.py` — Streamlit httpx client. **Not
  restored** (React brings its own fetch); useful only as reference.
- What the old API lacked (real gaps, must be written): CORS, `GET /health`,
  SPA token redirect in the Google OAuth callback.

---

## Phase 1 — API-only ML (Voyage embeddings + rerank, torch dies)

Branch: `feat/voyage-api-ml`

- `src/config.py`:
  - add `VOYAGE_API_KEY`, `VOYAGE_EMBEDDING_MODEL` (default `voyage-4-lite`),
    `VOYAGE_RERANKER_MODEL` (default `rerank-3-lite`), `VOYAGE_BASE_URL`
    (default `https://api.voyageai.com/v1`), `RERANK_TIMEOUT_S` (default `10`)
  - remove `RERANKER_MODEL`; VOYAGE key is optional at Config level
    (graceful degradation = loud startup WARNING, not crash — GROQ stays the
    only required key)
- NEW `src/retrieval/reranker.py` — public `rerank(query, chunks, top_k, score_threshold=None)`:
  - httpx POST `/rerank` + tenacity retry ×2 exponential backoff (same
    pattern as `src/generation/chain.py:236`)
  - map `data[].index` / `relevance_score` → chunk + `rerank_score`
  - **no local fallback** — API failure raises; `pipeline.py:96` already wraps
    into `RuntimeError("Error while reranking: …")`
  - score scale: Voyage returns [0,1]; local cross-encoder returned raw logits.
    `score_threshold` is only used by tests today (`pipeline.py:91` never passes
    it) — document the scale in the docstring
- `src/ingestion/embedder.py`: replace `HuggingFaceEmbeddings` with a LangChain
  `Embeddings` implementation calling Voyage `/embeddings` (batched in
  `embed_chunks`). Keep names `embed_query` / `embed_chunks` — consumers
  (`chroma_client.py:71`, `chroma_search.py:22`, `pipeline.py:14` of ingestion)
  stay untouched.
- Delete: `src/retrieval/cross_encoder.py`; drop `sentence-transformers`,
  `langchain-huggingface` from requirements.txt (torch disappears transitively).
  Add `httpx>=0.27.0` (already imported directly by `src/auth/google_oauth.py:6`,
  currently only transitive).
- `ui_core.py:72-84` warm-up: `_get_model` symbols vanish for both — warm-up
  becomes config validation + Voyage reachability ping (log-only, never raises).
- `eval/eval_runner.py:34`: switch its direct `HuggingFaceEmbeddings` to the
  new embedder.
- Re-ingest demo corpus (embedding dimension changes; old vectors invalid).
- Tests: rewrite `tests/test_cross_encoder.py` → `tests/test_reranker.py`
  (mock httpx; API-order, empty-chunks, invalid top_k, threshold, failure
  propagation); update `tests/test_embedder.py`; fix warm-up tests.
  **Never loosen existing assertions.**

**Gate:** `ruff check src tests app.py eval alembic` clean;
`pytest tests/ -v -m "not slow" --cov=src --cov-fail-under=70` green;
one live query shows rerank step ~0.3s (was ~10s).

## Phase 2 — Chroma → Qdrant Cloud

Branch: `feat/qdrant-vector-store`

- `src/db/chroma_client.py` → Qdrant (`qdrant-client`; evaluate
  `langchain-qdrant` for filter translation — the seam is centralized):
  - `metadata_where()` → Qdrant filter DSL (`must` + `metadata.tenant_id`,
    `metadata.workspace`)
  - 1:1 reimpl: `upsert_chunks`, `load_all_chunks` (scroll), `count_chunks`,
    `purge_tenant`, `reassign_tenant`, `has_chunks`, `reset_client`
  - `get_vectorstore()` keeps the langchain-shaped interface
- Config: `QDRANT_URL`, `QDRANT_API_KEY` — local dev uses the same Cloud
  cluster (offline/local-only vector mode dies; update the PR-7
  "local-only mode preserved" promise in docs).
- Drop `chromadb`, `langchain-chroma` from requirements; remove the Chroma
  service from `docker-compose.yml` (Qdrant container for self-host).
- BM25 unchanged: still rebuilt in-memory from `load_all_chunks()`.
- Migration: re-ingest script (raw PDFs → Voyage → Qdrant). Second re-ingest
  across phases 1+2 — demo corpus makes it cheap; note it to the user.
- Update ~8 test files with Chroma fixtures.

**Gate:** cross-tenant leak test green against Qdrant; `count_chunks` parity.

## Phase 3 — Restore FastAPI layer

Branch: `feat/restore-api`

- `git revert 34fa49d` and drop from the revert: `api_client.py`,
  `tests/test_api_client.py` (Streamlit-only). Restores routers + quotas + API tests.
- Deps back: `pyjwt` (removed by `refactor/drop-jwt`), `fastapi`, `uvicorn`,
  `authlib` — confirm against restored imports.
- New (the real gaps):
  - **CORS** middleware: allow `FRONTEND_URL` origin (static site), credentials off (Bearer auth).
  - **`GET /health`** — Render health check path (case-study-001 flagged missing).
  - Google OAuth callback → redirect `{FRONTEND_URL}/auth#token=…`;
    React parses the fragment → stores Bearer (PR-6 bridge pattern, now a real SPA).
  - `Config.validate()` wired into API lifespan — boot fails loud when GROQ missing.
  - Guest tier `/auth/anonymous`: React sends a client-generated device UUID
    (browser-side storage), backend issues the session as before.
- JWT storage: **Bearer in memory + localStorage** (matches restored API that
  already returns the token in JSON). httpOnly cookie deferred — cross-site
  cookie pain between static site and API origins.
- Re-add `JWT_SECRET_KEY` to Config (revert of drop-jwt scope — document in
  PLAN.md under a `restore/api-auth` row).

**Gate:** restored API tests green; curl E2E: anonymous → upload → chat → quota 429.

## Phase 4 — Render deploy (Docker + Blueprint)

Branch: `chore/render-deploy`

- `render.yaml` Blueprint, two services:
  - `web` (Docker): Dockerfile (slim `python:312-slim`, no torch),
    `startCommand: alembic upgrade head && uvicorn src.api.app:app`,
    health check path `/health`, env/secrets (GROQ_API_KEY, VOYAGE_API_KEY,
    QDRANT_URL, QDRANT_API_KEY, DATABASE_URL, JWT_SECRET_KEY, FRONTEND_URL),
    build filter (docs/** changes skip rebuild)
  - `static` (frontend): `buildCommand: npm ci && npm run build`,
    `staticPublishPath: frontend/dist`, SPA rewrite route, `VITE_API_URL`
    build env = API service URL
- `DATABASE_URL` → Neon Postgres in prod (Config already supports PG;
  SQLite stays for tests); `alembic upgrade head` runs at boot (fresh Neon).
- Cold start: BM25 rebuild in API lifespan (seconds over Qdrant HTTP);
  readiness gate before serving.
- Qdrant keep-alive: GitHub Action cron pinging the free cluster weekly
  (prevents 1-week suspend / 4-week deletion). **Mandatory — document loudly.**
- Fill `docs/case-studies/001` Solution + Decision + Action items → status CLOSED.

**Gate:** deployed URL: login → upload → cited answer; spin-down survival check
(data intact in Neon + Qdrant after service sleep + wake).

## Phase 5 — React frontend, full parity (all 8 pages)

Branch: `feat/react-frontend` (stacked PRs — repo rule: keep diffs <400 lines)

New `frontend/`: Vite + React + TypeScript.

| PR | Port from | Scope |
|---|---|---|
| 5a | `app.py` login HTML, `pages/1_Landing.py` | shell, routing, auth (Google + demo + guest), landing/marketing |
| 5b | `pages/2_Chat.py` | chat, citations, verifier badge, provenance trace, "waking backend" state |
| 5c | `pages/3_Documents.py`, `4_Dashboard.py` | upload + status polling + delete, usage stats |
| 5d | `pages/5_Settings.py`, `6_Billing.py`, `7_Admin.py` | workspace switch, profile, pricing modal, admin |

- `frontend/src/api/`: typed fetch client — Bearer header, 401 → login,
  429 → quota message (server detail), 60s timeout (parity with old
  `api_client._TIMEOUT`).
- Reuse existing design tokens (`ui_core` CSS, `docs/UI_DESIGN.md`,
  the custom HTML already built for landing/login) — no redesign.
- Streamlit untouched until all four PRs merge.

**Gate per PR:** `npm run build` clean; parity checklist against the Streamlit page.

## Phase 6 — Delete Streamlit + docs

Branch: `chore/drop-streamlit`

- Remove `app.py`, `pages/`, `ui_core.py`, `streamlit` deps, Streamlit-specific tests.
- AGENTS.md rewrite: entrypoints (`uvicorn src.api.app:app`, `npm run dev
  --prefix frontend`), runtime invariants (required keys: GROQ, VOYAGE, QDRANT,
  DATABASE_URL, JWT_SECRET_KEY — no local models), CI unchanged shape.
- docs/PLAN.md: Locked decisions updated; PR-6 / PR-6.1 annotated
  "superseded by PLAN-render-react"; new rows for phases 1–6; Phase-2 backlog
  "SPA" item marked done.
- README: new architecture diagram + env table; data-sovereignty copy moves to
  "self-host with docker-compose + your own keys" (chunks now transit Voyage,
  Qdrant, Groq — the "your data never leaves your server" pitch dies for the
  hosted demo and must be rewritten).
- Secret-scan gate empty:
  `git log -p | grep -iE "sk-[a-zA-Z0-9]{20,}|pk_live_[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"`

---

## Execution order

**Today:** 1 → 3 → 5 (one PR per phase, Phase 5 = four stacked PRs).
Phases 1 and 3 are independent of Phase 2 — Phase 3 restores the API against
**Chroma still in place**; no Qdrant work until Phase 2.

**Later:** 2 → 4 → 6. Phase 4 blocked on Phase 2 (Render free disk wipes —
embedded Chroma cannot live there). Phase 6 blocked on Phase 5 parity.

Conventional Commits, self-review `git diff main...HEAD` line by line before
push, ruff + fast-test gate green locally first.

## Risks / open items

1. **Qdrant free idle deletion** — keep-alive Action mandatory or demo data vanishes.
2. **Render free restarts any time** + ~60s spin-up — client must retry; UX copy required.
3. **Data-sovereignty pitch dies** — rewrite marketing copy (self-host path remains).
4. **Voyage grant burn** — 2 re-ingests + usage on one 200M pool; log WARNING at 80%
   (response `usage.total_tokens`).
5. **Free tier ≠ production** — Render says so explicitly; acceptable (demo/portfolio,
   budget $0).
6. **Keys** (agent must not touch `.env` without explicit approval, per
   AGENTS.md): Voyage API key ✓ in `.env`; still pending — Qdrant Cloud
   cluster, Neon database.
7. **PR-6.1 rationale reversal** — record in case-study-001 why the decision flipped
   (API-only ML removed the blocker), otherwise the doc reads as contradiction.

## Out of scope

Stripe (PR-5 still open), SSE streaming, Bangla/multilingual phase-2 work,
custom domain setup, monitoring beyond existing Langfuse.
