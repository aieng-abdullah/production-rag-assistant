# PLAN.md — Citation-Verified RAG SaaS

**Thesis:** ChatGPT guesses. We verify — every sentence, against the source,
before you see it.

Business discipline, portfolio expectation: no revenue required, money = plus.
One codebase, long-term commitment (2026→), all AI engineering skills applied.

## Locked decisions

| Axis            | Choice                                              |
|-----------------|-----------------------------------------------------|
| Niche           | Two workspaces, one engine: **Legal + Academic**    |
| Market          | Bangladesh-first, global/self-host friendly         |
| Language        | English v1 → Bangla phase 2 (multilingual-e5 + OCR) |
| Scope           | Multi-tenant, Google auth, free-tier quotas         |
| Stack           | FastAPI backend + Streamlit thin client             |
| Tenant isolation| Chroma metadata `tenant_id` filter                  |
| Ingestion       | FastAPI BackgroundTasks, poll status                |
| Billing         | Stripe env-flagged only — never on critical path    |
| Auth            | Google OAuth (env-gated) → JWT                      |
| Agentic v1      | Citation Verifier + Provenance trace (T1 + T4)      |
| Backend DB      | Postgres (SQLite in tests) + Alembic                |

## Architecture

```
Streamlit UI (thin client, httpx + JWT)
        │
FastAPI ├── /auth       Google OAuth → JWT
        ├── /documents  upload → BackgroundTasks ingest, list, delete
        ├── /chat       retrieve + generate + verify (tenant-scoped)
        ├── /answers    /{id}/trace provenance
        ├── /usage      quota stats
        └── /billing    Stripe — only if STRIPE_* env set
              │
   Postgres (users, workspaces, docs, answers, usage)
   Chroma (1 collection, tenant_id metadata)
   files  data/raw/{tenant_id}/
```

`src/` stays framework-free. Streamlit imports zero `src.*` after PR-6.

## PR breakdown (one PR = one phase, merge-to-main tracer bullets)

### PR-0 — Landmine removal + service layer
Branch: `fix/saas-service-layer`
- [x] Remove `_apply_provider_overrides()` global Config mutation (app.py:108)
      → request-scoped provider config (function args only)
- [x] Extract `src/services/rag_service.py`: `ingest(tenant_id, path)`,
      `query(tenant_id, q, bm25)`, `list_docs(tenant_id)`, `delete(tenant_id, doc_id)`
- [x] `app.py` refactored to call RAGService (behavior unchanged)
- [x] All existing 20 test files pass — refactor proven
**Hope:** green CI, zero behavior change, Config class immutable at runtime.

### PR-1 — Tenant-scoped core
Branch: `feat/tenant-metadata-scoping`
- [x] `upsert_chunks(tenant_id, ...)` stamps `metadata["tenant_id"]`
- [x] `vector_search` / `load_all_chunks` / `count_chunks` filter `where={tenant_id}`
- [ ] ~~BM25 per-tenant TTL cache~~ → moved to PR-3 (ships with the API that consumes it)
- [x] `collection.delete(where={tenant_id, doc_id})` on document delete
- [x] Migration: existing chunks tagged `tenant_id="default"`
- [x] **Cross-tenant leak test** (A never reads B) — merge gate
**Hope:** isolation proven by test, perf unchanged (filter is indexed metadata).

### PR-2 — Data layer + Google auth
Split: **PR-2a** `feat/data-layer` (models + Alembic) → **PR-2b** `feat/auth` (endpoints).
Branch: `feat/data-layer`, `feat/auth`
- [x] Deps: fastapi uvicorn sqlalchemy alembic authlib pyjwt httpx
      (per-PR: sqlalchemy+alembic in 2a; web/auth deps in 2b)
- [x] Models: User, Workspace(legal|academic), Document, Answer, AnswerTrace,
      UsageEvent, Subscription
- [x] Alembic migrations; SQLite URL override in tests
- [x] `/auth/google` + `/auth/google/callback` → JWT (7d, HS256, env secret)
- [x] Google creds absent → 501 + README setup instructions (open-source rule)
- [x] `require_user` dependency → `user_id` threads as tenant_id
**Hope:** sign-in works locally, no token in logs, 501 path tested.

### PR-3 — API endpoints + quotas
Branch: `feat/api`
- [x] BM25 per-tenant TTL cache (`cachetools`), invalidate on ingest/delete
      (moved from PR-1)
- [x] `POST /documents` → save `data/raw/{user_id}/` → bg task → `status=processing`
- [x] `GET /documents`, `GET /documents/{id}` (poll), `DELETE /documents/{id}`
- [x] `POST /chat` → same response shape as today's app
- [x] `GET /usage`; quotas: 20 queries/day, 5 docs, 100MB → 429 clear message
- [x] Upload hardening: PDF magic bytes, size cap, filename sanitization
**Hope:** full flow works via curl, quota 429 test green.

### PR-4 — Workspaces (two niches, one engine)
Branch: `feat/workspaces`
- [ ] Workspace profiles: `legal` (statute/section citation prompt, strict abstain),
      `academic` (paper/page, current behavior)
- [ ] Same pipeline, different `build_citation_prompt(profile)` + source-type filter
- [ ] Demo corpora: Contract Act 1872 + 2-3 arXiv papers
- [ ] Workspace-scoped tests for both profiles
**Hope:** one demo switches niches, legal prompt abstains on out-of-corpus.

### PR-4b — Citation Verifier + Provenance (agentic v1)
Branch: `feat/citation-verifier`
- [ ] Verifier: per-sentence entailment judge `SUPPORTED/PARTIAL/UNSUPPORTED`
      (cheap model tier, separate prompt channel — injection hardened)
- [ ] UNSUPPORTED → 1 retry → else `⚠ unverified` badge, never silent drop
- [ ] Response shape + `verification{status, per_sentence[]}`
- [ ] `AnswerTrace` persisted: queries issued, chunks retrieved/used/discarded,
      verification events
- [ ] `GET /answers/{id}/trace`
- [ ] Quota: verification = +1 unit (documented)
- [ ] Tests: fake-judge unit tests, retry bound ≤2, trace shape, quota math
**Hope:** Langfuse shows verifier spans; badge visible in UI; CI green.

### PR-5 — Stripe (flagged)
Branch: `feat/billing-flagged`
- [ ] Router registered only when `STRIPE_SECRET_KEY` etc. present
- [ ] `/billing/checkout`, `/billing/portal`, `/webhooks/stripe` (sig verify,
      idempotent, 400 on bad sig)
- [ ] Tiers: free (quotas) / pro (×10) — Subscription table lookup, fallback free
- [ ] `.env.example` empty Stripe vars; README "Running without payments" +
      "Enabling Stripe"
**Hope:** repo scans clean of secrets, flag-off app never exposes billing routes.

### PR-6 — Streamlit → API client
Branch: `refactor/frontend-api-client`
- [ ] Google sign-in button → JWT via URL **fragment** (`#token=`) → `st.session_state`
- [ ] Sidebar/chat/upload via `httpx` + Bearer; status polling + spinner
- [ ] Workspace switcher (Legal / Academic)
- [ ] Remove ALL direct `src.*` imports from `app.py`
- [ ] Logout clears session state; token never on disk
**Hope:** `grep "from src" app.py` = empty; UX feels same as before.

### PR-7 — Infra + business packaging
Branch: `chore/infra-docs`
- [ ] docker-compose: db(postgres:16) + api(uvicorn) + frontend(streamlit) + chroma
- [ ] Local-only mode preserved (AGENTS.md promise)
- [ ] CI: + API/verifier tests, `alembic upgrade head` step, 70% gate kept
- [ ] README: architecture diagram, env table, 5-min self-host
- [ ] ROADMAP.md: Bangla phase 2, OCR, T2 reflect loop, T3 workflow agents,
      bKash, teams, SSE streaming, SPA
- [ ] CHANGELOG.md + demo GIF
**Hope:** `docker compose up` = working product for reviewer in one command.

## Phase 2 backlog (explicit, not v1)
T2 reflective retrieval (LangGraph orchestrator) · T3 domain workflow agents ·
Bangla embeddings · Bangla OCR · bKash · teams/orgs · SSE streaming · SPA

## Security gates (ordered, each blocks next)
1. Global Config mutation gone (PR-0)
2. Cross-tenant leak test (PR-1)
3. JWT secret env-only, no tokens logged (PR-2)
4. Upload sanitization (PR-3)
5. Webhook sig + env-gated routes + zero secrets in git (PR-5)
6. Token memory-only (PR-6)
7. Verifier prompt-injection hardened (PR-4b)

## Non-goals v1
Revenue, Stripe-in-BD payments, LangGraph, teams, streaming, Bangla, OCR.

## Release plan

Versioning: semver, `v0.x.y` until v1.0. Every PR merge → tag only at milestones
below. GitHub Releases carry notes + docker pull instructions.

### Milestone releases

| Tag | After | Ships | Audience |
|-----|-------|-------|----------|
| `v0.1.0-core` | PR-1 | RAGService + tenant isolation, still Streamlit-monolith | internal, CI green |
| `v0.2.0-api` | PR-3 | FastAPI + auth + documents + chat + quotas, old UI | first public preview |
| `v0.3.0-workspaces` | PR-4 | legal + academic profiles, demo corpora | demo video, LinkedIn |
| `v0.4.0-verify` | PR-4b | **Verifier + provenance — flagship release** | portfolio centerpiece, show HN / BD AI communities |
| `v0.5.0-saas` | PR-6 | new frontend, billing flag, full multi-tenant | release candidate |
| `v1.0.0` | PR-7 | compose one-command boot, CI docs, README/ROADMAP/CHANGELOG | **public launch** |

### Release workflow (each milestone)
1. All PRs for milestone merged to `main`, CI green (70% gate)
2. Full test pass: `pytest tests/ -v -m "not slow" --cov=src --cov-fail-under=70`
3. Manual smoke: `docker compose up` → sign-in → upload → query → verify badge → trace
4. Secrets scan: `git log -p | grep -iE "sk-[a-zA-Z0-9]{20,}|pk_live_[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"` → empty
5. Tag `v0.x.y`, GitHub Release with notes (features + upgrade steps)
6. Update CHANGELOG.md; screenshot/GIF for release notes

### Deploy targets
- **v0.2+:** Streamlit Cloud (frontend) + any free host for API (Railway/Fly/Render)
      — demo URL lives here
- **v1.0:** docker-compose self-host documented; optional cloud deploy for live demo
- Envs: `staging` (manual tag) → `main` = production. No auto-deploy before v1.0.

### Launch checklist (v1.0 day)
- [ ] README badge row (CI, coverage, license, release)
- [ ] Demo GIF < 30s: two workspaces, verifier badge, provenance trace
- [ ] Post: LinkedIn (3-point description), r/BangladeshTech, BD AI Hub, HN Show
- [ ] ROADMAP.md public — signals long-term commitment
- [ ] GitHub Topics: `rag`, `fastapi`, `multi-tenant`, `citation-verification`
