# PLAN.md — Citation-Verified RAG SaaS

**Thesis:** ChatGPT guesses. We verify — every claim, against the source,
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
| Stack           | **FastAPI backend (Render Docker) + React SPA (Render Static Site)** — docs/PLAN-render-react.md |
| Embeddings      | **Voyage `voyage-4-lite` (API-only, no local model)**     |
| Reranker        | **Voyage `rerank-3-lite` (API-only, no local model)**     |
| Frontend        | **React SPA** (Streamlit deleted after parity — Phase 6)  |
| Tenant isolation| Vector-store metadata `tenant_id` filter                  |
| Ingestion       | In-process + progress bar (blocking, v1)            |
| Billing         | Stripe env-flagged only — never on critical path    |
| Auth            | Session-state demo → Google OAuth (later, hosting)  |
| Agentic v1      | Citation Verifier + Provenance trace (T1 + T4)      |
| Backend DB      | Postgres (SQLite in tests) + Alembic                |

## Architecture

```
streamlit run app.py        # the whole app — one process
        │
        ├── pages/            UI → src.services.RAGService (direct calls)
        │
   src/
        ├── ingestion/        PDF → chunks → embeddings (blocking + progress)
        ├── retrieval/        BM25 + vector → RRF → cross-encoder rerank
        ├── generation/       citation prompts + Groq + Pydantic validation
        ├── services/         RAGService facade, BM25 cache, rag warm-up
        └── db/               Chroma (1 collection, tenant_id metadata)
                                + SQLite (users, usage, answers)
        files  data/raw/{tenant_id}/
```

`src/` stays framework-free (no Streamlit imports). Pages import
`src.services` directly — single-process mode (PR-6.1; the earlier
httpx/JWT split is reverted, restorable from history if hosting lands —
see docs/case-studies/001-deployment-hosting-strategy.md).

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
      — pyjwt removed later by `refactor/drop-jwt` (session-state auth)
- [x] Models: User, Workspace(legal|academic), Document, Answer, AnswerTrace,
      UsageEvent, Subscription
- [x] Alembic migrations; SQLite URL override in tests
- [x] `/auth/google` + `/auth/google/callback` → JWT (7d, HS256, env secret)
      — superseded: single-process session-state auth (no JWT)
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

### PR-3b — Data retention + configurable quotas
Branch: `feat/data-retention`
- [x] Raw PDFs purged after successful ingest (kept on failure for retry/debug);
      API background task and Streamlit page behave identically
- [x] Quota limits env-tunable: `DAILY_QUERY_LIMIT` (20), `DOCUMENT_LIMIT` (5),
      `STORAGE_LIMIT_MB` (100) — `Config` → `src/services/quotas.py`
- [x] `purge_tenant()` + `RAGService.delete_tenant_data()` — chunks + raw files
      for one tenant (account-deletion seam)
- [x] `.env.example` ships (README tracked-gap note removed)
**Hope:** storage stops growing per ingest; quotas tunable without code change.

### PR-4 — Workspaces (two niches, one engine)
Branch: `feat/workspaces`
- [x] Workspace profiles: `legal` (statute/section citation prompt, strict abstain),
      `academic` (paper/page, current behavior) — `src/generation/profiles.py`
- [x] Same pipeline, different `build_citation_prompt(workspace)` + source-type
      filter (`workspace` metadata key, Chroma `$and` predicate, tuple-keyed
      BM25 cache, server-side abstention in the citation validator)
- [x] Demo corpora: Indian Evidence Act 1872 + 3 arXiv papers
      (`scripts/fetch_demo_corpora.py`; Contract Act 1872 has no clean
      machine-readable source — archive.org scan is OCR noise)
- [x] Workspace-scoped tests for both profiles (predicates, forwarding,
      stamping, profiles, abstention)
**Hope:** one demo switches niches, legal prompt abstains on out-of-corpus.

### PR-4b — Citation Verifier + Provenance (agentic v1)
Branch: stacked `feat/verify-schema` → `feat/regression-eval` (PRs #44–#48)
- [x] Structured output schema (both workspaces, one validator): `claims[]` with
      `citations[{source_id, verbatim quote}]`, `abstained: bool`, `abstain_reason`;
      prose `[SOURCE N]` validator retired
- [x] Deterministic quote verification (whitespace/case-normalized containment,
      range-checked source_ids) + sources resolved server-side from chunk
      metadata — model never writes doc_id/page_num
- [x] Verifier: per-claim entailment judge `SUPPORTED/PARTIAL/UNSUPPORTED`
      (cheap model tier, separate prompt channel — injection hardened)
- [x] UNSUPPORTED → retry with error feedback (bound ≤2) → else `⚠ unverified`
      badge or abstain, never silent drop
- [x] Response shape + `verification{status, per_claim[]}`
- [x] `AnswerTrace` persisted: queries issued, chunks retrieved/used/discussed,
      prompt version, model output, verification events
- [x] `GET /answers/{id}/trace`
- [x] Provenance metadata: pinpoint (page/section/paragraph), content hash,
      doc date/version, jurisdiction; stable source numbering
- [x] Quota: verification = +1 unit (documented)
- [x] Regression eval set: proviso case, missing cross-ref, conflicting
      sources, out-of-corpus → track citation precision/recall + abstention
      accuracy before/after prompt changes
- [x] Tests: fake-judge unit tests, retry bound ≤2, trace shape, quota math
**Hope:** Langfuse shows verifier spans; badge visible in UI; CI green.
**Status:** all gates green (ruff clean, 341 tests, cov 93.05%); live regression
eval PASS (precision 1.0 / recall 1.0 / abstention 1.0).

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
- [x] Google sign-in button → JWT via URL **fragment** (`#token=`) → `st.session_state`
      — superseded: JWT removed (`refactor/drop-jwt`); identity in session_state only
- [x] Sidebar/chat/upload via `httpx` + Bearer; status polling + spinner
- [x] Workspace switcher (Legal / Academic)
- [x] Remove ALL direct `src.*` imports from `app.py`
- [x] Logout clears session state; token never on disk
- [x] Guest tier (ChatGPT-style): `POST /auth/anonymous` per-device session,
      `ANON_QUERY_LIMIT=3` questions + `ANON_DOCUMENT_LIMIT=1` upload, then
      the login wall; tier quotas enforced in `src/services/quotas.py`
- [x] `POST /auth/demo` flag-gated (`ENABLE_DEMO_LOGIN`, auto-on without
      Google creds) + legacy `default`-tenant adoption on first demo sign-in
**Hope:** `grep "from src" app.py` = empty; UX feels same as before.

### PR-6.1 — Revert to single-process Streamlit (2026-10-05)
**Superseded 2026-10-07 by docs/PLAN-render-react.md Phases 1/3/5 —
API-only ML removed the "no free host runs API + ML" blocker.**
Branch: `revert/streamlit-only`
- [x] Pages call `src.services.RAGService` directly again (pre-PR-6 wiring)
- [x] Delete `api_client.py`, `src/api/`, `src/services/quotas.py` and their
      tests (restorable from git history — commits survive on main)
- [x] Session-state demo auth + local quota counters; no backend process
- [x] Rationale: no free host runs the API + ML backend —
      docs/case-studies/001-deployment-hosting-strategy.md (open case)
- [ ] Public demo works on Streamlit Cloud with `streamlit run app.py` only
**Hope:** demo reachable at $0; hosting case study closes with the API
decision (restore the split or stay single-process).

### refactor/drop-jwt — Session-state auth (no JWT)
Branch: `refactor/drop-jwt`
- [x] Delete `src/auth/jwt_handler.py`; drop pyjwt from requirements.txt
- [x] Google OAuth callback + guest session write `user_id`/`user_email`/`user_tier`
      directly to `st.session_state` (no token)
- [x] `menu.py` / `dependencies.py` gate on `session_state.user_id`
- [x] Remove `JWT_SECRET_KEY` from Config, `.env`, `.env.example`, secrets.toml
- [x] Fixes Streamlit Cloud boot: no `import jwt` / ModuleNotFoundError
**Tradeoff:** no signed expiry/tamper-proof token; session = browser session.
Re-add JWT only if API split returns.

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

### PR-8 — README as product description
Branch: `docs/readme-product` (after PR-7 — builds on its technical README)
- [ ] First screen sells: hero tagline, problem → solution, not setup steps
- [ ] Demo GIF / screenshots: chat answer with citations + verifier badge
- [ ] Feature bullets in product language (citation-verified answers,
      legal/academic workspaces, quota limits, free vs pro plans)
- [ ] "Who it's for" audience section (researchers, lawyers, students)
- [ ] Keep PR-7 quickstart + env table below the fold (technical depth stays)
- [ ] Plans blurb matches the pricing modal: Free live, Pro coming soon
**Hope:** visitor understands the product value in 10 seconds, then self-hosts.

## Phase 2 backlog (explicit, not v1)
T2 reflective retrieval (LangGraph orchestrator) · T3 domain workflow agents ·
Bangla embeddings · Bangla OCR · bKash · teams/orgs · SSE streaming · SPA

## Security gates (ordered, each blocks next)
1. Global Config mutation gone (PR-0)
2. Cross-tenant leak test (PR-1)
3. ~~JWT secret env-only, no tokens logged (PR-2)~~ — superseded: no JWT
4. Upload sanitization (PR-3)
5. Webhook sig + env-gated routes + zero secrets in git (PR-5)
6. ~~Token memory-only (PR-6)~~ — superseded by PR-6.1 + drop-jwt: no tokens
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
