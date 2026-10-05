# Case Study 001: Deployment & Hosting Strategy for Citation-Verified RAG

**Status:** OPEN — solution pending research
**Date opened:** 2026-10-05
**Owner:** Abdullah
**Related:** PLAN.md PR-6 (service/API split), PR-55 (frontend API client), PR-57 (lottie animations)

---

## Problem

PR-6 split the application into two processes:

- **Streamlit frontend** (`streamlit run app.py`) — UI, session state
- **FastAPI backend** (`uvicorn src.api.app:app`) — auth, quotas, ingest, chat

Both run fine locally (single laptop, `localhost:8001`). There is **no
hosting strategy that runs both publicly at $0/month** with acceptable
performance. The frontend alone fits Streamlit Cloud; the backend does not
fit anywhere in the free tier with its ML workload.

The deployment gap blocks the public demo (the README "live demo" link on
Streamlit Cloud shows a broken app: no backend → "Backend unreachable",
raw OAuth-bridge text, no ingest/chat functionality).

---

## Constraints

| # | Constraint | Detail |
|---|------------|--------|
| 1 | Budget | $0/mo target; ~$5/mo absolute max |
| 2 | ML workload in backend | torch + `all-MiniLM-L6-v2` (~300MB RSS) + cross-encoder reranker (~90MB) |
| 3 | Latency budget | Client timeout 60s (`api_client._TIMEOUT`); rerank on a weak CPU = 30–100s |
| 4 | Background ingest | PDF upload must not block the UI (202 + polling already built) |
| 5 | Data sovereignty pitch | BD market positioning: "self-hosted, your data never leaves your server" |
| 6 | Model downloads | Fresh instance needs ~90MB of HuggingFace weights or it OOMs/slow-starts |

---

## Investigation / Findings (2026-10-05)

### Option matrix

| Option | Specs | Cost | Verdict |
|--------|-------|------|---------|
| Streamlit Cloud (UI only) | Runs one process: `streamlit run app.py` | $0 | Backend **cannot** run here |
| Render Free | 512MB RAM, 0.1 CPU, spins down, no persistent disk | $0 | ❌ torch + embedder ≈ 450MB → OOM risk; cross-encoder would add ~90MB and 30–100s/query on 0.1 CPU (breaks 60s timeout) |
| OpenRouter free embeddings (`nvidia/nemotron-3-embed-1b:free`) | 20 rpm, **50 req/day** (1,000/day after $10 credits) | $0 | ❌ Dead for RAG: one 20–50 chunk PDF ingest = entire daily quota |
| HF Spaces CPU Basic | 7GB RAM, 2 vCPU, Docker, sleeps after 48h idle | $0 | ✅ Fits ML workload; cold start ~1–2min |
| Oracle Cloud Free Tier | 4 ARM cores, 24GB RAM, 200GB disk, always free (card for verification) | $0 | ✅ Best specs; requires VPS ops (SSH, Docker, TLS) |
| Cheap VPS (Contabo/Hetzner) | 4 vCPU, 8GB RAM | ~$5/mo | ✅ Production-ready; costs money |
| Single-process Streamlit (revert PR-6) | All logic in one process; blocks on ingest | $0 | ⚠️ Works everywhere; regresses the API split (1,250 lines, 391 tests, PLAN locked decision) |

### Side findings

- Streamlit Cloud secrets are exposed via `st.secrets`, **not** `os.environ`
  — any config read through `os.getenv` needs a mirror step.
- Backend has **no `/health` endpoint** (Render/HF health checks need one).
- Chroma is embedded (`persist_directory`), not an HTTP service — backend is
  self-contained in one container; no separate vector DB to host.
- `data/app.db` starts as a 0-byte file; fresh instances need
  `alembic upgrade head` at boot.
- The OAuth bridge JS was shipped unwrapped (PR-6) and rendered as visible
  text until `fix/oauth-bridge` (`03f9c4e`) — root cause: `st.html` executes
  only `<script>`-wrapped JS.

---

## Solution

<!-- BLANK — research pending. Fill in: chosen hosting, why, deploy steps,
     cost, and how each constraint above is satisfied. Then close. -->

---

## Decision

<!-- BLANK -->

---

## Action items

<!-- BLANK -->

---

## Closing criteria

- [ ] Hosting chosen and documented here
- [ ] Public demo reachable, backend functional (login, upload, chat with citations)
- [ ] Cost verified against budget
- [ ] Status → **CLOSED**
