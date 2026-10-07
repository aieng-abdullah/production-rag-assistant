# Chat Behavior & Performance Fix Plan

Symptoms (user transcript, 2026-10-07):
- `"hi"` → 10–14s "Thinking…" → abstain lecture + 8 sources, burns query quota
- `"what is attention layer"` → valid but terse, search-engine feel
- follow-ups impossible — each turn is amnesia (no conversation memory)
- whole pipeline feels slow / dead-air during query

## Root causes (verified in code)

| # | Cause | Evidence |
|---|-------|----------|
| 1 | Greetings treated as corpus queries | `pages/2_Chat.py:116 handle_query` → quota → RAG, no chitchat branch |
| 2 | No conversation memory | `chain.py:419 generate()` and `Citation_system.py:89 build_citation_prompt(query, chunks, workspace)` — no history param anywhere |
| 3 | No streaming | no `st.write_stream`; structured JSON + Pydantic validation + repair loop cannot stream token-by-token |
| 4 | Dead "Thinking…" during query | `pages/2_Chat.py` chat render — stage feedback only in logs |
| 5 | Abstain contradicts sources UI | abstained answer still renders `Sources —` chips |
| 6 | BM25 index rebuilds repeatedly | in-memory TTL cache, `TTL_SECONDS = 300` → 2.6s stall every 5 min + first query per tenant |
| 7 | Generation = two serial Groq calls | answer model + judge model, measured 6.1s of ~7s total |

## Measured latency (warm run, 8-core machine)

| Stage | ms |
|---|---|
| BM25 index build | 2,580 |
| BM25 search | 0.8 |
| Vector search | 21 |
| RRF fusion | 0.1 |
| Rerank (20 pairs) | 565 |
| Generation (2 Groq calls) | 6,116 |
| Repair loop | 0 (never fires) |
| **Warm total** | **~7,000** |

LLM pair = 73% of latency. BM25 rebuild = recurring stall. Old "~10s reranker"
claim was cold-start model load, not steady-state.

---

## Fix A — smalltalk short-circuit (fast, no LLM)

- New `src/generation/smalltalk.py`:
  `greeting_reply(query) -> str | None`
  - whole-message match (lowercase, strip punctuation), NOT substring
  - greetings (`hi`, `hello`, `hey`, `salam`, `good morning`, …), thanks,
    identity (`who are you`, `what can you do`)
  - returns canned warm reply mentioning workspace docs
- `handle_query`: check **before** quota + **before** "Thinking…" →
  append user message + render reply, no quota increment, no retrieval
- Tests: greeting matched · `"history"` not matched · real question
  untouched · `guest_queries` unchanged

## Fix C — conversation memory (biggest chatbot gap)

- `build_citation_prompt(..., history: list[dict] | None)`:
  - new `<conversation>` block above query (last 6 turns, role + text)
  - instruction: resolve pronouns (`it`, `that`) from history;
    evidence rules unchanged (sources block stays below)
- Query rewrite: small Groq call (`llama-3.1-8b-instant`) turns follow-ups
  into standalone retrieval query — **only when history exists**
  (`"and multi-head?"` → `"multi-head attention mechanism"`)
  - failure → fall back to raw query (never block)
- Thread through: `generate()` → `_run_pipeline`/`_generate_traced`
  → `rag_service.generate_answer` → `handle_query` passes
  `st.session_state.messages[-6:]`
- Prompt version bump: academic-v5 → v6, legal-v4 → v5
- Tests: history in prompt · rewrite called only with history ·
  rewrite failure falls back · Langfuse traced path passes history too

## Fix D — stage progress (perceived speed)

- Replace dead "Thinking…" lottie with stage captions driven by pipeline
  stages: "Searching 1,900 passages…" → "Reranking…" → "Writing answer…"
- `src/` stays framework-free: stages surfaced via callback/progress hook
- Chat UI: keep small typing indicator, drop per-query lottie

## Fix E — streaming (PARKED)

Groq streams fine, but structured JSON + verification + repair loop cannot
stream token-by-token. Needs prose-first redesign. Out of scope.

## Fix — abstain × sources contradiction

- **Chosen:** relabel chips when abstained →
  `Related passages (none support an answer)` (honest, keeps pointer)
- Alt (rejected): hide sources entirely on abstain

---

## Fix F1 — BM25 disk cache (kill 2.6s rebuilds)

- Persist index pickle per tenant+workspace under `data/bm25/`
- `invalidate()` hook (already called on ingest/delete) removes file
- TTL no longer triggers rebuild — rebuild only on actual corpus change
- Background prefetch at Chat open (current warmup skips BM25 —
  "needs tenant_id"; session known by then)
- Tests: save/load roundtrip · invalidate deletes file · stale/wrong
  tenant file never loads

## Fix F2 — judge model trim (PARKED, needs eval first)

Judge = serial Groq call (~2s). Faster `VERIFY_MODEL`
(e.g. `llama-3.1-8b-instant`) would cut generation 6.1s → ~4s, but
entailment quality must be checked against eval set first. Not in scope
until eval run confirms no faithfulness regression.

## Fix F3 — perceived speed

Covered by Fix D (stage captions + drop lottie).

## Fix F4 — rewrite model constraint

Covered by Fix C (`llama-3.1-8b-instant`, history-only, +0.3s max).

## Fix F5 — minor streamlining (optional)

- Streamlit config: `browser.gatherUsageStats = false`, headless
- Per-query lottie removal (in Fix D)

**Post-fix expectation:** ~7s → ~4.5s warm, no 2.6s stalls,
conversational behavior, live-feeling UI via captions.
Floor without streaming: serial LLM pair ≈ 3–4s.

---

## File touch list

| File | Change |
|------|--------|
| `src/generation/smalltalk.py` | NEW — greeting detector |
| `src/generation/Citation_system.py` | history block in prompt |
| `src/generation/chain.py` | thread history + rewrite + stage hook |
| `src/generation/profiles.py` | prompt version bump v6/v5 |
| `src/services/rag_service.py` | history param pass-through |
| `src/services/bm25_cache.py` | disk persist + invalidate wiring |
| `pages/2_Chat.py` | smalltalk branch, history pass, stage captions, abstain relabel, BM25 prefetch |
| `tests/test_smalltalk.py` | NEW |
| `tests/test_prompt_history.py` | NEW |
| `tests/test_bm25_persist.py` | NEW |

## PR slicing

1. `feat/chat-compact-sources` — already open, awaiting merge
2. `feat/login-html-ui` — already open, awaiting merge
3. `feat/landing-html` — already open, awaiting merge
4. `feat/smalltalk` — Fix A (independent)
5. `feat/conversation-memory` — Fixes C + D + abstain relabel
6. `feat/bm25-disk-cache` — Fix F1

Open PRs 1–3 stack on each other; 4–6 stack after them as merged.
Merge order: login → landing → compact-sources → smalltalk →
conversation-memory → bm25-disk-cache.

## Test plan

```bash
ruff check src tests app.py eval alembic
pytest tests/ -v -m "not slow" --cov=src --cov-fail-under=70
```

Manual checklist:
- `"hi"` → instant reply, quota untouched, no "Thinking…"
- `"what is attention layer"` → cited answer
- `"and multi-head?"` → rewritten, retrieves heads content
- abstained query → relabeled `Related passages` header
- second query < 10s after ingest (no BM25 rebuild stall)
- PDF upload → BM25 invalidated → rebuild once
