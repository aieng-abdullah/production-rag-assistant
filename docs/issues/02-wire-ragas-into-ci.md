# Wire Ragas and the citation gate into CI

**Labels:** `ci`, `eval`, `priority:high`

## Problem

Both quality gates run only on a maintainer's laptop. `.github/workflows/ci.yml`
has exactly two gates:

```
lint:  ruff check src tests app.py eval alembic
test:  pytest tests/ -v -m "not slow" --cov-fail-under=0
```

`eval/eval_runner.py` and `eval/verify_eval.py` are absent from CI. AGENTS.md
states this explicitly: *"Eval is LOCAL-ONLY: it is not run in CI."*

Consequences:

- The `context_precision` gate has been FAILing at 0.375 in a committed file
  and CI stayed green. Nobody was stopped.
- `results.json` drifted to 5 samples against a 29-item dataset (see the eval
  baseline issue) and no check noticed.
- A PR can regress `src/retrieval/**` or `src/generation/**` — the exact paths
  AGENTS.md gates on both evals — and merge with the gates never having run.

## Blocker: paid API keys

Ragas needs `GROQ_API_KEY`, `VOYAGE_API_KEY`, and `QDRANT_URL` +
`QDRANT_API_KEY`. Voyage trial is 3 RPM / 10K TPM
(`src/ingestion/embedder.py`, `src/retrieval/reranker.py`), so a naive 29-item
run will rate-limit hard. This is the real design problem, not a config detail.

`eval/verify_eval.py` is cheaper — deterministic, no judge model — and should
go first.

## Scope

- [ ] **Gate 1 — deterministic citation checks.** Wire `python3 eval/verify_eval.py`
      into CI. No LLM judge, no embeddings, no rate-limit exposure. Fail the
      job when any row in `verify_results.json` reads FAIL. Commit
      `verify_results.json` and diff it in the job so drift is visible.
- [ ] **Gate 2 — Ragas.** Decide between three options and record the choice in
      the PR body:
  - scheduled workflow (nightly on `main`) rather than per-PR, so per-PR
      feedback stays fast and the paid calls stay bounded
      - per-PR job with Voyage request pacing already in place
      - PR runs against a small committed fixture subset (fast, deterministic)
        while the full 29-item run stays scheduled
- [ ] Add a cheap in-PR proxy for Ragas if Gate 2 goes scheduled — a subset
      eval or a retrieval unit-test suite that asserts the fused top-k on a
      fixed query set, so regression is caught in seconds rather than nightly.
- [ ] Store `GROQ_API_KEY`, `VOYAGE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` as
      repo secrets. Never commit keys — see the AGENTS.md pre-push secret scan.
- [ ] Guard against the classic failure mode: upload `results.json` /
      `verify_results.json` as build artifacts so a failing run is inspectable
      without re-running.
- [ ] Document the required secrets in README so a fork can run the same gates.

## Acceptance

- [ ] `.github/workflows/ci.yml` has a deterministic citation gate that fails
      the build on a FAIL row
- [ ] Ragas runs on a schedule (or an agreed per-PR path) and publishes metrics
      as artifacts
- [ ] A deliberately broken citation is rejected by CI, proven by a test PR or
      a local dry run
- [ ] AGENTS.md updated — the "Eval is LOCAL-ONLY" line is now wrong
- [ ] Fork path documented in README

## Risk

Low for Gate 1, medium for Gate 2 — secret exposure and runaway API spend are
the real risks. Keep Gate 2 on a schedule with a concurrency group so a burst
of merges cannot stack runs. Note the free-tier query limit economics from
AGENTS.md before adding eval volume.
