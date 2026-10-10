# Refresh eval baseline and fix context precision (0.375 vs 0.70 gate)

**Labels:** `eval`, `retrieval`, `priority:high`

## Problem

`results.json` is stale. The dataset was expanded to 29 items in `6d680c8`
(`test: expand ragas eval dataset`), but the committed baseline still contains
**5 samples** — so every metric in it was measured against a subset that no
longer exists.

```
$ python3 -c "import json;print(len(json.load(open('data/eval_dataset.json'))))"
29
$ python3 -c "import json;print(len(json.load(open('results.json'))['samples']))"
5
```

Current committed numbers:

| metric | score | threshold | status |
| --- | --- | --- | --- |
| faithfulness | 1.00 | 0.75 | PASS |
| answer_relevancy | 0.878 | 0.75 | PASS |
| context_precision | **0.375** | **0.70** | **FAIL** |
| context_recall | 1.00 | 0.70 | PASS |

`context_precision` is the one gate that fails, and it is failing by 46%. It
also means `faithfulness: 1.00` is not trustworthy — it was measured on 5
samples, and Ragas faithfulness only inspects the context that was actually
retrieved, so junk context inflates precision rather than exposing itself.

Every downstream retrieval change (chunk expansion, reflective loops,
cross-reference following) is unmeasurable until this baseline is real.

## Why precision is low

`src/config.py:73-77`:

```
CHUNK_SIZE = 256
CHUNK_OVERLAP = 100
TOP_K_RERANK = 8
```

256-character chunks with a top-5 handoff into generation. For legal text a
clause depends on the surrounding section, so the chunk holding the actual
definition often does not contain the words the query matched. BM25 and vector
search both match on lexical overlap inside a 256-char window; the reranker
(`rerank-3-lite`, `src/retrieval/reranker.py:1`) can only reorder what
retrieval already handed it.

Note the legacy boilerplate filter (`src/retrieval/boilerplate.py:5`) already
exists — it drops chunks that are pure term-frequency noise. That is the right
shape of fix, just applied too late (post-retrieval) rather than at chunking.

## Scope

- [ ] Re-run `python3 eval/eval_runner.py` against the full 29-item dataset and
      commit the refreshed `results.json`. Record the true faithfulness /
      answer_relevancy numbers before touching retrieval — they may not be 1.00
      / 0.878 at n=29.
- [ ] Identify which samples drive precision down. `eval/eval_runner.py:323`
      `per_sample_report()` already emits per-sample rows — add a per-sample
      context_precision column so the failing items are visible by name.
- [ ] Fix precision to ≥0.70 on the refreshed baseline. Likely levers, in
      order of preference:
  - tighten the chunk-size / top-k tradeoff as a config sweep, measured not
    guessed (`CHUNK_SIZE`, `TOP_K_RERANK` are already env-overridable in
    principle — confirm they are actually read from env)
  - extend the existing boilerplate filter to catch real junk rather than
    keyword-stuffed headers
  - section-aware chunking — see the separate parent-section expansion issue
- [ ] Keep both local gates green: `eval/eval_runner.py` and
      `eval/verify_eval.py`, and refresh `verify_results.json` too.

## Acceptance

- [ ] `results.json` contains 29 samples
- [ ] `python3 eval/eval_runner.py` exits 0 — no FAIL rows
- [ ] `context_precision >= 0.70`
- [ ] `python3 eval/verify_eval.py` exits 0, `verify_results.json` refreshed
- [ ] `ruff check src tests eval alembic` clean
- [ ] `pytest tests/ -v -m "not slow" --cov-fail-under=70` green

## Risk

Chunk-size changes alter every retrieval result, so this invalidates the
committed baseline by construction. That is the point — but it also means any
pull request touching `CHUNK_SIZE` must re-run both evals and commit new
results, per the AGENTS.md quality-gate contract. Rollback is reverting the
config plus the two results files together.

## Notes

`eval/eval_runner.py:53` hardcodes `THRESHOLDS`. Consider moving them to a
config block or env so the precision gate can be ratcheted as it improves
rather than rewritten in a commit.
