# Reflective retrieval — sufficiency check and one bounded re-retrieval round

**Labels:** `retrieval`, `agent`, `priority:medium`

## Problem

The pipeline is single-shot. `src/retrieval/pipeline.py` runs BM25 and vector
search, fuses with RRF, reranks, and hands the result straight to generation.
Nothing checks whether the retrieved set actually answers the question before
the LLM is called.

`src/generation/query_rewrite.py:65` exists and is worth being precise about: it
resolves pronouns in **multi-turn chat only** — "what about the second one?" —
and returns early when history is empty, so stateless requests pay nothing. It
is not a reflection loop and there is no sufficiency judgement anywhere.

The failure mode is visible in the committed eval output. A single-pass system
that retrieved badly produces a confident answer from the wrong chunks. The
judge then marks those claims `UNSUPPORTED` and the answer ships with an
`unverified` badge — correct behaviour, but the user got a bad answer when a
second retrieval pass would have found the right chunk. Every unverified
answer is a candidate retrieval miss.

## Latency is the binding constraint

README performance table:

```
Full request   p50 1.54s   p90 7.32s   p95 11.09s   p99 14.06s
```

Each loop iteration multiplies the tail. `src/config.py:56` sets
`RERANK_TIMEOUT_S = 10`, so a second round can add a full rerank cycle on the
worst path.

AGENTS.md records the reranker as the historical bottleneck (72% of latency
before the move to Voyage `rerank-3-lite`). Reranker batching or a narrower
candidate pool per round is a prerequisite for shipping this safely, not a
follow-up.

## Scope

- [ ] **Sufficiency judge.** After fusion + rerank, a cheap call decides whether
      the retrieved set answers the query. Reuse the `Config.VERIFY_MODEL` tier
      (`src/config.py:62`) — same reasoning as `verifier.py:6`: a different
      model family from the generator so it is not grading its own homework.
- [ ] **Reformulate + re-retrieve.** On insufficiency, reformulate the query
      and retrieve once more. Hard cap **1** extra round initially, in config.
      Cap is not a suggestion — it is the latency guarantee.
- [ ] **Merge strategy.** Decide how round-two results combine with round one.
      Simple re-rank of the union is preferred over anything clever.
- [ ] **Never fail the request.** The sufficiency judge failing must degrade to
      "proceed with what we have", matching the existing pattern in
      `verifier.py:139-147` where a judge outage marks claims unverified and the
      answer still ships. Log at WARNING with tenant and reason.
- [ ] **Query decomposition: separate issue, not here.** Cross-paper synthesis
      only, not every query. Decomposition on every query would multiply cost
      for the common case.
- [ ] **Quota economics.** Verification already costs 2 units per answer
      (`src/api/chat.py:6`). A reflection round adds retrieval calls — check
      whether that is charged and whether the free-tier `DAILY_QUERY_LIMIT`
      (default 20) still makes sense.
- [ ] **Trace the loop.** `AnswerTrace` records queries issued and chunks
      retrieved. A second round must show up in the trace, otherwise the
      observability story regresses.
- [ ] **Reranker optimisation or SSE streaming lands alongside.** The UI plan
      already anticipates "searching again..." — without it a silent 20-second
      wait is a support problem.

## Acceptance

- [ ] Latency measured on a warm run: p95 increase documented and accepted, not
      assumed
- [ ] `results.json` shows `answer_relevancy` or `context_precision` improving,
      with all four gates green
- [ ] A test proves the cap holds — 3 rounds is impossible
- [ ] A test proves judge outage degrades rather than blocks
- [ ] Trace shows both rounds with distinct queries
- [ ] Cost per query measured against free-tier limits before merge

## Risk

This is the change most likely to blow the latency budget and the most likely to
be judged on the README's own numbers. Land the reranker optimisation first, or
ship behind a flag and measure both arms on the eval set before enabling.
