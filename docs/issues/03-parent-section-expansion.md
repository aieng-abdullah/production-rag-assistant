# Expand chunks to parent section / neighbours at answer time

**Labels:** `retrieval`, `legal`, `priority:high`

## Problem

Chunks are 256 characters with 100 overlap (`src/config.py:73-74`), and the
reranker hands 5 chunks to generation. `src/ingestion/` contains no notion of a
parent section, a heading, or a neighbouring chunk — grep for
`parent|neighbor|adjacent|section` returns nothing.

The split between *retrieval granularity* and *answer granularity* is wrong for
legal text. A clause that defines a term lives in one 256-char window; the
sentence that relies on it lives two windows away, because the window also
swallowed a heading, a table of contents line, and a page break. Retrieval
finds the definition. The answer never sees it.

This is the same root cause as the `context_precision` 0.375 failure: recall of
individual windows is fine, precision at answer time is not.

## Approach

Keep 256-char chunks for retrieval — they are good at lexical and vector
matching, and rewriting the chunker risks regressing every existing eval. Widen
only what reaches the prompt.

- [ ] **Section detection.** At ingest, stamp each chunk with its parent section
      identifier. PDF text already carries headings; derive them in
      `src/ingestion/` and persist in chunk `metadata` alongside the existing
      `tenant_id` / `doc_id` stamps.
- [ ] **Neighbour expansion.** When building the answer-time context, replace
      each retrieved chunk with its section or a bounded ±N neighbour window.
      `N` must be configurable — start at 1 and measure.
- [ ] **Token budget.** Widening 5 chunks can blow the prompt. Add an explicit
      context assembly step with a character or token ceiling and deterministic
      truncation order, so the widening cannot silently raise cost or hit
      provider limits.
- [ ] **Provenance must follow.** `AnswerTrace` (`src/api/answers.py`) records
      chunks retrieved / used / discussed, plus pinpoint page/section. Expanded
      context has to keep that accurate, or the trace becomes a lie.
- [ ] **Judge must see the same text.** `judge_claims()`
      (`src/generation/verifier.py:121`) receives `chunks` and indexes by
      `citation.source_id - 1`. Whatever transformation happens between retrieval
      and the prompt has to happen before the judge, or the judge validates
      against text the user never saw.
- [ ] **Quote verification still passes.** `verify_quotes()`
      (`src/generation/schema.py:82`) does normalised containment against the
      cited chunk. Widened text makes this easier, not harder — confirm it does
      not regress.
- [ ] **Section-aware retrieval also serves legal cross-reference following**
      (separate issue). Design the metadata so a later step can ask "give me
      section 158 of the Contract Act" without re-ingesting.

## Acceptance

- [ ] `context_precision` improves measurably over the refreshed baseline from
      the eval baseline issue, with both local eval gates green
- [ ] `verify_results.json` shows no regression — quote verification still 1.00
      precision
- [ ] Trace endpoint returns correct provenance for an expanded answer
- [ ] Prompt length stays within a configured budget; exceeding it logs and
      truncates rather than failing the request
- [ ] Ingestion time impact measured and recorded (this runs on every upload)
- [ ] Existing chunk metadata consumers (`vector_search`, `load_all_chunks`,
      `count_chunks`, BM25 rebuild) unaffected — they filter on `tenant_id`

## Risk

Changing what the judge and the user see is the sharp edge here. The judge
indexes `chunks[source_id - 1]`, so an assembly step inserted in the wrong place
silently verifies against the wrong text. Implement expansion as one function
that both the prompt builder and the judge call, not two parallel paths.

Do not re-ingest existing corpora without a migration note — section metadata
is absent on stored chunks, so old chunks need a re-ingest path or a
fallback that degrades to the current behaviour.
