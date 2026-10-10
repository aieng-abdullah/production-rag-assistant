# Workflow agents — structured extraction with per-field citations

**Labels:** `agent`, `legal`, `priority:low`

## Problem

Answering a question produces prose. A contract review needs a *record*: who
signed, what each party owes, when, under which clause. The prose path cannot
serve that, and forcing prose to carry it means the citation model has to be
redesigned for a job it is not shaped for.

`src/generation/schema.py` already has the substrate — a Pydantic structured
output (`claims[]` with `citations[]` of `source_id` + `quote`), an abstain
flag, and deterministic quote verification. A workflow agent is the same
mechanism with a domain schema instead of a claims schema.

## Scope

- [ ] **Contract review extraction.** Fields: party, obligation, effective date,
      clause reference. Every field carries its own citation with a verbatim
      quote — not a single document-level citation for the whole record.
- [ ] **Abstain per field, not per document.** A contract with no effective date
      yields `effective_date: null` with a reason. The existing model is
      document-level (`StructuredAnswer.abstained`); this needs field-level
      abstain or the whole record degrades because one field is missing.
- [ ] **Compliance check.** Same shape, different field set: requirement,
      status (met / not met / not determinable), evidence quote.
- [ ] **Corpus-bounded.** Extraction reads only the user's uploaded documents.
      No web browsing, no external lookup — this preserves the *"only your
      documents"* guarantee that the README sells. Anything beyond the corpus
      must abstain with an explicit reason.
- [ ] **Reuse the verifier.** `judge_claims()`
      (`src/generation/verifier.py:121`) judges claim-plus-evidence. A field
      value plus its quote is the same shape. Do not build a second verification
      path.
- [ ] **Quotas.** Verification costs 2 units today (`src/api/chat.py:6`). A
      20-field extraction verified field-by-field is a different cost class.
      Decide and document the charging model before merge.
- [ ] **Response shape.** New endpoint or a mode on `/chat`? Adding a second
      answer type to the existing response breaks the frontend contract — decide
      before writing code.
- [ ] **Frontend.** `TraceModal` and the citation expanders already render
      per-claim verdicts. Check whether a field/claim grid reuses them or needs
      new components.

## Acceptance

- [ ] At least 3 golden documents extracted with per-field citations
- [ ] Every extracted field's quote verifies via the deterministic
      containment check — no field ships an uncited value
- [ ] Field-level abstention proven by test on a document missing a field
- [ ] Out-of-corpus request abstains rather than reaching externally
- [ ] Quota math documented and tested
- [ ] Both local eval gates green

## Risk

**Deliberately sequenced last.** This is the highest per-query cost of anything
proposed, on the same engine that already fails `context_precision`. Extraction
against badly-retrieved chunks produces confidently-cited garbage — the worst
possible failure mode for a compliance tool, because a bad citation looks
exactly as trustworthy as a good one.

Ship it after the eval baseline, CI gates, and parent-section expansion are
done. The measure that matters for a compliance record is not coverage, it is
how often a wrong value ships with a valid-looking citation.

AGENTS.md guardrail: *"Don't polish the UI before the underlying
service/pipeline is proven by tests — service layer first."*
