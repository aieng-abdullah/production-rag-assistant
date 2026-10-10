# Legal cross-reference following (section refs and defined terms)

**Labels:** `legal`, `retrieval`, `agent`, `priority:medium`

## Problem

Legal text is full of internal pointers. Section 158 says *"subject to the
provisions of section 12"*. A definition in section 2 governs every later use
of the term. A proviso modifies the section it sits inside. The retrieval
system treats all of these as ordinary prose, so a question about a proviso
returns the section without the sub-section that qualifies it, and the answer
is confidently wrong.

The eval dataset already anticipates this class — `data/verify_eval.json`
contains `legal-cross-ref-158` (`class: missing-cross-ref`) and two
`legal-proviso-*` items. Those items currently pass, but they pass by luck of
corpus, not by mechanism. The Contract Act and Companies Act roadmap both need
this.

## Scope

- [ ] **Cross-reference detection.** Recognise section pointers in statute text:
      "section N", "sections N and M", "subject to section N", "notwithstanding
      section N", chapter references. This is a parsing concern over chunk text,
      not an LLM call — keep it deterministic and testable.
- [ ] **Fetch-and-cite.** When a cited chunk contains a cross-reference, fetch
      the referenced section and cite it. This must be a real retrieval step, so
      the fetch lands in the trace and in `AnswerTrace`.
- [ ] **Defined-term resolution.** Same mechanism for terms defined in a
      definitions section. A defined term needs a registry built at ingest, not
      a regex per query.
- [ ] **Recursion bound.** Cross-references chain. Hard cap on follow depth —
      one level is almost certainly right. A section that chains three deep is
      where cost and latency go to die.
- [ ] **Missing reference handling.** A reference to a section not in the corpus
      must degrade explicitly: log, and tell the user the referenced text is not
      available. Never silently drop it, and never invent the content.
- [ ] **Bounded rounds.** This composes with reflective retrieval (separate
      issue). Define how many total retrieval rounds cross-ref following may
      add on top of the reflection budget. Ship behind the same config caps.
- [ ] **Bundle with parent-section expansion.** Section metadata from the chunk
      expansion issue is the substrate here — "give me section 158" should be a
      metadata lookup, not a fresh search. Do the expansion issue first.

## Acceptance

- [ ] `legal-cross-ref-158` and both `legal-proviso-*` items in
      `data/verify_eval.json` verified by mechanism, with a test that fails if
      cross-reference following is disabled
- [ ] New eval items covering a dangling cross-reference and a two-level chain
- [ ] `verify_results.json` refreshed, all gates green
- [ ] Depth cap enforced by test
- [ ] Corpus-missing reference degrades with an explicit message, proven by test
- [ ] Latency impact measured and recorded

## Risk

Guardrail from AGENTS.md: *"Don't hardcode workspace-specific logic (legal vs
academic) into the pipeline — workspace profiles live in prompt config; the
engine stays shared."*

Cross-reference following is legal-domain behaviour. It belongs behind the
`legal` workspace profile, not baked into `src/retrieval/`. The academic
workspace should get an explicit no-op, not a silent one. A shared engine with
a domain-specific strategy is the shape to aim for.

Also: this is answer-time fetch, which means a user-visible latency increase on
exactly the queries where they care most about an answer.
