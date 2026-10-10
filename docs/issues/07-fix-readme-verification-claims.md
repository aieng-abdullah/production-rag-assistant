# Fix README and PLAN.md claims — validator does not verify against source

**Labels:** `docs`, `priority:medium`

## Problem

The README makes claims the code does not support, and the gap is exactly on
the product's core differentiator.

`README.md:44` — the hero line:

> ChatGPT guesses. We verify — **every sentence, against the source**, before
> you see it.

`README.md:23`:

> every sentence cites its page-level source, validated before you see it

`README.md:59`:

> Pydantic checks **every sentence** for valid `[SOURCE N]`; violations
> rejected

`README.md:83` and `README.md:289` repeat it. `docs/PLAN.md:3` carries the same
line as the project thesis.

## What the code actually does

Stronger than the README describes in one respect, weaker in another.

**Stronger — the prose validator is retired.** `src/generation/Citation_system.py:5`:

```
per-sentence prose regex validator is retired — structured claims are
validated by `src/generation.schema` (schema + deterministic quote checks).
```

There is no per-sentence regex gate. Every claim must carry at least one
citation (`Claim.citations` is `min_length=1`, `schema.py:49`), each with a
verbatim quote, and `verify_quotes()` (`schema.py:82`) does normalised
containment of that quote in the cited chunk. `judge_claims()`
(`verifier.py:121`) then scores entailment as `SUPPORTED / PARTIAL /
UNSUPPORTED`, bounded at 2 retries.

So the "every sentence" phrasing describes a mechanism that no longer exists.

**Weaker — UNSUPPORTED still ships.** `build_verification()`
(`verifier.py:157`) returns `status: "unverified"`, and the answer renders with
a badge. It is not blocked. A user who does not read the badge receives an
unsupported claim that carries a real-looking citation.

Faithfulness 1.00 does not close this. Ragas measures against the context that
was retrieved, so it tests claim-vs-retrieved-context, not claim-vs-corpus. And
`results.json` covers 5 of 29 dataset items — see the eval baseline issue.

## Scope

- [ ] Rewrite the hero line and the pipeline/validation table rows to describe
      the structured-claim mechanism that actually runs
- [ ] State plainly what happens on `UNSUPPORTED`: ships with a badge, not
      blocked. If that is the intended product behaviour, say so. If it should
      block, that is a code change, not a docs change — file it separately.
- [ ] Replace "every sentence, against the source" with wording the
      implementation supports. Candidate: "every claim carries a verbatim quote,
      checked against its source, and scored by a separate judge model."
- [ ] Fix `docs/PLAN.md:3` thesis line to match
- [ ] Note the model tiers where relevant — the judge is a *different model*
      from the generator (`verifier.py:6`), which is the strongest part of this
      story and currently unmentioned

## Acceptance

- [ ] No README or PLAN.md line claims a check the code does not perform
- [ ] The `UNSUPPORTED` path is described accurately
- [ ] `git diff` reviewable without needing the source open

## Risk

Docs-only, no runtime blast radius. But this is the project's primary
differentiator claim — an overclaim here is the kind of thing that turns into a
trust problem the first time a user gets a bad answer with a valid-looking
citation. Worth getting exactly right rather than optimistically.
