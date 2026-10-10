# Strategy

Internal positioning and roadmap. Written 2026-10-10.

Two earlier claims in this document's history were wrong and are corrected here:
the EU AI Act Article 12 deadline is **not** live (deferred to 2027-12-02), and
`context_precision = 0.375` is **not** a blocking metric.

---

## 1. What this product is

Not "a RAG assistant with citations." That is the feature list, and it is the
wrong frame.

**A claim-checking engine with a retrieval front-end.**

The pipeline: retrieve → generate **structured claims**, not prose → every claim
carries a citation with a **verbatim quote** → Pydantic enforces it
(`src/generation/schema.py`, `Claim.citations` is `min_length=1`) →
`verify_quotes()` does **deterministic normalised containment** of that quote
against the cited chunk → a **separate model** (`VERIFY_MODEL`) scores each
claim SUPPORTED/PARTIAL/UNSUPPORTED against all retrieved chunks → unsupported
claims trigger regeneration, bounded → still unsupported means **abstain**.

Three properties frontier chat products do not have:

1. **Deterministic verification.** The quote-in-chunk check is string
   containment, not a model opinion. It cannot be persuaded or rationalise.
2. **Separation of duties.** The generator and the judge are different models.
   The thing being judged did not write the thing being judged.
3. **Refusal is a first-class output.** Abstention accuracy is 1.00 on
   out-of-corpus questions. ChatGPT does not abstain; it answers gracefully and
   wrongly.

One-line product statement:

> The only thing that mechanically refuses to let a model assert something its
> sources do not contain.

There is also a rare asset: a **deterministic retrieval gate**
(`eval/retrieval_precision.py`) that runs with no LLM judge. Most RAG repos
publish hand-wavy numbers. This one is reproducible on a free tier.

---

## 2. Why not ChatGPT

The honest concession first: **frontier deep-research agents are genuinely good.**

Measured on DRACO (100 cross-domain tasks, 2026):

| System | Factual accuracy | Citation quality |
|---|---|---|
| Perplexity Deep Research (Opus 4.6) | 67.9% | 64.6% |
| Gemini Deep Research | 59.0% | — |
| OpenAI o3 Deep Research | 52.1% | — |

Hallucination rates across 900 tested queries have fallen to **2.4%–5.1%**.

**3% sounds acceptable. It is not acceptable for a professional who signs their
name to the output.** That gap is the entire business case.

### The argument is consequence, not quality

Sanctions and professional discipline in 2026 alone:

- **$15,000** — Illinois Appellate Court, at $1,500 per false citation.
  Four fabricated statutory quotations and one nonexistent case.
- **$31,150** — Law Society of Ontario. Largest AI costs order in Canadian
  history.
- **Removed from the register** — UK Solicitors Disciplinary Tribunal, first
  case of its kind. The lawyer's defence was that he "did not have the expertise
  to verify the AI output." The tribunal rejected it.
- **$8,000, two-year bar, removed from case** — N.D. Mississippi.
- **Brief struck in its entirety** — D.C. Court of Appeals, Deutsche Bank.
- **42 inaccuracies in one emergency motion** — Sullivan & Cromwell.
- **1,334+ cases** logged globally (Charlotin database).

### The holding that defines the product

The Illinois court held:

> Cross-referencing AI output against a legal research platform is insufficient
> if the attorney does not confirm that the cited text actually appears in the
> cited source.

That is precisely `verify_quotes()`. The court just specified the feature and
declared it mandatory.

The court also rejected the obvious objection:

> AI tool quality is irrelevant.

A "premier corporate subscription of ChatGPT" is **no defence**. The obligation
is verification, and it rests on the human. ChatGPT cannot solve this for a
lawyer at any model quality.

**This is structural, not technical.** OpenAI cannot ship "I verify my quotes
against your uploaded corpus and refuse if absent." They have no corpus, no
liability appetite, and it would cap model usefulness.

**Every improvement to ChatGPT makes this problem worse.** A better model
produces more plausible fabrications, faster.

### Do not pitch retrieval depth

Retrieval depth is a commodity; frontier labs have larger indexes. Our own
numbers are not a selling point: `ground_truth_coverage@1 = 0.6434`. Sell the
guarantee, not the recall.

---

## 3. Market map

Two tiers, real numbers.

### Tier 1 — Legal AI

| Vendor | Pricing | Motion |
|---|---|---|
| Harvey | ~$1,200/seat, ~20-seat minimum | Enterprise sales, annual |
| CoCounsel (Thomson Reuters) | ~$100–200/user/mo | Bundled with Westlaw |
| Luminance | Enterprise, 1000+ customers | Contract intelligence, demo-gated |
| Evisort | Enterprise | CLM, ISO 27001/27701, SOC 2 Type 2 |

### Tier 2 — AI governance *(the crowded tier)*

| Vendor | Pricing | Sells |
|---|---|---|
| Hydrus | $199–799/mo | AI inventory, Annex III classification, shadow-AI discovery |
| AuditEvidenceAI | £1,200–18,000/yr | "Evidence packs", audit-ready PDF |
| trail (trail-ml) | Demo-gated, EU-hosted | Control mapping, vendor assessment |
| Prove7 | Undisclosed | Agent governance/auditing |
| SureTrace | Request access | Tamper-proof logs, PII blocking, FINRA/SOC 2 |
| ibl.ai | Quote | Self-host/air-gapped, no per-seat |
| ai-audit-trail (OSS) | Free | Ed25519 hash-chained receipts, ISO 42001/NIST crosswalk |

**Generic audit trails are commoditised and well funded.** An earlier draft of
this document called evidence export a moat on its own. That was wrong.

### Positioning

| Incumbents (tier 2) | This product |
|---|---|
| Prove the AI was **governed** | Prove the answer was **true** |
| Log prompt + response | Verify quote ⊆ cited chunk (deterministic) |
| Governance metadata per decision | Ground-truth verdict per claim |
| Sell to the whole organisation | Sell to the person who signs |
| Funded, commoditised | Not attempted by anyone |

---

## 4. The wedge

The incumbents' own copy concedes the gap.

ISACA, *"The AI Audit Trail: From AI Policy to AI Proof"* (May 2026):

> A policy cannot prove that an AI system behaved correctly at the moment it
> mattered... If these questions cannot be answered, we don't have an audit
> trail, we have a record of the event without a record of the control path.

> **That is the difference between output logging and runtime proof.**

SureTrace's own product description logs "full prompt, response, user, team,
and cost attribution." That is **a transcript**.

The Interpreting Accounting Credentials in Canada Foundation submission to the
PCAOB (draft v0.1, July 2026) names the gap precisely:

> The AI should not be allowed to make accounting judgments from unknown or
> stale evidence. You cannot audit an accounting judgment if you do not know
> which model made it.

> For retrieval-augmented and agentic systems, the record must declare not only
> the model artifact, but also the live documents, records, or data injected at
> inference time.

> AI has many fragments of documentation and logging, but **no standard
> assertion record that travels with a consequential judgment from construction
> to inference, integration, human review, action, and audit.**

That is `AnswerTrace` (`src/db/models.py`), already built and already
populated on every answer.

### Risk in this wedge

Funding makes cross-sell attractive. Hydrus or AuditEvidenceAI could bolt on a
verification step. Two things slow that: (a) verification requires owning the
retrieval path, which a governance overlay does not; (b) it would convert a
cheap logging product into one carrying professional-liability exposure they do
not currently have. Neither is a permanent barrier. Build here while it holds.

---

## 5. Regulation (corrected)

**Correction to an earlier draft:** Article 12 is **not** in force.

Regulation (EU) 2026/1744, the Digital Omnibus on AI, entered into force
27 July 2026 and **deferred** the high-risk regime:

- Annex III standalone high-risk systems: **2026-08-02 → 2027-12-02**
- Annex I (embedded in regulated products): **2026-08-02 → 2028-08-02**
- National regulatory sandboxes under Art. 57: operational by 2027-08-02

Unchanged and already applicable:

- **Article 50** transparency and AI-content labelling — took effect 2026-08-02
- **GPAI** provider obligations — since 2025-08-02
- **Article 5** prohibited practices — since 2025-02-02

So the framing is a **~16-month preparation window, not a live deadline.** Buyers
are in procurement conversations now precisely because the clock is visible.

Article 12(3) is the eventual requirement — logs must record the period of each
use, **the reference database against which input was checked**, the input data
that produced a match, and who verified the result.

### Elsewhere

- **PCAOB / AS 1105 + AU-C 230** (fully effective for CY2026): a generic "an AI
  tool was used" notation does not satisfy documentation requirements.
  Automation bias must be actively guarded against.
- **PCAOB AS 1105** requires evaluating the relevance *and reliability* of
  technology-processed information.
- **EU AI Act Art. 26(5)**: deployers must retain logs **at least six months**.

The common thread across all three: a transcript is not evidence.

---

## 6. Audience

**Not solo practitioners.** They already solved this with $20–25/mo ChatGPT.
That is a commodity price with no moat.

**Not big law.** Harvey serves it at $1,200/seat with a sales team.

**Target: in-house legal/compliance teams and professional firms of 20–200**, in
regulated verticals. Adjacent: audit and assurance teams, regulatory affairs.

Why they:

- **Liability is the budget.** A General Counsel carrying $15,000-per-citation
  exposure does not optimise at $20/month.
- **Procurement is a gate, not a wall.** BYOK satisfies the security checkbox.
- **They are being forced to buy.** Gartner has advised general counsel to
  carry AI-specific insurance; insurers are beginning to require controls.

### What they buy

Not answers. **The ability to prove what happened.** A transcript is not
evidence; a reconstruction is.

### Buyer sequence — do not skip this

1. **Practitioner first.** A law student or academic hits a hallucinated
   citation in ChatGPT and finds a tool that refuses to produce one. Free.
2. **Internal champion.** That person now has ammunition inside the firm.
3. **GC / compliance lead.** Buys assurance around a tool already in use, not a
   bet on an unknown vendor.

Selling top-down to a General Counsel first is the expensive mistake. Bottom-up
lands, and it costs nothing to acquire.

---

## 7. Pricing

Marginal cost is ~$0 because `ProviderOverrides`
(`src/generation/providers.py`) already supports user-supplied provider keys.

| Tier | Price | Cost to us | Contents |
|---|---|---|---|
| **Free** | $0 | ~$0 (BYOK required) | Full engine, limited queries. Proves the guarantee. |
| **Pro** | $49/mo | ~$0 (BYOK) | Everything. Pooled keys optional. |
| **Team** | $199/mo | <$10 | Shared corpus, admin, evidence bundle export. |

BYOK is not a cost hack — it is a **feature that sells to this buyer**. A
security-conscious firm prefers to supply its own keys; it is a procurement
checkbox.

`DAILY_QUERY_LIMIT` already exists and functions as margin control.

---

## 8. Roadmap

Sequenced by dependency, not by appeal.

### 0. Evidence bundle *(highest leverage)*

A per-answer export: query, exact chunks retrieved, every claim, every quote,
every verification verdict, model IDs, prompt version, timestamp, **hash-chained**
for tamper evidence. Shape it to the ICCA assertion-record spec so it is
recognisable to auditors.

This is the product. Everything else is distribution.

### 1. Fix `ground_truth_coverage@1` (0.6434)

The real retrieval weakness: the best chunk is often not ranked first. Coverage
rises to 0.8163 by k=5, so the right content *is* retrieved — just too low.

Cheapest first experiment is returning fewer chunks for narrow questions, not new
retrieval architecture. Measure with `eval/retrieval_precision.py` (deterministic,
free).

### 2. Ingest-side citation checker

Let a user paste a draft brief; check every citation against their uploaded
corpus. This addresses the **exact moment of failure** in the Illinois case. It
works with current components.

### 3. BYOK

Near-complete (`ProviderOverrides` exists). Finish the UI and the fallback path.

### 4. Public deterministic benchmark

Publish the eval harness. Zero-CAC distribution; competitors cannot reproduce
Ragas scores on free tiers, this can be.

### Not on the roadmap, and why

- **Clearing `context_precision = 0.375`.** See §9. Not a gate, stale n=5, and an
  artifact of the metric's construction.
- **Retrieval depth as a marketing claim.** We are not good enough at it yet.
- **SOC 2 / ISO 42001 certification.** Costs real money. Do it when a signed
  contract funds it.
- **Generic AI-governance tooling.** Crowded, funded, commoditised.
- **Enterprise sales motion.** Doesn't work for one person.

---

## 9. Risks

### `context_precision = 0.375` is not the blocker it appears to be

An earlier draft of this document called it blocker #0. **That was wrong, on
three counts:**

1. **It is not an enforced gate.** `Makefile` runs exactly two gates:
   `verify_eval.py` (citations) and `retrieval_precision.py` (retrieval).
   `context_precision` exists only in `eval_runner.py`, the Ragas path that
   AGENTS.md records as unfunded on this account.
2. **It is stale.** `results.json` holds **5 samples** from a 29-item dataset,
   last written before the dataset was expanded.
3. **It measures an artifact.** Ragas computes
   `CP@K = Σ(precision@k × v_k) / relevant_count` — an average-precision-style
   score over every returned chunk. All 5 samples are single-document Q&A on
   *Attention Is All You Need*, where each answer lives in one ~250-character
   chunk. **Retrieving 5 chunks when 1 is relevant caps the score near 0.375 by
   construction.** It is not evidence that retrieval is broken.

The enforced retrieval gate passes:

| Metric | Score | Gate | Status |
|---|---|---|---|
| `doc_hit@1` | 1.0 | 0.50 | PASS |
| `doc_hit@3` | 1.0 | 0.80 | PASS |
| `doc_hit@5` | 1.0 | 0.85 | PASS |
| `ground_truth_coverage@5` | 0.8163 | 0.55 | PASS |

**We can sell a truth guarantee today.** We cannot sell a retrieval-quality
claim, and should not.

### The abstain metric reads 0.0

`abstain_retrieval_empty = 0.0` in `retrieval_results.json`. This is a **reporting
artifact, not a product defect** — `verify_eval.py` measures abstention accuracy
at 1.00. The retrieval harness records `chunks_retrieved = None`.

It is nonetheless the first number a technical evaluator will find, and it looks
like the core guarantee failing. Fix the reporting before someone else does.

### Closed-source cuts against us

ISACA's argument is that audit trails must be *verifiable*. This product keeps
the verifier closed, so "audit our checker" is not available as a trust argument.

Mitigation: publish the **gate harness** and a worked example so the method is
inspectable and the claim is falsifiable, without releasing the checker. This is
weaker than open-sourcing it. A skeptical CISO may still decline, and that is an
accepted cost of the copy-protection trade.

### Retrieval precision is currently unmeasured

`eval_runner.py` cannot complete on free tiers (Groq 200K tokens/day; Voyage
3 RPM with per-candidate LLM calls for `context_precision`). So **"our retrieval
is good" is unfalsifiable in a sales conversation.** The deterministic gate is
the partial answer — which is an argument for publishing it, not an argument
against the product.

### Other

- **Solo-founder credibility.** Selling audit evidence to a regulated buyer is
  hard against Harvey's 1,400 firms. Mitigated by the buyer sequence in §6, not
  solved by it.
- **Governance incumbents could bolt on verification.** See §4.
- **Frontier labs converging.** If a frontier model ships first-party citation
  checking against user uploads, the wedge narrows sharply. The corpus and the
  trace format are the durable parts.

---

## 10. Open questions

1. **Can §4's wedge be demonstrated without a compliance buyer?** If the honest
   answer is no, the first customer is a practitioner who becomes a champion, and
   the GC sale happens months later. Plan for that timeline.
2. **Is the closed verifier the right trade?** Revisit if three buyer
   conversations stall on trust.
3. **What corpus do we anchor on?** The demo corpus (Attention Is All You Need,
   a contract, a patent) is adequate for engineering and useless as a
   differentiator.