"""Landing page — public marketing (docs/UI_DESIGN.md §5).

Custom HTML marketing build (Option 1): one st.html document styled by
styles/main.css `.lp-*` classes. CTAs are plain links — page routes strip
the numeric prefix (pages/2_Chat.py → /Chat).
No `src.*` imports here (AGENTS.md guardrail).
"""

import streamlit as st

from menu import menu
from ui_core import init_session_state, load_css, page_config

page_config("GroundedAI — Citation-enforced RAG")
init_session_state()
load_css()
menu()

st.html(
    """
<div class="lp-wrap">
  <div class="lp-hero">
    <span class="hero-badge">Legal &amp; academic · citation-enforced RAG</span>
    <h1>Ask your documents. Trust the answer.</h1>
    <p>A research assistant that retrieves from <strong>your</strong> PDFs and
       cites every sentence to its page — validated by code, not requested
       by prompts. Warm when it can help; honest when it can't.</p>
    <div class="lp-ctas">
      <a class="lp-cta lp-cta-primary" href="/Chat">Try the live demo</a>
      <a class="lp-cta lp-cta-ghost" href="/">Sign in</a>
    </div>
  </div>

  <div class="lp-stats">
    <div class="lp-stat"><strong>1.00</strong><span>Faithfulness (Ragas golden set)</span></div>
    <div class="lp-stat"><strong>0.88</strong><span>Answer relevancy vs 0.75 gate</span></div>
    <div class="lp-stat"><strong>Page-level</strong><span>Citations on every claim</span></div>
    <div class="lp-stat"><strong>100%</strong><span>Claims grounded or abstained</span></div>
  </div>

  <div class="lp-section">
    <h2>Why GroundedAI</h2>
    <div class="lp-grid">
      <div class="lp-card">
        <span class="lp-card-tag">Verify, don't guess</span>
        <h3>Citations enforced by schema</h3>
        <p>Every generated sentence must carry a valid <code>[SOURCE N]</code>
           marker. The validator rejects anything uncited before you see it —
           a code gate, not a prompt suggestion.</p>
      </div>
      <div class="lp-card">
        <span class="lp-card-tag">Honest by default</span>
        <h3>Abstains instead of fabricating</h3>
        <p>If your corpus doesn't cover the question, the assistant says so
           and points at what it does contain — completeness traded for
           accuracy.</p>
      </div>
      <div class="lp-card">
        <span class="lp-card-tag">Two workspaces, one engine</span>
        <h3>Legal &amp; academic profiles</h3>
        <p>Legal reports what statutes say — no advice, strict abstain.
           Academic answers from your papers with provenance. The engine is
           shared; only the profile changes.</p>
      </div>
      <div class="lp-card">
        <span class="lp-card-tag">Retrieval that finds it</span>
        <h3>Hybrid search + rerank</h3>
        <p>BM25 keywords fused with vector semantics (RRF), then a
           cross-encoder reranker picks the passages that actually answer
           you — measured with Langfuse traces.</p>
      </div>
    </div>
  </div>

  <div class="lp-section">
    <h2>How it works</h2>
    <div class="lp-steps">
      <div class="lp-step">
        <span class="lp-step-num">1</span>
        <h3>Upload a PDF</h3>
        <p>Page-aware parsing keeps every page number intact — your corpus
           stays tenant-isolated.</p>
      </div>
      <div class="lp-step">
        <span class="lp-step-num">2</span>
        <h3>Ask in plain language</h3>
        <p>Hybrid retrieval finds the passages; the assistant answers only
           from them, identifying sources first.</p>
      </div>
      <div class="lp-step">
        <span class="lp-step-num">3</span>
        <h3>Check the citation</h3>
        <p>Each claim links its <code>[SOURCE N]</code> to a page — click
           through, read the evidence yourself.</p>
      </div>
    </div>
  </div>

  <div class="lp-section">
    <h2>FAQ</h2>
    <div class="lp-faq">
      <details>
        <summary>How is this different from ChatGPT?</summary>
        <p>It retrieves from your documents only, cites every claim with page
           numbers, and refuses to answer when the corpus does not support
           it. Unsupported sentences are rejected by validation — not by
           hoping the model behaves.</p>
      </details>
      <details>
        <summary>What does "citation-enforced" mean?</summary>
        <p>Generated claims are checked deterministically: the quote must
           appear word-for-word in the cited chunk, and every sentence needs
           a valid source marker. Failures go through a repair loop; nothing
           unverified ships.</p>
      </details>
      <details>
        <summary>Is there a free tier?</summary>
        <p>Yes — Free ($0) covers the demo: 500 queries/day, 5 documents,
           community support. Pro ($9/mo) and Teams ($29/mo) are listed as
           previews; billing activates later.</p>
      </details>
      <details>
        <summary>Where does my data live?</summary>
        <p>ChromaDB vectors with tenant isolation — your documents are
           invisible to other users. Prefer full control? Self-host with one
           command: <code>streamlit run app.py</code>.</p>
      </details>
    </div>
  </div>

  <div class="lp-section">
    <h2>Pricing</h2>
    <div class="lp-price-grid">
      <div class="lp-price lp-price-featured">
        <h3>Free</h3>
        <div class="lp-price-amount">$0</div>
        <ul>
          <li>500 queries / day</li>
          <li>5 documents</li>
          <li>1 workspace</li>
          <li>Community support</li>
        </ul>
        <span class="lp-price-note">Live now</span>
      </div>
      <div class="lp-price">
        <h3>Pro</h3>
        <div class="lp-price-amount">$9<span style="font-size:0.9rem;font-weight:600;color:#64748b;"> / mo</span></div>
        <ul>
          <li>5,000 queries / day</li>
          <li>100 documents</li>
          <li>All workspaces</li>
          <li>Priority support</li>
        </ul>
        <span class="lp-price-note lp-price-note-muted">Coming soon</span>
      </div>
      <div class="lp-price">
        <h3>Teams</h3>
        <div class="lp-price-amount">$29<span style="font-size:0.9rem;font-weight:600;color:#64748b;"> / mo</span></div>
        <ul>
          <li>25,000 queries / day</li>
          <li>Unlimited documents</li>
          <li>Shared workspaces</li>
          <li>Admin + SSO</li>
        </ul>
        <span class="lp-price-note lp-price-note-muted">Coming soon</span>
      </div>
    </div>
  </div>

  <p class="lp-foot">
    Open source: Groq · LangChain · Chroma · Streamlit · Langfuse · Ragas
    &nbsp;·&nbsp;
    <a href="https://github.com/aieng-abdullah/production-rag-assistant"
       target="_blank" rel="noopener">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/aieng-abdullah/production-rag-assistant/issues"
       target="_blank" rel="noopener">Contact / issues</a>
  </p>
</div>
""",
)
