import { Link } from "react-router-dom";
import CitationDemo from "../components/CitationDemo";
import HealthBadge from "../components/HealthBadge";
import CountUp from "../components/CountUp";
import Reveal from "../components/Reveal";

const FEATURES: [string, string, string][] = [
  [
    "Proven, not promised",
    "Citations enforced by code",
    "Every sentence must carry a valid [SOURCE N] marker, and the quote must appear word for word in that chunk. Failures go through a repair loop; nothing unverified ships.",
  ],
  [
    "Honest by default",
    "It refuses when your corpus is silent",
    "Ask what the documents do not cover and the assistant abstains on the record instead of inventing an answer. Completeness traded for accuracy.",
  ],
  [
    "Evidence, not vibes",
    "Page-level provenance on every claim",
    "Each citation resolves to a document and a page number, so a reader can open the original and check the words themselves.",
  ],
  [
    "Retrieval that finds it",
    "Hybrid search, fused, then reranked",
    "BM25 keywords and vector semantics fuse with Reciprocal Rank Fusion, then a reranker picks the passages that actually answer you.",
  ],
  [
    "One engine, two profiles",
    "Built for legal and academic work",
    "The legal profile reports what statutes say and never gives advice. The academic profile answers from your papers with provenance. Same engine, tuned voice.",
  ],
  [
    "Yours to keep",
    "Tenant-isolated and self-hostable",
    "Documents live in your tenant, invisible to other users. Run the whole stack on your machine with one docker compose command.",
  ],
];

const STEPS: [string, string][] = [
  [
    "Upload a PDF",
    "Page-aware parsing keeps every page number intact, so citations stay precise and your corpus stays tenant-isolated.",
  ],
  [
    "Ask in plain language",
    "Hybrid retrieval finds the passages, the reranker orders them, and the assistant answers only from what it found.",
  ],
  [
    "Check the citation",
    "Every claim links its [SOURCE N] to a page. Click through and read the evidence yourself.",
  ],
];

const FAQ: [string, string][] = [
  [
    "What exactly is GroundedAI?",
    "A citation-enforced research assistant. You upload your PDFs, ask questions in plain language, and it answers from those documents only, citing every sentence to a page. The enforcement is the product: claims are validated by code before you see them.",
  ],
  [
    "How is this different from ChatGPT?",
    "It answers only from the documents you upload, cites every claim with a page number, and refuses when the corpus does not support the answer. Unsupported sentences are rejected by validation, not by hoping the model behaves.",
  ],
  [
    'What does "citation-enforced" mean?',
    "Claims are checked deterministically: the quote must appear word for word in the cited chunk, and every sentence needs a valid source marker. Failures trigger a repair loop. Nothing unverified reaches you.",
  ],
  [
    "How accurate is it?",
    "On the Ragas golden set the assistant scores 1.00 faithfulness and 0.88 answer relevancy against a 0.75 gate. Every release is re-evaluated against those gates before it ships.",
  ],
  [
    "How fast are answers?",
    "Retrieval and reranking take about a second, and the language model streams the rest. Typical cited answers land in three to six seconds.",
  ],
  [
    "Is there a free tier?",
    "Yes. Guests get 3 questions and 1 document with no signup. The Free plan ($0) includes 10 verified answers a day, 5 documents, and 100 MB of storage. Pro ($9/mo) multiplies every quota by ten. Teams ($29/mo) is listed as a preview; billing activates when Stripe goes live.",
  ],
  [
    "Where does my data live?",
    "Documents are parsed and embedded into a tenant-isolated vector store on the host you use. Embedding requests go to Voyage AI and answer generation goes to Groq; your content is never used for training. Prefer full control? Self host with docker compose up.",
  ],
  [
    "Who is this for?",
    "Lawers reviewing statutes and contracts, and researchers working through papers. Anyone who needs answers they can defend with a page number.",
  ],
];

const PRICING: {
  name: string;
  amount: string;
  unit?: string;
  features: string[];
  note: string;
  featured?: boolean;
}[] = [
  {
    name: "Free",
    amount: "$0",
    features: ["10 answers / day", "5 documents", "1 workspace", "Community support"],
    note: "Live now",
    featured: true,
  },
  {
    name: "Pro",
    amount: "$9",
    unit: " / mo",
    features: ["100 answers / day", "50 documents", "All workspaces", "Priority support"],
    note: "Coming soon",
  },
  {
    name: "Teams",
    amount: "$29",
    unit: " / mo",
    features: ["25,000 queries / day", "Unlimited documents", "Shared workspaces", "Admin + SSO"],
    note: "Coming soon",
  },
];


const UPCOMING: [string, string, string][] = [
  [
    "Coming soon",
    "Medical domain",
    "A clinical profile tuned for records, guidelines, and literature, with strict abstain on dosage and diagnosis questions.",
  ],
  [
    "Coming soon",
    "Finance & compliance",
    "Filings, policies, and audit trails answered with clause-level citations and reviewer-friendly provenance.",
  ],
  [
    "On the roadmap",
    "Developer API",
    "The same citation-enforced pipeline as a documented HTTP API for embedding into your own tools.",
  ],
];

export default function Landing() {
  return (
    <main className="shell">
      <section className="hero">
        <Reveal>
          <span className="chip">Legal &amp; academic · citation-enforced RAG</span>
        </Reveal>
        <Reveal delay={90}>
          <h1>
            Your research assistant,
            <br />
            <em>with the proof built in.</em>
          </h1>
        </Reveal>
        <Reveal delay={180}>
          <p>
            GroundedAI is a <strong>citation-enforced research assistant</strong>{" "}
            for legal and academic work. It retrieves from{" "}
            <strong>your</strong> PDFs, answers in plain language, and backs
            every sentence with its page. Warm when it can help; honest when
            it can't.
          </p>
        </Reveal>
        <Reveal delay={260}>
          <div className="hero-ctas">
            <Link className="btn btn-primary" to="/login">
              Try it
            </Link>
            <Link className="btn btn-ghost" to="/login">
              Sign in
            </Link>
          </div>
        </Reveal>
      </section>

      <Reveal>
        <section className="demo-section">
          <div className="section-eyebrow">
            <span>See it work</span>
          </div>
          <h2>The assistant answers. The proof comes attached.</h2>
          <CitationDemo />
        </section>
      </Reveal>

      <Reveal>
        <section className="stats">
          <div className="stat">
            <strong>
              <CountUp value={1} decimals={2} />
            </strong>
            <span>Faithfulness (Ragas)</span>
          </div>
          <div className="stat">
            <strong>
              <CountUp value={0.88} decimals={2} />
            </strong>
            <span>Relevancy vs 0.75 gate</span>
          </div>
          <div className="stat">
            <strong>
              <CountUp value={100} suffix="%" />
            </strong>
            <span>Claims grounded or abstained</span>
          </div>
          <div className="stat">
            <strong>Page-level</strong>
            <span>Citations on every claim</span>
          </div>
        </section>
      </Reveal>

      <section className="section">
        <Reveal>
          <h2>Why GroundedAI</h2>
        </Reveal>
        <div className="grid">
          {FEATURES.map(([tag, title, body], index) => (
            <Reveal key={title} delay={(index % 3) * 80}>
              <article className="card">
                <span className="chip">{tag}</span>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section">
        <Reveal>
          <h2>How it works</h2>
        </Reveal>
        <div className="steps">
          {STEPS.map(([title, body], index) => (
            <Reveal key={title} delay={index * 100}>
              <article className="step">
                <span className="step-num">{index + 1}</span>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section about" id="about">
        <Reveal>
          <div className="section-eyebrow">
            <span>About us</span>
          </div>
          <h2>Assistants should show their work.</h2>
        </Reveal>
        <Reveal delay={80}>
          <p className="about-lede">
            GroundedAI started from a simple frustration: assistants that
            answer with total confidence and no receipts. We wanted a
            research tool for statutes, contracts, and papers that behaves
            the way a good analyst does. Retrieve carefully, answer only what
            the evidence supports, and show the page number for every claim.
          </p>
        </Reveal>
        <Reveal delay={140}>
          <p className="about-lede">
            So we built a citation-enforced research assistant and made the
            enforcement mechanical. Claims are validated by code, evaluations
            gate every release, and refusals are a feature rather than an
            embarrassment. The team is small, the stack is open, and the
            standard is simple: if we cannot prove it, we do not ship it.
          </p>
        </Reveal>
        <div className="grid about-grid">
          <Reveal delay={60}>
            <article className="card">
              <span className="chip">Value</span>
              <h3>Evidence over confidence</h3>
              <p>A hedged, sourced answer beats a fluent guess every time.</p>
            </article>
          </Reveal>
          <Reveal delay={140}>
            <article className="card">
              <span className="chip">Value</span>
              <h3>Refusals over hallucination</h3>
              <p>When the corpus is silent, we say so plainly instead of inventing.</p>
            </article>
          </Reveal>
          <Reveal delay={220}>
            <article className="card">
              <span className="chip">Value</span>
              <h3>Your documents, your tenant</h3>
              <p>Isolation by default, deletion on request, training on your data: never.</p>
            </article>
          </Reveal>
        </div>
      </section>

      <section className="section">
        <Reveal>
          <h2>FAQ</h2>
        </Reveal>
        <div className="faq">
          {FAQ.map(([question, answer], index) => (
            <Reveal key={question} delay={index * 40}>
              <details>
                <summary>{question}</summary>
                <p>{answer}</p>
              </details>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section">
        <Reveal>
          <h2>What is next</h2>
        </Reveal>
        <div className="grid">
          {UPCOMING.map(([tag, title, body], index) => (
            <Reveal key={title} delay={index * 80}>
              <article className="card card-soon">
                <span className="chip chip-soon">{tag}</span>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section">
        <Reveal>
          <h2>Pricing</h2>
        </Reveal>
        <div className="price-grid">
          {PRICING.map((plan, index) => (
            <Reveal key={plan.name} delay={index * 90}>
              <article className={`price${plan.featured ? " price-featured" : ""}`}>
                <h3>{plan.name}</h3>
                <div className="price-amount">
                  {plan.amount}
                  {plan.unit && <span className="price-unit">{plan.unit}</span>}
                </div>
                <ul>
                  {plan.features.map((feature) => (
                    <li key={feature}>{feature}</li>
                  ))}
                </ul>
                <span className={`price-note${plan.featured ? "" : " price-note-muted"}`}>
                  {plan.note}
                </span>
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <footer className="footer">
        <a href="https://github.com/aieng-abdullah/production-rag-assistant" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>
        &nbsp;·&nbsp;
        <Link to="/privacy">Privacy</Link>
        &nbsp;·&nbsp;
        <Link to="/terms">Terms</Link>
        <HealthBadge />
      </footer>
    </main>
  );
}
