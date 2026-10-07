import { useRef, useState, type MouseEvent } from "react";

type Mode = "cited" | "abstain";

interface Source {
  id: number;
  doc: string;
  page: number;
  quote: string;
}

const SOURCES: Source[] = [
  {
    id: 1,
    doc: "meridian-msa.pdf",
    page: 12,
    quote:
      "This Agreement terminates upon ninety (90) days prior written notice by either party, unless renewed in writing.",
  },
  {
    id: 2,
    doc: "meridian-msa.pdf",
    page: 19,
    quote:
      "Sections 7 (Confidentiality) and 11 (Indemnity) survive termination of this Agreement.",
  },
];

type Segment = { text: string } | { cite: number };

const CITED_SEGMENTS: Segment[] = [
  { text: "Either party may end the agreement with 90 days of written notice" },
  { cite: 1 },
  { text: ", and confidentiality duties remain in effect after the agreement ends" },
  { cite: 2 },
  { text: "." },
];

const CLAIMS = [
  {
    n: 1,
    text: "Either party may end the agreement with 90 days of written notice",
    verdict: "SUPPORTED",
    reason: "quote matches meridian-msa.pdf page 12",
  },
  {
    n: 2,
    text: "Confidentiality duties survive termination",
    verdict: "SUPPORTED",
    reason: "quote matches meridian-msa.pdf page 19",
  },
];

const ABSTAIN_Q = "Who countersigned the 2021 amendment?";
const ABSTAIN_A =
  "I don't have enough information to answer this question based on the provided sources.";

/** Interactive proof: cited answer vs honest abstain, 3D tilt, verdict report. */
export default function CitationDemo() {
  const [mode, setMode] = useState<Mode>("cited");
  const [active, setActive] = useState(1);
  const cardRef = useRef<HTMLDivElement>(null);

  const source = SOURCES.find((entry) => entry.id === active)!;

  function onMove(event: MouseEvent<HTMLDivElement>) {
    const card = cardRef.current;
    if (!card || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }
    const rect = card.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width - 0.5;
    const py = (event.clientY - rect.top) / rect.height - 0.5;
    card.style.setProperty("--rx", `${(-py * 5).toFixed(2)}deg`);
    card.style.setProperty("--ry", `${(px * 6).toFixed(2)}deg`);
  }

  function onLeave() {
    const card = cardRef.current;
    card?.style.setProperty("--rx", "0deg");
    card?.style.setProperty("--ry", "0deg");
  }

  return (
    <div className="demo-3d" onMouseMove={onMove} onMouseLeave={onLeave}>
      <div className="demo" ref={cardRef}>
        <div className="demo-head">
          <div className="demo-tabs" role="tablist" aria-label="Proof mode">
            <button
              role="tab"
              aria-selected={mode === "cited"}
              className={`demo-tab${mode === "cited" ? " is-active" : ""}`}
              onClick={() => setMode("cited")}
            >
              Cited answer
            </button>
            <button
              role="tab"
              aria-selected={mode === "abstain"}
              className={`demo-tab${mode === "abstain" ? " is-active" : ""}`}
              onClick={() => setMode("abstain")}
            >
              Honest abstain
            </button>
          </div>
          <span className="demo-pass">
            <span className="demo-pass-dot" aria-hidden="true" />
            {mode === "cited" ? "2 of 2 claims verified" : "0 claims emitted"}
          </span>
        </div>

        <div className="demo-q">
          <span className="demo-label">Question</span>
          <p>{mode === "cited" ? "When does the agreement end?" : ABSTAIN_Q}</p>
        </div>

        <div className="demo-a">
          <span className="demo-label">
            {mode === "cited" ? "Verified answer" : "Refusal on the record"}
          </span>
          {mode === "cited" ? (
            <p>
              {CITED_SEGMENTS.map((segment, index) =>
                "text" in segment ? (
                  <span key={index} className="demo-text">
                    {segment.text}
                  </span>
                ) : (
                  <button
                    key={index}
                    className={`demo-chip${active === segment.cite ? " is-active" : ""}`}
                    style={{ animationDelay: `${300 + index * 160}ms` }}
                    onClick={() => setActive(segment.cite)}
                    aria-label={`Show source ${segment.cite}`}
                  >
                    [SOURCE {segment.cite}]
                  </button>
                ),
              )}
            </p>
          ) : (
            <p className="demo-abstain">&ldquo;{ABSTAIN_A}&rdquo;</p>
          )}
        </div>

        {mode === "cited" ? (
          <div className="demo-source" key={source.id}>
            <div className="demo-source-head">
              <span className="demo-label">Evidence</span>
              <span className="demo-source-page">
                {source.doc} · page {source.page}
              </span>
            </div>
            <p className="demo-quote">&ldquo;{source.quote}&rdquo;</p>
            <span className="demo-match">quote matches the source word for word</span>
          </div>
        ) : (
          <div className="demo-source demo-source-empty" key="abstain">
            <div className="demo-source-head">
              <span className="demo-label">Evidence</span>
              <span className="demo-source-page">none cited</span>
            </div>
            <p className="demo-quote">
              The corpus does not contain a countersignature, so no source was
              offered and no claim was written.
            </p>
            <span className="demo-match">nothing fabricated</span>
          </div>
        )}

        <div className="demo-report">
          <span className="demo-label">Verification report</span>
          {mode === "cited" ? (
            <ul>
              {CLAIMS.map((claim) => (
                <li key={claim.n}>
                  <span className="demo-verdict">{claim.verdict}</span>
                  <span className="demo-claim">
                    {claim.n}. {claim.text}
                  </span>
                  <span className="demo-reason">{claim.reason}</span>
                </li>
              ))}
            </ul>
          ) : (
            <ul>
              <li>
                <span className="demo-verdict demo-verdict-abstain">ABSTAINED</span>
                <span className="demo-claim">1. refusal issued with zero claims</span>
                <span className="demo-reason">validator released the response unmodified</span>
              </li>
            </ul>
          )}
        </div>

        <p className="demo-hint">
          Flip tabs above. Every sentence is checked against the chunk it cites.
        </p>
      </div>
    </div>
  );
}
