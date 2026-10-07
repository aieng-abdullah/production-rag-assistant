import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import { useToast } from "./Toast";

interface TraceChunk {
  source_id: number;
  doc_id: string;
  page_num: number;
  cited: boolean;
}
interface TraceEntry {
  workspace: string;
  prompt_version: string;
  model: string;
  verify_model: string;
  token_usage: { input_tokens: number; output_tokens: number } | null;
  chunks: TraceChunk[];
  claims: { text: string; citations: { source_id: number; quote: string }[] }[];
  abstained: boolean;
  abstain_reason: string | null;
  verification: { status: string; per_claim: { verdict: string; reason: string }[] };
}
interface TraceResponse {
  answer: { id: number; query: string; answer: string; created_at: string };
  trace: TraceEntry[];
}

/** Provenance viewer: model, prompt version, chunks, claims, token usage. */
export default function TraceModal({
  answerId,
  onClose,
}: {
  answerId: number;
  onClose: () => void;
}) {
  const { toast } = useToast();
  const [data, setData] = useState<TraceResponse | null>(null);

  useEffect(() => {
    api<TraceResponse>(`/answers/${answerId}/trace`)
      .then(setData)
      .catch((error) =>
        toast(error instanceof ApiError ? error.message : "Trace unavailable", "error"),
      );
  }, [answerId, toast]);

  const entry = data?.trace[0];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal glass"
        role="dialog"
        aria-label="Provenance trace"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-head">
          <span className="chip">Provenance trace</span>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {!data && !entry && <p className="modal-loading">Loading trace…</p>}

        {entry && (
          <>
            <div className="trace-meta">
              <span>
                <em>Model</em> {entry.model}
              </span>
              <span>
                <em>Verifier</em> {entry.verify_model}
              </span>
              <span>
                <em>Prompt</em> {entry.prompt_version}
              </span>
              <span>
                <em>Workspace</em> {entry.workspace}
              </span>
              {entry.token_usage && (
                <span>
                  <em>Tokens</em> {entry.token_usage.input_tokens} in ·{" "}
                  {entry.token_usage.output_tokens} out
                </span>
              )}
            </div>

            <div className="trace-block">
              <span className="demo-label">Chunks retrieved ({entry.chunks.length})</span>
              <ul className="trace-list">
                {entry.chunks.map((chunk) => (
                  <li key={chunk.source_id} className={chunk.cited ? "is-cited" : ""}>
                    <span className="trace-src">[SOURCE {chunk.source_id}]</span>
                    {chunk.doc_id} · page {chunk.page_num > 0 ? chunk.page_num : "—"}
                    {chunk.cited && <span className="trace-cited-mark">cited</span>}
                  </li>
                ))}
              </ul>
            </div>

            <div className="trace-block">
              <span className="demo-label">Claims checked ({entry.claims.length})</span>
              <ul className="trace-list">
                {entry.claims.map((claim, index) => (
                  <li key={index}>
                    <span className="trace-claim">{claim.text}</span>
                    {claim.citations.map((cite) => (
                      <span key={cite.source_id} className="trace-quote">
                        [SOURCE {cite.source_id}] &ldquo;{cite.quote.slice(0, 140)}
                        {cite.quote.length > 140 ? "…" : ""}&rdquo;
                      </span>
                    ))}
                  </li>
                ))}
                {entry.abstained && (
                  <li className="trace-claim">
                    Abstained: {entry.abstain_reason ?? "no supporting evidence"}
                  </li>
                )}
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
