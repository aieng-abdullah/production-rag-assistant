import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import TraceModal from "../components/TraceModal";
import { useToast } from "../components/Toast";
import WorkspaceSwitch, { WS_ACCENT, type Workspace } from "../components/WorkspaceSwitch";

interface Source {
  doc_id: string;
  page_num: number;
  text: string;
  source_id: number;
}
interface PerClaim {
  claim: number;
  text: string;
  verdict: "SUPPORTED" | "PARTIAL" | "UNSUPPORTED";
  reason: string;
}
interface Verification {
  status: "verified" | "partial" | "unverified" | "abstained";
  per_claim: PerClaim[];
}
interface ChatResponse {
  answer_id: number | null;
  answer: string;
  sources: Source[];
  verification: Verification;
}
interface Usage {
  tier: string;
  queries: { used: number; limit: number };
  documents: { used: number; limit: number };
}

type Message =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      loading?: boolean;
      failed?: string;
      answer?: string;
      sources?: Source[];
      verification?: Verification;
      answerId?: number | null;
    };

const WARM_ABSTAIN =
  "I couldn't find that in your documents, so I'd rather tell you than guess. Try rephrasing your question, or upload a document that covers it.";

function recordHistory(query: string, status: string) {
  try {
    const raw = localStorage.getItem("gai_history");
    const items: { q: string; status: string; ts: number }[] = raw
      ? JSON.parse(raw)
      : [];
    items.unshift({ q: query, status, ts: Date.now() });
    localStorage.setItem("gai_history", JSON.stringify(items.slice(0, 30)));
  } catch {
    /* private mode / quota: history is a nicety, never fatal */
  }
}

const STARTERS = [
  "What are the main obligations of each party?",
  "Summarize the termination clause",
  "Which deadlines are mentioned in this document?",
];

function renderAnswer(text: string) {
  const parts = text.split(/(\[SOURCE \d+\])/g);
  return parts.map((part, index) => {
    const match = part.match(/^\[SOURCE (\d+)\]$/);
    if (!match) return <span key={index}>{part}</span>;
    return (
      <span
        key={index}
        className="cite-chip"
        onClick={() => {
          document.getElementById(`source-${match[1]}`)?.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }}
        role="button"
        tabIndex={0}
      >
        {part}
      </span>
    );
  });
}

function StatusBadge({ verification }: { verification?: Verification }) {
  if (!verification) return null;
  const label = verification.status;
  return (
    <span className={`status-badge status-${verification.status}`}>{label}</span>
  );
}

export default function Chat() {
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [workspace, setWorkspace] = useState<Workspace>(() => {
    const stored = localStorage.getItem("gai_workspace");
    return stored === "legal" || stored === "academic" ? stored : "academic";
  });
  const [usage, setUsage] = useState<Usage | null>(null);
  const [traceId, setTraceId] = useState<number | null>(null);
  const busy = useRef(false);
  const threadRef = useRef<HTMLDivElement>(null);

  async function refreshUsage() {
    try {
      setUsage(await api<Usage>("/usage"));
    } catch {
      /* quota pill degrades silently; chat still works */
    }
  }

  useEffect(() => {
    void refreshUsage();
  }, []);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function send(raw: string) {
    const query = raw.trim();
    if (!query || busy.current) return;
    busy.current = true;
    setInput("");
    setMessages((current) => [
      ...current,
      { role: "user", content: query },
      { role: "assistant", loading: true },
    ]);

    try {
      const response = await api<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify({ query, workspace }),
      });
      setMessages((current) => {
        const next = [...current];
        next[next.length - 1] = {
          role: "assistant",
          answer: response.answer,
          sources: response.sources,
          verification: response.verification,
          answerId: response.answer_id,
        };
        return next;
      });
      recordHistory(query, response.verification.status);
      void refreshUsage();
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Something went wrong";
      if (error instanceof ApiError && error.status === 0) {
        toast("Backend is waking up (cold start). Try again shortly.", "error");
      } else if (error instanceof ApiError && error.status === 429) {
        toast(message, "error");
      } else {
        toast(message, "error");
      }
      setMessages((current) => {
        const next = [...current];
        next[next.length - 1] = {
          role: "assistant",
          failed: message,
        };
        return next;
      });
    } finally {
      busy.current = false;
    }
  }

  function clearChat() {
    setMessages([]);
  }

  function lastQuestion() {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === "user") return message.content;
    }
    return "";
  }

  const remaining =
    usage !== null ? Math.max(usage.queries.limit - usage.queries.used, 0) : null;

  return (
    <main className="chat-page shell">
      <header className="chat-bar">
        <WorkspaceSwitch value={workspace} onChange={setWorkspace} />
        <div className="chat-bar-right">
          {usage && (
            <span className="quota-pill" title={`${usage.tier} tier`}>
              {remaining} / {usage.queries.limit} queries left
            </span>
          )}
          <button className="chat-clear" onClick={clearChat} disabled={!messages.length}>
            Clear
          </button>
        </div>
      </header>

      <div className="chat-thread" ref={threadRef} style={{ ["--ws" as string]: WS_ACCENT[workspace] }}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <span className="chip">Research assistant</span>
            <h2>Ask anything about your documents.</h2>
            <p>
              Every answer cites its page. If the corpus does not cover the
              question, the assistant says so instead of guessing.
            </p>
            <div className="chat-starters">
              {STARTERS.map((starter) => (
                <button key={starter} className="starter" onClick={() => void send(starter)}>
                  {starter}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) =>
          message.role === "user" ? (
            <div className="msg msg-user" key={index}>
              <p>{message.content}</p>
            </div>
          ) : message.loading ? (
            <div className="msg msg-ai msg-loading" key={index}>
              <div className="thinking-dots" aria-label="Thinking">
                <span />
                <span />
                <span />
              </div>
              <p>Thinking through your documents…</p>
            </div>
          ) : message.failed ? (
            <div className="msg msg-ai msg-failed" key={index}>
              <p>
                I hit an error while generating that answer. Please try again.
              </p>
              <span className="msg-error-detail">{message.failed}</span>
              <button
                className="starter retry-btn"
                onClick={() => {
                  const question = lastQuestion();
                  setMessages((current) => current.slice(0, -1));
                  void send(question);
                }}
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="msg msg-ai" key={index}>
              <div className="msg-ai-head">
                <StatusBadge verification={message.verification} />
                <div className="msg-actions">
                  <button
                    onClick={() => {
                      void navigator.clipboard.writeText(message.answer ?? "");
                      toast("Answer copied.", "success");
                    }}
                  >
                    Copy
                  </button>
                  {message.answerId != null && (
                    <button onClick={() => setTraceId(message.answerId as number)}>
                      Trace
                    </button>
                  )}
                </div>
              </div>

              <p className="msg-answer">
                {renderAnswer(
                  message.verification?.status === "abstained"
                    ? WARM_ABSTAIN
                    : message.answer ?? "",
                )}
              </p>

              {message.sources && message.sources.length > 0 && (
                <div className="msg-sources">
                  <span className="msg-sources-label">
                    {message.verification?.status === "abstained"
                      ? "Related passages (none support an answer)"
                      : "Sources"}
                  </span>
                  <span className="msg-source-chips">
                    {message.sources.map((source) => (
                      <span className="src-chip" key={source.source_id}>
                        [{source.source_id}] p.
                        {source.page_num > 0 ? source.page_num : "—"}
                      </span>
                    ))}
                  </span>
                  <details className="msg-source-details">
                    <summary>
                      Read the {message.sources.length} source passages
                    </summary>
                    {message.sources.map((source) => (
                      <div className="src-row" id={`source-${source.source_id}`} key={source.source_id}>
                        <div className="src-row-head">
                          <span className="src-id">[{source.source_id}]</span>
                          <span className="src-doc">
                            {source.doc_id} · page{" "}
                            {source.page_num > 0 ? source.page_num : "—"}
                          </span>
                        </div>
                        <p>
                          {source.text.slice(0, 500)}
                          {source.text.length > 500 ? "…" : ""}
                        </p>
                      </div>
                    ))}
                  </details>
                </div>
              )}

              {message.verification &&
                message.verification.per_claim.length > 0 && (
                  <details className="msg-claims">
                    <summary>
                      Verifier: {message.verification.per_claim.length} claims checked
                    </summary>
                    {message.verification.per_claim.map((claim) => (
                      <div className="claim-row" key={claim.claim}>
                        <span className={`claim-verdict claim-${claim.verdict.toLowerCase()}`}>
                          {claim.verdict}
                        </span>
                        <span className="claim-text">{claim.text}</span>
                      </div>
                    ))}
                  </details>
                )}
            </div>
          ),
        )}
      </div>

      <form
        className="chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          void send(input);
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask a question about your documents…"
          maxLength={2000}
          aria-label="Question"
        />
        <button
          type="submit"
          className="btn btn-primary chat-send"
          disabled={!input.trim() || Boolean(messages.find((m) => m.role === "assistant" && "loading" in m && m.loading))}
        >
          Ask
        </button>
      </form>

      {remaining !== null && remaining <= 2 && remaining > 0 && (
        <p className="chat-quota-warn">Only {remaining} queries left today.</p>
      )}

      {traceId !== null && <TraceModal answerId={traceId} onClose={() => setTraceId(null)} />}
    </main>
  );
}
