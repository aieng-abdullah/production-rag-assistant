import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { useToast } from "../components/Toast";
import { tierLabel } from "../utils/tier";

interface Usage {
  tier: string;
  queries: { used: number; limit: number };
  documents: { used: number; limit: number };
  storage: { used_bytes: number; limit_bytes: number };
}
interface HistoryItem {
  q: string;
  status: string;
  ts: number;
}

const STATUS_COPY: Record<string, string> = {
  verified: "verified",
  partial: "partially verified",
  unverified: "unverified",
  abstained: "abstained",
};

function pct(used: number, limit: number) {
  return Math.min(Math.round((used / Math.max(limit, 1)) * 100), 100);
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export default function Dashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    api<Usage>("/usage")
      .then(setUsage)
      .catch((error) =>
        toast(error instanceof ApiError ? error.message : "Usage unavailable", "error"),
      );
    try {
      const raw = localStorage.getItem("gai_history");
      if (raw) setHistory(JSON.parse(raw));
    } catch {
      /* history is a nicety; missing localStorage must not break the page */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="panel-page shell">
      <header className="panel-head">
        <div>
          <span className="chip">Overview</span>
          <h1>Dashboard</h1>
          <p className="panel-lede">
            {user ? "Real quota numbers straight from the API." : ""}
          </p>
        </div>
        <div className="panel-head-actions">
          <Link className="btn btn-ghost" to="/chat">
            Ask a question
          </Link>
          <Link className="btn btn-primary" to="/documents">
            Upload document
          </Link>
        </div>
      </header>

      <section className="metric-grid">
        <div className="card metric-card">
          <span className="demo-label">Queries today</span>
          <strong className="metric-value">
            {usage ? usage.queries.limit - usage.queries.used : "–"}
          </strong>
          <span className="metric-sub">
            {usage ? `${usage.queries.used} of ${usage.queries.limit} used` : "loading"}
          </span>
          <div className="meter">
            <span style={{ width: usage ? `${pct(usage.queries.used, usage.queries.limit)}%` : "0%" }} />
          </div>
        </div>

        <div className="card metric-card">
          <span className="demo-label">Documents</span>
          <strong className="metric-value">
            {usage ? usage.documents.used : "–"}
          </strong>
          <span className="metric-sub">
            {usage ? `of ${usage.documents.limit} on this plan` : "loading"}
          </span>
          <div className="meter">
            <span
              style={{ width: usage ? `${pct(usage.documents.used, usage.documents.limit)}%` : "0%" }}
            />
          </div>
        </div>

        <div className="card metric-card">
          <span className="demo-label">Storage</span>
          <strong className="metric-value">
            {usage ? formatBytes(usage.storage.used_bytes) : "–"}
          </strong>
          <span className="metric-sub">
            {usage ? `of ${formatBytes(usage.storage.limit_bytes)}` : "loading"}
          </span>
          <div className="meter">
            <span
              style={{
                width: usage
                  ? `${pct(usage.storage.used_bytes, usage.storage.limit_bytes)}%`
                  : "0%",
              }}
            />
          </div>
        </div>

        <div className="card metric-card">
          <span className="demo-label">Plan</span>
          <strong className="metric-value">
            {usage ? tierLabel(usage.tier) : "–"}
          </strong>
          <span className="metric-sub">
            {usage?.tier === "anonymous"
              ? "Sign in for higher limits"
              : usage?.tier === "pro"
                ? "Pro quotas active (10x)"
                : "Upgrade in Billing for 10x"}
          </span>
        </div>
      </section>

      <section className="doc-list-section">
        <span className="demo-label">Recent questions (this browser)</span>
        {history.length === 0 ? (
          <div className="panel-empty">
            <p>No queries yet. Ask something in Chat and it shows up here.</p>
            <Link className="starter" to="/chat">
              Open Chat
            </Link>
          </div>
        ) : (
          <ul className="doc-list history-list">
            {history.map((item, index) => (
              <li key={`${item.ts}-${index}`} className="doc-row">
                <span className="history-q">{item.q}</span>
                <span className={`status-badge status-${item.status}`}>
                  {STATUS_COPY[item.status] ?? item.status}
                </span>
                <span className="doc-date">
                  {new Date(item.ts).toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="panel-note">
          Per-day usage charts and cross-device history need the usage-history
          endpoint (PLAN PR-4b). This list stays local to your browser.
        </p>
      </section>
    </main>
  );
}
