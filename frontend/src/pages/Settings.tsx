import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { useToast } from "../components/Toast";
import WorkspaceSwitch, { type Workspace } from "../components/WorkspaceSwitch";

interface Usage {
  tier: string;
  queries: { used: number; limit: number };
  documents: { used: number; limit: number };
  storage: { used_bytes: number; limit_bytes: number };
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export default function Settings() {
  const { user, logout } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [workspace, setWorkspace] = useState<Workspace>(() => {
    const stored = localStorage.getItem("gai_workspace");
    return stored === "legal" || stored === "academic" ? stored : "academic";
  });

  useEffect(() => {
    api<Usage>("/usage")
      .then(setUsage)
      .catch((error) =>
        toast(error instanceof ApiError ? error.message : "Usage unavailable", "error"),
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function changeWorkspace(ws: Workspace) {
    setWorkspace(ws);
    localStorage.setItem("gai_workspace", ws);
    toast(`Default workspace saved: ${ws}`, "success");
  }

  return (
    <main className="panel-page shell">
      <header className="panel-head">
        <div>
          <span className="chip">Preferences</span>
          <h1>Settings</h1>
          <p className="panel-lede">
            Account, default workspace, and live quota usage.
          </p>
        </div>
      </header>

      <section className="card settings-card">
        <span className="demo-label">Account</span>
        <div className="settings-row">
          <div>
            <strong className="settings-strong">User #{user?.id ?? "–"}</strong>
            <span className="metric-sub">
              {usage?.tier === "anonymous"
                ? "Guest session"
                : usage?.tier === "member"
                  ? "Member"
                  : "…"}
            </span>
          </div>
          <button
            className="btn btn-ghost"
            onClick={() => {
              logout();
              navigate("/", { replace: true });
            }}
          >
            Sign out
          </button>
        </div>
        <p className="panel-note">
          Email and profile management arrive with the account API endpoint.
          Until then this browser session is your identity.
        </p>
      </section>

      <section className="card settings-card">
        <span className="demo-label">Default workspace</span>
        <div className="settings-row">
          <WorkspaceSwitch value={workspace} onChange={changeWorkspace} />
          <span className="metric-sub">Legal = navy · Academic = teal</span>
        </div>
        <p className="panel-note">
          Saved in this browser and used when Chat or Documents opens. Server-side
          default needs a profile endpoint (follow-up).
        </p>
      </section>

      <section className="card settings-card">
        <span className="demo-label">Usage</span>
        <div className="metric-grid settings-metrics">
          <div className="settings-metric">
            <strong>{usage ? `${usage.queries.used}/${usage.queries.limit}` : "–"}</strong>
            <span className="metric-sub">Queries today</span>
          </div>
          <div className="settings-metric">
            <strong>{usage ? `${usage.documents.used}/${usage.documents.limit}` : "–"}</strong>
            <span className="metric-sub">Documents</span>
          </div>
          <div className="settings-metric">
            <strong>{usage ? formatBytes(usage.storage.used_bytes) : "–"}</strong>
            <span className="metric-sub">
              Storage of {usage ? formatBytes(usage.storage.limit_bytes) : "–"}
            </span>
          </div>
        </div>
        <p className="panel-note">Limits reset daily at midnight UTC.</p>
      </section>

      <section className="card settings-card">
        <span className="demo-label">Model providers</span>
        <p className="panel-note">
          The deployment operator configures LLM and embedding keys on the server
          (.env). They are never accepted from the browser, so there is nothing to
          edit here.
        </p>
      </section>
    </main>
  );
}
