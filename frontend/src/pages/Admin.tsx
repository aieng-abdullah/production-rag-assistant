import { useCallback, useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { useToast } from "../components/Toast";

interface AdminStats {
  total_users: number;
  total_documents: number;
  total_answers: number;
  queries_today: number;
}

interface AdminUser {
  id: number;
  email: string;
  name: string | null;
  tier: string;
  doc_count: number;
  created_at: string | null;
}

export default function Admin() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [denied, setDenied] = useState<"denied" | "disabled" | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      api<AdminStats>("/admin/stats"),
      api<AdminUser[]>("/admin/users"),
    ])
      .then(([s, u]) => {
        setStats(s);
        setUsers(u);
        setDenied(null);
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 403) {
          // Revoked admin: drop the cached nav flag so the link disappears too.
          if (user) sessionStorage.removeItem(`gai.is_admin.${user.id}`);
          setDenied("denied");
        } else if (error instanceof ApiError && error.status === 404) {
          setDenied("disabled");
        } else {
          toast(error instanceof ApiError ? error.message : "Admin data unavailable", "error");
        }
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(load, [load]);

  async function changeTier(user: AdminUser, tier: string) {
    setBusyId(user.id);
    try {
      await api(`/admin/users/${user.id}/tier`, {
        method: "PUT",
        body: JSON.stringify({ tier }),
      });
      setUsers((prev) => prev.map((u) => (u.id === user.id ? { ...u, tier } : u)));
      toast(`Tier updated: ${user.email} → ${tier}`, "success");
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Tier update failed", "error");
    } finally {
      setBusyId(null);
    }
  }

  async function removeUser(user: AdminUser) {
    if (!window.confirm(`Delete ${user.email} and all their data? This cannot be undone.`)) {
      return;
    }
    setBusyId(user.id);
    try {
      await api(`/admin/users/${user.id}`, { method: "DELETE" });
      setUsers((prev) => prev.filter((u) => u.id !== user.id));
      toast(`Deleted ${user.email}`, "success");
      load();
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Delete failed", "error");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <main className="panel-page shell">
      <header className="panel-head">
        <div>
          <span className="chip">Admin</span>
          <h1>Admin panel</h1>
          <p className="panel-lede">
            User management and system totals. Access restricted to configured admin accounts.
          </p>
        </div>
      </header>

      {denied === "denied" ? (
        // 403: not allow-listed — send them straight back, no admin UI shown.
        <Navigate to="/" replace />
      ) : denied === "disabled" ? (
        <section className="card">
          <span className="demo-label">Not enabled</span>
          <p className="panel-note">
            Admin panel is not enabled on this deployment. The operator must set{" "}
            <code>ADMIN_EMAILS</code> on the API and redeploy.
          </p>
        </section>
      ) : (
        <>
          <section className="metric-grid">
            <div className="card metric-card">
              <span className="demo-label">Users</span>
              <strong className="metric-value">{stats ? stats.total_users : "–"}</strong>
              <span className="metric-sub">Registered accounts</span>
            </div>
            <div className="card metric-card">
              <span className="demo-label">Documents</span>
              <strong className="metric-value">{stats ? stats.total_documents : "–"}</strong>
              <span className="metric-sub">Uploaded across all users</span>
            </div>
            <div className="card metric-card">
              <span className="demo-label">Answers</span>
              <strong className="metric-value">{stats ? stats.total_answers : "–"}</strong>
              <span className="metric-sub">Total generated</span>
            </div>
            <div className="card metric-card">
              <span className="demo-label">Queries today</span>
              <strong className="metric-value">{stats ? stats.queries_today : "–"}</strong>
              <span className="metric-sub">Since midnight UTC</span>
            </div>
          </section>

          <section className="card">
            <span className="demo-label">Users</span>
            {loading ? (
              <p className="panel-note">Loading…</p>
            ) : users.length === 0 ? (
              <p className="panel-note">No users found.</p>
            ) : (
              <ul className="doc-list admin-user-list">
                {users.map((u) => (
                  <li key={u.id} className="doc-row admin-user-row">
                    <div className="admin-user-meta">
                      <strong>{u.email}</strong>
                      <span className="metric-sub">
                        User #{u.id}
                        {u.name ? ` · ${u.name}` : ""} · {u.doc_count} docs
                        {u.created_at ? ` · joined ${u.created_at.slice(0, 10)}` : ""}
                      </span>
                    </div>
                    <div className="admin-user-actions">
                      <select
                        className="admin-tier-select"
                        value={u.tier}
                        disabled={busyId === u.id}
                        onChange={(e) => changeTier(u, e.target.value)}
                        aria-label={`Tier for ${u.email}`}
                      >
                        <option value="free">free</option>
                        <option value="pro">pro</option>
                      </select>
                      <button
                        className="btn btn-ghost danger"
                        disabled={busyId === u.id}
                        onClick={() => removeUser(u)}
                      >
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </main>
  );
}
