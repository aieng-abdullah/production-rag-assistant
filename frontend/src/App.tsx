import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { ApiError, api } from "./api/client";
import { useAuth } from "./auth/AuthContext";
import ErrorBoundary from "./components/ErrorBoundary";
import ScrollToTop from "./components/ScrollToTop";
import { ToastProvider } from "./components/Toast";
import Admin from "./pages/Admin";
import AuthCallback from "./pages/AuthCallback";
import Chat from "./pages/Chat";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Settings from "./pages/Settings";
import Billing from "./pages/Billing";
import Privacy from "./pages/Privacy";
import Terms from "./pages/Terms";

function RequireAuth({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

/** Probe /admin/stats once per session: Admin link only for allow-listed accounts. */
function useIsAdmin(): boolean {
  const { user } = useAuth();
  const [isAdmin, setIsAdmin] = useState(
    () => user !== null && sessionStorage.getItem(`gai.is_admin.${user.id}`) === "1",
  );
  useEffect(() => {
    if (!user) {
      setIsAdmin(false);
      return;
    }
    const key = `gai.is_admin.${user.id}`;
    const cached = sessionStorage.getItem(key);
    if (cached !== null) {
      setIsAdmin(cached === "1");
      return;
    }
    api("/admin/stats")
      .then(() => {
        sessionStorage.setItem(key, "1");
        setIsAdmin(true);
      })
      .catch((error) => {
        // Cache only positive hits: a "0" would go stale when the operator
        // adds/removes ADMIN_EMAILS (sessionStorage outlives the change).
        // Non-admins re-probe once per page load; backend stays authoritative.
        if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
          sessionStorage.removeItem(key);
        }
        setIsAdmin(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);
  return isAdmin;
}

function Nav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const isAdmin = useIsAdmin();
  const [menuOpen, setMenuOpen] = useState(false);
  const closeMenu = () => setMenuOpen(false);
  const signOut = () => {
    closeMenu();
    logout();
    navigate("/", { replace: true });
  };
  return (
    <header className="nav">
      <div className="nav-inner">
        <Link to="/" className="brand" onClick={closeMenu}>
          Grounded<span>AI</span>
        </Link>
        <div className="nav-actions">
          <nav>
            <Link to="/">Product</Link>
            <Link className="nav-about" to="/#about">About</Link>
            {user && <Link to="/chat">Chat</Link>}
            {user && <Link className="nav-wide" to="/documents">Documents</Link>}
            {user && <Link className="nav-wide" to="/dashboard">Dashboard</Link>}
            {user && <Link className="nav-wide" to="/settings">Settings</Link>}
            {user && isAdmin && <Link to="/admin">Admin</Link>}
            {user ? (
              <button
                className="btn btn-ghost nav-btn"
                onClick={() => {
                  logout();
                  navigate("/", { replace: true });
                }}
              >
                Sign out
              </button>
            ) : (
              <Link className="btn btn-primary nav-btn" to="/login">
                Sign in
              </Link>
            )}
          </nav>
          <button
            type="button"
            className="nav-toggle"
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={menuOpen}
            aria-controls="nav-menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              aria-hidden="true"
              focusable="false"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              {menuOpen ? (
                <>
                  <path d="M6 6l12 12" />
                  <path d="M18 6L6 18" />
                </>
              ) : (
                <>
                  <path d="M4 7h16" />
                  <path d="M4 12h16" />
                  <path d="M4 17h16" />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav id="nav-menu" className="nav-menu">
          <Link to="/" onClick={closeMenu}>Product</Link>
          <Link to="/#about" onClick={closeMenu}>About</Link>
          {user && <Link to="/chat" onClick={closeMenu}>Chat</Link>}
          {user && <Link to="/documents" onClick={closeMenu}>Documents</Link>}
          {user && <Link to="/dashboard" onClick={closeMenu}>Dashboard</Link>}
          {user && <Link to="/settings" onClick={closeMenu}>Settings</Link>}
          {user && isAdmin && <Link to="/admin" onClick={closeMenu}>Admin</Link>}
          {user ? (
            <button className="btn btn-ghost" onClick={signOut}>
              Sign out
            </button>
          ) : (
            <Link className="btn btn-primary" to="/login" onClick={closeMenu}>
              Sign in
            </Link>
          )}
        </nav>
      )}
    </header>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <ScrollToTop />
        <Nav />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/auth" element={<AuthCallback />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/terms" element={<Terms />} />
          <Route
            path="/chat"
            element={
              <RequireAuth>
                <Chat />
              </RequireAuth>
            }
          />
          <Route
            path="/documents"
            element={
              <RequireAuth>
                <Documents />
              </RequireAuth>
            }
          />
          <Route
            path="/dashboard"
            element={
              <RequireAuth>
                <Dashboard />
              </RequireAuth>
            }
          />
          <Route
            path="/settings"
            element={
              <RequireAuth>
                <Settings />
              </RequireAuth>
            }
          />
          <Route
            path="/billing"
            element={
              <RequireAuth>
                <Billing />
              </RequireAuth>
            }
          />
          <Route
            path="/admin"
            element={
              <RequireAuth>
                <Admin />
              </RequireAuth>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </ToastProvider>
    </ErrorBoundary>
  );
}
