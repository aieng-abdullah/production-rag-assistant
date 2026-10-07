import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "./auth/AuthContext";
import ErrorBoundary from "./components/ErrorBoundary";
import ScrollToTop from "./components/ScrollToTop";
import { ToastProvider } from "./components/Toast";
import AuthCallback from "./pages/AuthCallback";
import Chat from "./pages/Chat";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Privacy from "./pages/Privacy";
import Terms from "./pages/Terms";

function RequireAuth({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function Nav() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  return (
    <header className="nav">
      <div className="nav-inner">
        <Link to="/" className="brand">
          Grounded<span>AI</span>
        </Link>
        <nav>
          <Link to="/">Product</Link>
          <Link className="nav-about" to="/#about">About</Link>
          {user && <Link to="/chat">Chat</Link>}
          {user && <Link className="nav-wide" to="/documents">Documents</Link>}
          {user && <Link className="nav-wide" to="/dashboard">Dashboard</Link>}
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
      </div>
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
          <Route path="*" element={<NotFound />} />
        </Routes>
      </ToastProvider>
    </ErrorBoundary>
  );
}
