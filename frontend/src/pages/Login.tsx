import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { useToast } from "../components/Toast";

function GoogleIcon() {
  return (
    <svg viewBox="0 0 48 48" width="18" height="18" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
    </svg>
  );
}

export default function Login() {
  const { googleLogin, guestLogin, demoLogin } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [busy, setBusy] = useState<"guest" | "demo" | null>(null);

  async function run(kind: "guest" | "demo", action: () => Promise<void>) {
    setBusy(kind);
    try {
      await action();
      toast("Welcome in.", "success");
      navigate("/chat", { replace: true });
    } catch (exc) {
      toast(exc instanceof ApiError ? exc.message : "Sign-in failed", "error");
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="auth-wrap">
      <div className="auth-hero">
        <span className="chip">Legal &amp; academic · citation-enforced RAG</span>
        <h1>Your citation-enforced research assistant</h1>
        <p>
          Ask your statutes, contracts, and papers questions in plain
          language. Every sentence cites its page-level source, validated by
          code before you see it.
        </p>
      </div>

      <div className="auth-card glass">
        <h2>Sign in to GroundedAI</h2>

        <button className="btn auth-btn" onClick={googleLogin}>
          <GoogleIcon />
          Sign in with Google
        </button>

        <div className="auth-divider"><span>or</span></div>

        <button
          className="btn btn-primary auth-btn"
          disabled={busy !== null}
          onClick={() => run("guest", guestLogin)}
        >
          {busy === "guest" ? "Signing in…" : "Continue as guest"}
        </button>

        <button
          className="btn btn-ghost auth-btn"
          disabled={busy !== null}
          onClick={() => run("demo", demoLogin)}
        >
          {busy === "demo" ? "Signing in…" : "Try the demo account"}
        </button>

        <p className="auth-note">
          No account needed for guest access: 3 questions free to try it out.
          Your documents stay private and isolated to your session.
        </p>
        <p className="auth-consent">
          By continuing you agree to the <Link to="/terms">Terms</Link> and
          acknowledge the <Link to="/privacy">Privacy Policy</Link>.
        </p>
      </div>

      <div className="auth-trust">
        <div className="trust-card glass">
          <strong>1.00</strong>
          <span>Faithfulness on golden set (Ragas)</span>
        </div>
        <div className="trust-card glass">
          <strong>Page-level</strong>
          <span>Every sentence cites its source page</span>
        </div>
        <div className="trust-card glass">
          <strong>Abstains</strong>
          <span>Says "not in your documents" instead of guessing</span>
        </div>
      </div>

      <p className="auth-foot">
        Open source stack: Groq · LangChain · Chroma · FastAPI · React
        &nbsp;·&nbsp; <Link to="/privacy">Privacy</Link>
        &nbsp;·&nbsp; <Link to="/terms">Terms</Link>
      </p>
    </main>
  );
}
