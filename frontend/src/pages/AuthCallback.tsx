import { useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

/** OAuth landing: `{FRONTEND_URL}/auth#token=…`. Store it, then enter the app. */
export default function AuthCallback() {
  const { consumeFragmentToken } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const processed = useRef(false);

  useEffect(() => {
    if (processed.current) return;
    if (!location.pathname.startsWith("/auth")) return;
    processed.current = true;

    const hasToken = consumeFragmentToken();
    navigate(hasToken ? "/chat" : "/login", { replace: true });
  }, [consumeFragmentToken, navigate, location.pathname]);

  return (
    <main className="placeholder glass">
      <h2>Signing you in…</h2>
      <p>Completing your session.</p>
    </main>
  );
}
