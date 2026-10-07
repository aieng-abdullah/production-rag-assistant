import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

/** OAuth landing: `{FRONTEND_URL}/auth#token=…`. Store it, then enter the app. */
export default function AuthCallback() {
  const { consumeFragmentToken } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (consumeFragmentToken()) {
      navigate("/chat", { replace: true });
    } else {
      navigate("/login", { replace: true });
    }
  }, [consumeFragmentToken, navigate]);

  return (
    <main className="placeholder glass">
      <h2>Signing you in…</h2>
      <p>Completing your session.</p>
    </main>
  );
}
