import { useEffect } from "react";
import { API_BASE } from "../api/client";

/** Live service badge in the footer: pings GET /health on mount. */
export default function HealthBadge() {
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        const dot = document.getElementById("health-dot");
        const label = document.getElementById("health-label");
        if (cancelled || !dot || !label) return;
        const up = response.ok;
        dot.style.background = up ? "#5f7d52" : "#a4553f";
        label.textContent = up ? "All systems operational" : "Degraded";
      } catch {
        /* unreachable API renders degraded, never throws */
      }
    };
    check();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <span className="health">
      <span id="health-dot" className="health-dot" />
      <span id="health-label">Checking status…</span>
    </span>
  );
}
