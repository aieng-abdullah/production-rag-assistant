import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import { useToast } from "../components/Toast";
import { tierLabel } from "../utils/tier";

interface Usage {
  tier: string;
  queries: { used: number; limit: number };
  documents: { used: number; limit: number };
}

const PLANS = [
  {
    name: "Free",
    price: "$0",
    period: "forever",
    features: [
      "10 verified answers per day",
      "5 documents",
      "100 MB storage",
      "Google or guest sign-in",
    ],
    cta: null,
    note: "Your current plan",
  },
  {
    name: "Pro",
    price: "$9",
    period: "per month",
    features: [
      "100 verified answers per day (10x)",
      "50 documents",
      "100 MB storage (shared cap)",
      "Stripe billing portal for invoices",
    ],
    cta: "Upgrade to Pro",
    note: null,
  },
  {
    name: "Teams",
    price: "$29",
    period: "per month",
    features: [
      "Unlimited workspaces",
      "Admin + SSO on the roadmap",
      "Shared corpora",
    ],
    cta: null,
    note: "Coming soon",
  },
];

export default function Billing() {
  const { toast } = useToast();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api<Usage>("/usage").then(setUsage).catch(() => {
      /* plan card degrades to placeholders */
    });
  }, []);

  async function openCheckout() {
    setBusy(true);
    try {
      const { url } = await api<{ url: string }>("/billing/checkout", {
        method: "POST",
      });
      window.location.assign(url);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        toast(
          "Stripe billing is not configured on this deployment yet.",
          "info",
        );
      } else {
        toast(error instanceof ApiError ? error.message : "Checkout failed", "error");
      }
    } finally {
      setBusy(false);
    }
  }

  async function openPortal() {
    setBusy(true);
    try {
      const { url } = await api<{ url: string }>("/billing/portal", {
        method: "POST",
      });
      window.location.assign(url);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        toast("Billing portal is not configured on this deployment yet.", "info");
      } else {
        toast(error instanceof ApiError ? error.message : "Portal failed", "error");
      }
    } finally {
      setBusy(false);
    }
  }

  const currentPlan = usage === null ? "…" : tierLabel(usage.tier);

  return (
    <main className="panel-page shell">
      <header className="panel-head">
        <div>
          <span className="chip">Plans</span>
          <h1>Billing</h1>
          <p className="panel-lede">
            Free keeps research moving. Pro multiplies daily answers and
            documents by ten.
          </p>
        </div>
      </header>

      <section className="card settings-card">
        <span className="demo-label">Current plan</span>
        <div className="settings-row">
          <div>
            <strong className="settings-strong">{currentPlan}</strong>
            <span className="metric-sub">
              {usage
                ? `${usage.queries.used}/${usage.queries.limit} queries used today`
                : "loading usage"}
            </span>
          </div>
          <button
            className="btn btn-ghost"
            onClick={() => void openPortal()}
            disabled={busy}
          >
            Manage billing
          </button>
        </div>
        <p className="panel-note">
          Stripe activates per deployment with environment keys; until then these
          buttons report honestly instead of pretending.
        </p>
      </section>

      <section className="pricing-grid">
        {PLANS.map((plan) => (
          <div
            key={plan.name}
            className={`card pricing-card${plan.cta ? " pricing-featured" : ""}`}
          >
            <span className="demo-label">{plan.name}</span>
            <p className="pricing-price">
              {plan.price}
              <em>{plan.period}</em>
            </p>
            <ul className="pricing-features">
              {plan.features.map((feature) => (
                <li key={feature}>{feature}</li>
              ))}
            </ul>
            {plan.cta ? (
              <button
                className="btn btn-primary pricing-cta"
                onClick={() => void openCheckout()}
                disabled={busy}
              >
                {busy ? "Redirecting…" : plan.cta}
              </button>
            ) : (
              <span className="pricing-cta pricing-current">{plan.note}</span>
            )}
          </div>
        ))}
      </section>

      <section className="card settings-card">
        <span className="demo-label">Invoices</span>
        <p className="panel-note">
          Invoices and receipts live in the Stripe customer portal once Pro is
          active (Manage billing above).
        </p>
      </section>
    </main>
  );
}
