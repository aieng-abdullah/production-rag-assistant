/**
 * Shared billing-tier labels. `GET /usage` returns the effective tier:
 * "anonymous" (guest session), "free", or "pro" (Subscription row).
 * "member" is accepted as a legacy free-tier value.
 */
export function tierLabel(tier: string | null | undefined): string {
  if (tier === "anonymous") return "Guest";
  if (tier === "pro") return "Pro";
  if (tier === "free" || tier === "member") return "Free";
  return "…";
}
