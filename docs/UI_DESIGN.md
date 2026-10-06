# UI Design Spec — Streamlit Frontend

**Owner ticket:** PLAN PR-6 (`refactor/frontend-api-client`), accents in PR-4b/PR-5/PR-7.
**Inspiration:** [antoineross/streamlit-saas-starter](https://github.com/antoineross/streamlit-saas-starter)
(sidebar `page_link` menu + redirect guard, landing/dashboard page split, shadcn-styled cards).
**Adapted, not copied:** we use Google OAuth → session-state auth (PLAN locked; JWT removed PR-6.1+drop-jwt), env-flagged Stripe,
our own quota/verification features. No Supabase.

**Guardrail (AGENTS.md):** no UI polish before service layer is proven.
This spec is documentation now; code lands in PR-6.

---

## 1. Principles

1. Trust-first: verification/provenance is the product — surfaces get the accent color.
2. Thin client (post-PR-6.1): pages import `src.services` directly in single-process
   mode; auth identity lives in `st.session_state` (no JWT, no Bearer).
3. shadcn look: neutral canvas, white cards, 1px borders, one primary accent.
4. Feedback: ephemeral success = toast; actionable error = inline banner. Never both-only-toast.
5. One concern per page. Chat never hosts settings; Documents never hosts chat.
6. **No emoji in the UI** — Material Symbols icons only.

## 2. Information architecture

```
Login.py                 # entrypoint: hero + Google sign-in (unauthenticated menu)
menu.py                  # sidebar nav + menu_with_redirect()  (pattern from reference)
styles/main.css          # CSS custom properties (see §4)
pages/
  1_Landing.py           # marketing: hero, features, workspaces, FAQ, pricing, contact
  2_Chat.py              # main app: chat + citations + workspace switcher
  3_Documents.py         # upload → status polling → list/delete
  4_Dashboard.py         # quota metric cards, usage history, trace viewer (PR-4b)
  5_Settings.py          # account, workspace prefs, dev provider keys, logout
```

- `st.set_option("client.showSidebarNavigation", False)` → custom menu only.
- `menu.py`: `authenticated_menu()` (Chat, Documents, Dashboard, Settings, Logout) vs
  `unauthenticated_menu()` (Landing, Login). `menu_with_redirect()` → `st.switch_page("app.py")`
  when `st.session_state.user_id` missing. Every page starts with `menu_with_redirect()`.

## 3. Auth & session flow (PR-2b Google OAuth + PR-6.1 session-state)

```
app.py: st.link_button("Sign in with Google", get_google_auth_url())
  → Google consent → app.py callback (st.query_params["code"])
  → handle_callback: exchange code → userinfo → upsert user → session_state
  → st.session_state: user_id, user_email, user_tier (+ guest counters reset)
Sidebar: avatar/email line + "Log out"
Logout: clear every session_state key + st.switch_page("app.py")
Guest: create_guest_session() sets user_id="guest_*" (no JWT)
```

State keys (only these): `user_email`, `user_id`, `user_tier`, `workspace`,
`messages`, `doc_statuses`, `quota`, `guest_queries`, `guest_docs`.

## 4. Design system

### 4.1 Base theme — `.streamlit/config.toml` (static)

```toml
[theme]
base = "light"
primaryColor = "#4F46E5"          # indigo-600
backgroundColor = "#FAFAFA"
secondaryBackgroundColor = "#F1F5F9"   # slate-100 (sidebar, inputs)
textColor = "#0F172A"                   # slate-900
borderColor = "#E2E8F0"
baseRadius = "8px"
```

Dark mode: **not v1**. Toggle later by flipping `base` + token overrides.

### 4.2 Semantic tokens — `styles/main.css`

```css
:root {
  --verify: #10B981;        /* emerald-500: verified badge, success states */
  --verify-bg: #ECFDF5;
  --citation: #6366F1;      /* indigo-500: [SOURCE N] chips */
  --citation-bg: #EEF2FF;
  --warn: #F59E0B;          /* quota ≥80%, processing */
  --error: #EF4444;         /* failed ingest, abstain notice */
  --border: #E2E8F0;
  --radius: 8px;
  --ws-accent: #4F46E5;     /* workspace accent, overridden per switcher */
}
[data-workspace="legal"]    { --ws-accent: #1E3A5F; }  /* navy */
[data-workspace="academic"] { --ws-accent: #0D9488; }  /* teal */
```

Workspace coupling: Chat/Documents pages wrap content and inject
`st.markdown(f"<style>:root{{--ws-accent: {hex}}}</style>")` keyed on
`st.session_state.workspace` (Streamlit theme itself cannot hot-swap — CSS override can).
Applies to: sidebar active states, chat send button, workspace badge, upload button.

### 4.3 Components

- **Cards/metrics/buttons:** `streamlit_shadcn_ui` (`ui.card`, `ui.metric_card`,
  `ui.link_button`, `ui.input`, `ui.textarea`) — pin compatible with streamlit `>=1.40`.
- **Icons:** **Material Symbols only — no emoji anywhere in the Streamlit UI**
  (toasts, labels, chips, badges, buttons). Examples: `:material/check_circle:`,
  `:material/scale:`, `:material/upload_file:`, `:material/error:`.
- **Typography:** Streamlit default (markdown hierarchy); no custom font v1.

## 5. Page inventory

| Page | Layout | Key components |
|---|---|---|
| **Login.py** | `layout="centered"` | logo, title, tagline ("ChatGPT guesses. We verify."), Google button, 3-line FAQ teaser, GitHub link |
| **1_Landing** | wide | hero (gradient band `#4F46E5→#7C3AED`, CTA→Login), logo cloud (Groq/LangChain/Chroma/Streamlit), 2-col alternating feature rows (Verification, Legal workspace, Academic workspace — screenshot placeholders), demo video slot, pricing cards (`ui.card` ×3, PR-5 activates links), FAQ expanders, contact `st.form` |
| **2_Chat** | wide, chat-first | sidebar: workspace switcher + doc list; main: `st.chat_message` history, citation expanders headed `[SOURCE N] · doc · p.N` with **verify badge** (PR-4b: claim-level pass/fail, trace link), `st.chat_input`, footer: latency + trace id |
| **3_Documents** | centered | uploader + `st.button(type="primary")`, per-doc row: status chip (processing/done/failed), inline `st.progress` while running, delete button, `st.status` polling loop for `processing → done` flips (toast on flip, see §6) |
| **4_Dashboard** | wide | `ui.metric_card` ×3 (Queries today/quota, Documents, Plan), 7-day usage chart (`st.bar_chart` v1, lightweight-charts later), answer history table → expandable provenance trace (PR-4b) |
| **5_Settings** | centered | account (email, sign-out), workspace default, provider keys (dev-only inputs), danger zone: delete account (POST, confirm) |

## 6. Toast / notification contract (native only)

No third-party notification deps. Native `st.toast` supports stacking, hover-pause,
update-token, `duration="short|long|infinite"` on installed Streamlit (1.58).

| Pattern | When | Example |
|---|---|---|
| Simple toast | single-step success/info | `st.toast("Workspace: Legal", icon=":material/scale:")` |
| Update-token | multi-step **within one run** | `msg = st.toast("Embedding…")` → `msg.toast("Indexing…")` → `msg.toast("Done", icon=":material/check_circle:")` |
| Transition toast | poll-loop detects state flip | `processing → done` fires once per flip in the detecting run |
| `duration="infinite"` | auth-critical | "Session expired — sign in again" (+ inline banner + redirect) |
| Inline `st.error/warning` | actionable failures | ingest failed, quota hard-stop — **always** paired with a toast, never toast-only |

Rules: max one toast per user action; icons = Material Symbols (**no emoji**); text ≤5 words;
never toast inside `st.cache_*`; errors stay visible inline after toast fades.

## 7. Phasing

| When | Deliverable |
|---|---|
| **This PR** | `docs/UI_DESIGN.md` only (no code) |
| **PR-6a** | `menu.py` + login gate + session-state auth (JWT later removed) |
| **PR-6b** | `pages/` restructure, shadcn components, `styles/main.css`, theme tokens, toasts |
| **PR-4b** | verify badge + provenance view (Chat + Dashboard) |
| **PR-5** | pricing cards activate (env-flag), upgrade toast |
| **PR-7** | screenshots/GIF from designed UI for README/release |

Split PR-6 if diff >400 lines (AGENTS PR contract).

## 8. Open decisions

- **Branding:** product name TBD ("Citation-Verified RAG SaaS" descriptive; final name TBD).
- Screenshot assets for Landing feature rows: capture after PR-6b.
- Reference repo uses deprecated `experimental_rerun` — audit any borrowed snippet against
  Streamlit 1.58 API before use.
