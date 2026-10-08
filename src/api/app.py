"""FastAPI app factory (PLAN.md PR-2b/PR-3).

Routers: /auth, /documents, /chat, /usage now; workspaces + verifier next.
`Config.validate()` runs in the lifespan — the API process boots loud when
required keys are missing (PLAN-render-react Phase 3), instead of failing
mid-request with a cryptic provider error.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.answers import router as answers_router
from src.api.auth import router as auth_router
from src.api.chat import router as chat_router
from src.api.documents import recover_stale_documents, router as documents_router
from src.api.rate_limit import RateLimitMiddleware
from src.api.usage import router as usage_router
from src.config import Config

__all__ = ["create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Config.validate()
    recover_stale_documents()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Citation-Verified RAG API",
        version="0.2.0",
        docs_url="/docs",
        lifespan=lifespan,
    )
    # Inner: rejects over-quota clients before routing. OPTIONS preflight is
    # exempted inside the middleware so CORS negotiation never 429s.
    app.add_middleware(RateLimitMiddleware)
    # CORS outermost: 429/500 responses below still carry the headers the SPA
    # needs to read their bodies. One static origin (Render Static Site).
    # Bearer-token auth → no cookies, so credentials stay off.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[Config.FRONTEND_URL.rstrip("/")],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness probe for Render's health check path — no downstream calls."""
        return {"status": "ok"}

    app.include_router(auth_router)
    # PLAN PR-6: demo sign-in exists only while demo mode resolves on —
    # deployments with Google creds configured never expose /auth/demo.
    if Config.ENABLE_DEMO_LOGIN == "on":
        from src.api.auth import demo_router

        app.include_router(demo_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(answers_router)
    app.include_router(usage_router)
    # Admin routes exist only when ADMIN_EMAILS is configured — flag-off
    # deployments must never expose /admin (fail-closed, same pattern as billing).
    if Config.ADMIN_EMAILS:
        from src.api.admin import router as admin_router

        app.include_router(admin_router)
    # PLAN PR-5: billing routes exist only when the Stripe flag is on —
    # flag-off deployments must never expose /billing or /webhooks.
    if Config.STRIPE_SECRET_KEY:
        from src.api.billing import router as billing_router
        from src.api.billing import webhook_router as stripe_webhook_router

        app.include_router(billing_router)
        app.include_router(stripe_webhook_router)
    return app


app = create_app()
