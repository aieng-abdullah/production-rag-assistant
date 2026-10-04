"""FastAPI app factory (PLAN.md PR-2b).

Routers: /auth now; /documents, /chat, /usage land in PR-3.
No LLM-key validation here — Streamlit's `app.py` owns `Config.validate()`;
this process must boot even when only auth endpoints are exercised.
"""

from fastapi import FastAPI

from src.api.auth import router as auth_router

__all__ = ["create_app"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Citation-Verified RAG API",
        version="0.1.0",
        docs_url="/docs",
    )
    app.include_router(auth_router)
    return app


app = create_app()
