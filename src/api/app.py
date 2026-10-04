"""FastAPI app factory (PLAN.md PR-2b/PR-3).

Routers: /auth, /documents, /chat, /usage now; workspaces + verifier next.
No LLM-key validation here — Streamlit's `app.py` owns `Config.validate()`;
this process must boot even when only auth endpoints are exercised.
"""

from fastapi import FastAPI

from src.api.auth import router as auth_router
from src.api.chat import router as chat_router
from src.api.documents import router as documents_router
from src.api.usage import router as usage_router

__all__ = ["create_app"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Citation-Verified RAG API",
        version="0.2.0",
        docs_url="/docs",
    )
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(usage_router)
    return app


app = create_app()
