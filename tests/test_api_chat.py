"""Chat + usage endpoint tests (PLAN.md PR-3): auth, quota 429, response
shape parity with the Streamlit app, usage recording on success only."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import UsageEvent
from src.generation.Citation_system import CitedAnswer, Source
from src.services import RAGService
from src.services.bm25_cache import clear_all

FAKE_ANSWER = CitedAnswer(
    answer="[SOURCE 1] The contract voids under section 23.",
    sources=[Source(doc_id="contract_act", page_num=3, text="An agreement...")],
    verification={
        "status": "verified",
        "per_claim": [
            {
                "claim": 0,
                "text": "The contract voids under section 23.",
                "verdict": "SUPPORTED",
                "reason": "entailed",
            }
        ],
    },
    trace={
        "workspace": "legal",
        "prompt_version": "legal-v2",
        "model": "test-model",
        "verify_model": "test-judge",
        "token_usage": {"input_tokens": 10, "output_tokens": 5},
        "chunks": [
            {"source_id": 1, "doc_id": "contract_act", "page_num": 3, "cited": True}
        ],
        "claims": [
            {
                "text": "The contract voids under section 23.",
                "citations": [{"source_id": 1, "quote": "An agreement..."}],
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "verification": {"status": "verified"},
    },
)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'chat.db'}")
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        Config, "JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef"
    )
    reset_engine()
    Base.metadata.create_all(get_engine())
    clear_all()
    yield create_app()
    reset_engine()
    clear_all()


@pytest.fixture()
def client(api):
    return TestClient(api)


@pytest.fixture()
def headers():
    return {"Authorization": f"Bearer {create_token(1)}"}


def _fake_generate(monkeypatch, cited: CitedAnswer = FAKE_ANSWER):
    monkeypatch.setattr(
        "src.api.chat.get_bm25", lambda tenant, workspace=None: None
    )
    monkeypatch.setattr(
        RAGService,
        "generate_answer",
        lambda self, tenant, query, bm25_index=None, provider_overrides=None,
        workspace="academic": cited,
    )


def _seed_query_events(count: int, age_days: int = 0) -> None:
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    with session_scope() as session:
        for _ in range(count):
            session.add(
                UsageEvent(user_id=1, kind="query", units=1, created_at=created)
            )


def test_chat_requires_auth(client):
    response = client.post("/chat", json={"query": "what?"})

    assert response.status_code == 401


def test_chat_empty_query_422(client, headers):
    response = client.post("/chat", json={"query": ""}, headers=headers)

    assert response.status_code == 422


def test_chat_returns_app_shape(client, headers, monkeypatch):
    _fake_generate(monkeypatch)

    response = client.post("/chat", json={"query": "void contract?"}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_ANSWER.answer
    assert body["sources"] == [
        {"doc_id": "contract_act", "page_num": 3, "text": "An agreement..."}
    ]
    assert body["verification"]["status"] == "verified"
    assert body["verification"]["per_claim"][0]["verdict"] == "SUPPORTED"
    assert body["answer_id"] == 1


def test_chat_forwards_legal_workspace(client, headers, monkeypatch):
    """PR-4: request workspace reaches both bm25 index and generate_answer."""
    seen: dict = {}

    def fake_get_bm25(tenant, workspace=None):
        seen["bm25"] = (tenant, workspace)
        return None

    def fake_generate(
        self, tenant, query, bm25_index=None, provider_overrides=None,
        workspace="academic",
    ):
        seen["gen"] = workspace
        return FAKE_ANSWER

    monkeypatch.setattr("src.api.chat.get_bm25", fake_get_bm25)
    monkeypatch.setattr(RAGService, "generate_answer", fake_generate)

    response = client.post(
        "/chat", json={"query": "q", "workspace": "legal"}, headers=headers
    )

    assert response.status_code == 200
    assert seen == {"bm25": ("1", "legal"), "gen": "legal"}


def test_chat_rejects_unknown_workspace(client, headers, monkeypatch):
    _fake_generate(monkeypatch)

    response = client.post(
        "/chat", json={"query": "q", "workspace": "medical"}, headers=headers
    )

    assert response.status_code == 422


def test_chat_records_usage_on_success(client, headers, monkeypatch):
    _fake_generate(monkeypatch)

    client.post("/chat", json={"query": "q"}, headers=headers)
    usage = client.get("/usage", headers=headers).json()

    # Verified answer costs query + verify = 2 units.
    assert usage["queries"]["used"] == 2


def test_chat_429_when_daily_quota_exhausted(client, headers, monkeypatch):
    _fake_generate(monkeypatch)
    _seed_query_events(20)

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 429
    assert "20 queries/day" in response.json()["detail"]
    assert response.headers["Retry-After"]


def test_chat_quota_counts_today_only(client, headers, monkeypatch):
    _fake_generate(monkeypatch)
    _seed_query_events(18, age_days=0)
    _seed_query_events(5, age_days=1)  # yesterday — must not count

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 200
    assert client.get("/usage", headers=headers).json()["queries"]["used"] == 20


def test_chat_429_when_only_one_unit_left(client, headers, monkeypatch):
    """Verified answers need 2 units — 19/20 used must reject the 20th."""
    _fake_generate(monkeypatch)
    _seed_query_events(19)

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 429
    assert client.get("/usage", headers=headers).json()["queries"]["used"] == 19


def test_chat_failed_generation_502_and_free(client, headers, monkeypatch):
    monkeypatch.setattr(
        "src.api.chat.get_bm25", lambda tenant, workspace=None: None
    )

    def boom(self, tenant, query, bm25_index=None, provider_overrides=None,
             workspace="academic"):
        raise RuntimeError("llm down")

    monkeypatch.setattr(RAGService, "generate_answer", boom)

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 502
    # Failed generations don't burn quota.
    assert client.get("/usage", headers=headers).json()["queries"]["used"] == 0


def test_usage_shape_and_auth(client, headers):
    response = client.get("/usage", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["queries"] == {"used": 0, "limit": 20}
    assert body["documents"] == {"used": 0, "limit": 5}
    assert body["storage"] == {
        "used_bytes": 0,
        "limit_bytes": 100 * 1024 * 1024,
    }


def test_usage_requires_auth(client):
    assert client.get("/usage").status_code == 401
