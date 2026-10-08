"""GET /answers/{id}/trace (PLAN PR-4b-iii): shape, ownership, auth."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import Answer, AnswerTrace
from src.generation.Citation_system import CitedAnswer, Source
from src.services import RAGService
from src.services.bm25_cache import clear_all

FAKE_TRACE = {
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
    "verification": {
        "status": "verified",
        "per_claim": [
            {"claim": 0, "text": "claim", "verdict": "SUPPORTED", "reason": "ok"}
        ],
    },
}

FAKE_ANSWER = CitedAnswer(
    answer="[SOURCE 1] The contract voids under section 23.",
    sources=[Source(doc_id="contract_act", page_num=3, text="An agreement...")],
    verification=FAKE_TRACE["verification"],
    trace=FAKE_TRACE,
)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'answers.db'}")
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


def _seed_answer(user_id: int, query: str = "what is void?") -> int:
    with session_scope() as session:
        answer = Answer(user_id=user_id, query=query, answer="text [SOURCE 1]")
        session.add(answer)
        session.flush()
        session.add(AnswerTrace(answer_id=answer.id, payload=dict(FAKE_TRACE)))
        return answer.id


def test_trace_requires_auth(client):
    assert client.get("/answers/1/trace").status_code == 401


def test_trace_unknown_id_404(client, headers):
    assert client.get("/answers/999/trace", headers=headers).status_code == 404


def test_trace_other_tenant_404(client, headers):
    """Foreign answer id must look identical to unknown — no existence leak."""
    foreign_id = _seed_answer(user_id=2)

    assert client.get(f"/answers/{foreign_id}/trace", headers=headers).status_code == 404


def test_chat_then_trace_roundtrip(client, headers, monkeypatch):
    monkeypatch.setattr(
        "src.api.chat.get_bm25", lambda tenant, workspace=None: None
    )
    monkeypatch.setattr(
        RAGService,
        "generate_answer",
        lambda self, tenant, query, bm25_index=None, provider_overrides=None,
        workspace="academic", history=None: FAKE_ANSWER,
    )

    chat = client.post("/chat", json={"query": "q"}, headers=headers).json()
    assert chat["answer_id"] is not None

    response = client.get(f"/answers/{chat['answer_id']}/trace", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["query"] == "q"
    assert body["answer"]["id"] == chat["answer_id"]
    trace = body["trace"][0]
    assert trace["prompt_version"] == "legal-v2"
    assert trace["chunks"][0]["cited"] is True
    assert trace["claims"][0]["citations"][0]["source_id"] == 1
    assert trace["verification"]["status"] == "verified"
    assert trace["token_usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_trace_shape_keys(client, headers):
    answer_id = _seed_answer(user_id=1)

    body = client.get(f"/answers/{answer_id}/trace", headers=headers).json()

    assert set(body) == {"answer", "trace"}
    assert set(body["answer"]) == {"id", "query", "answer", "created_at"}
    trace = body["trace"][0]
    for key in (
        "workspace",
        "prompt_version",
        "model",
        "verify_model",
        "token_usage",
        "chunks",
        "claims",
        "abstained",
        "verification",
    ):
        assert key in trace


def test_chat_survives_persist_failure(client, headers, monkeypatch):
    """Trace outage degrades to answer_id: null — never 500s."""
    monkeypatch.setattr(
        "src.api.chat.get_bm25", lambda tenant, workspace=None: None
    )
    monkeypatch.setattr(
        RAGService,
        "generate_answer",
        lambda self, tenant, query, bm25_index=None, provider_overrides=None,
        workspace="academic", history=None: FAKE_ANSWER,
    )
    monkeypatch.setattr(
        "src.api.chat.persist_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )

    response = client.post("/chat", json={"query": "q"}, headers=headers)

    assert response.status_code == 200
    assert response.json()["answer_id"] is None
