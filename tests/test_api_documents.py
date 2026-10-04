"""Document endpoint tests (PLAN.md PR-3): upload hardening, quotas,
status transitions, tenant scoping, delete."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.security import create_token
from src.config import Config
from src.db.database import Base, get_engine, reset_engine, session_scope
from src.db.models import Document
from src.services.bm25_cache import clear_all

PDF_BYTES = b"%PDF-1.4\n1 0 obj\nendobj\ntrailer\n%%EOF"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'docs.db'}")
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


def _upload(client, headers, name="paper.pdf", content=PDF_BYTES):
    return client.post(
        "/documents",
        files={"file": (name, content, "application/pdf")},
        headers=headers,
    )


def _seed_document(
    user_id: int, filename: str = "seed.pdf", status: str = "ready"
) -> int:
    with session_scope() as session:
        row = Document(user_id=user_id, filename=filename, status=status)
        session.add(row)
        session.flush()
        return row.id


def test_upload_requires_auth(client):
    response = client.post(
        "/documents", files={"file": ("x.pdf", PDF_BYTES, "application/pdf")}
    )

    assert response.status_code == 401


def test_upload_rejects_non_pdf_magic(client, headers):
    response = _upload(client, headers, name="fake.pdf", content=b"MZ not a pdf")

    assert response.status_code == 400
    assert "magic" in response.json()["detail"]


def test_upload_rejects_empty_file(client, headers):
    response = _upload(client, headers, name="empty.pdf", content=b"")

    assert response.status_code == 400


def test_upload_rejects_unsalvageable_filename(client, headers):
    response = _upload(client, headers, name="\u2603\u2603.pdf", content=PDF_BYTES)

    assert response.status_code == 400


def test_upload_rejects_non_pdf_extension(client, headers):
    response = _upload(client, headers, name="notes.txt", content=PDF_BYTES)

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_upload_sanitizes_traversal_path(client, headers, api, monkeypatch):
    monkeypatch.setattr("src.api.documents._ingest", lambda tenant, path, workspace=None: None)

    response = _upload(client, headers, name="sub/../../evil report.pdf")

    assert response.status_code == 202
    assert response.json()["filename"] == "evil_report.pdf"
    data_dir = Config.DATA_DIR
    # Ingest no-op succeeded → raw file purged (PR-3b), never written outside raw/.
    assert not (data_dir / "raw" / "1" / "evil_report.pdf").exists()
    assert not (data_dir / "sub").exists()


def test_upload_marks_ready_after_background_ingest(client, headers, monkeypatch):
    monkeypatch.setattr("src.api.documents._ingest", lambda tenant, path, workspace=None: None)

    response = _upload(client, headers)

    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    # TestClient runs background tasks inline — poll returns the final state.
    polled = client.get(f"/documents/{response.json()['id']}", headers=headers)
    assert polled.status_code == 200
    assert polled.json()["status"] == "ready"
    # Chunks live → raw PDF purged (PR-3b).
    assert not (Config.DATA_DIR / "raw" / "1" / "paper.pdf").exists()


def test_upload_forwards_workspace_to_ingest(client, headers, monkeypatch):
    """PR-4: form field `workspace` flows into the ingest seam."""
    seen: dict = {}

    def fake_ingest(tenant, path, workspace=None):
        seen["ws"] = workspace

    monkeypatch.setattr("src.api.documents._ingest", fake_ingest)

    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
        data={"workspace": "legal"},
        headers=headers,
    )

    assert response.status_code == 202
    assert seen["ws"] == "legal"


def test_upload_rejects_unknown_workspace(client, headers, monkeypatch):
    monkeypatch.setattr(
        "src.api.documents._ingest",
        lambda tenant, path, workspace=None: None,
    )

    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
        data={"workspace": "medical"},
        headers=headers,
    )

    assert response.status_code == 422


def test_upload_marks_failed_when_ingest_raises(client, headers, monkeypatch):
    def boom(tenant, path, workspace=None):
        raise RuntimeError("corrupt pdf")

    monkeypatch.setattr("src.api.documents._ingest", boom)

    response = _upload(client, headers)
    polled = client.get(f"/documents/{response.json()['id']}", headers=headers)

    assert polled.json()["status"] == "failed"
    # Failed ingest keeps the raw file for retry/debug (PR-3b).
    assert (Config.DATA_DIR / "raw" / "1" / "paper.pdf").is_file()


def test_upload_quota_document_limit(client, headers):
    for n in range(5):
        _seed_document(user_id=1, filename=f"doc{n}.pdf")

    response = _upload(client, headers)

    assert response.status_code == 429
    assert "Document quota" in response.json()["detail"]


def test_upload_quota_storage_limit(client, headers, monkeypatch):
    monkeypatch.setattr("src.services.quotas.STORAGE_LIMIT_BYTES", 0)

    response = _upload(client, headers)

    assert response.status_code == 429
    assert "Storage quota" in response.json()["detail"]


def test_post_write_storage_recheck_catches_race(client, headers, monkeypatch):
    """Pre-check blind (simulates concurrent writer) → post-write check trips."""
    monkeypatch.setattr(
        "src.api.documents.check_document_quota", lambda session, uid, size: None
    )
    monkeypatch.setattr("src.services.quotas.STORAGE_LIMIT_BYTES", 0)

    response = _upload(client, headers)

    assert response.status_code == 429
    assert not (Config.DATA_DIR / "raw" / "1" / "paper.pdf").exists()


def test_conditional_insert_enforces_doc_limit(client, headers, monkeypatch):
    """Pre-check blind (race) → atomic count+insert rejects, file rolled back."""
    monkeypatch.setattr(
        "src.api.documents.check_document_quota", lambda session, uid, size: None
    )
    for n in range(5):
        _seed_document(user_id=1, filename=f"doc{n}.pdf")

    response = _upload(client, headers)

    assert response.status_code == 429
    assert "Document quota" in response.json()["detail"]
    assert not (Config.DATA_DIR / "raw" / "1" / "paper.pdf").exists()


def test_startup_marks_stale_processing_as_failed(api, headers):
    """Orphaned processing rows (crashed worker) fail loudly at boot."""
    _seed_document(user_id=1, filename="stuck.pdf", status="processing")

    with TestClient(api) as booting_client:
        rows = booting_client.get("/documents", headers=headers).json()

    assert rows[0]["status"] == "failed"


def test_duplicate_filename_reused_after_purge(client, headers, monkeypatch):
    """PR-3b: successful ingest removes the raw file, freeing its name."""
    monkeypatch.setattr("src.api.documents._ingest", lambda tenant, path, workspace=None: None)

    first = _upload(client, headers, name="dup.pdf")
    second = _upload(client, headers, name="dup.pdf")

    assert first.json()["filename"] == "dup.pdf"
    assert second.json()["filename"] == "dup.pdf"


def test_duplicate_filename_gets_unique_path_while_file_exists(client, headers, monkeypatch):
    """A kept file (failed ingest) still collides → claim_path suffixes it."""
    def boom(tenant, path, workspace=None):
        raise RuntimeError("ingest down")

    monkeypatch.setattr("src.api.documents._ingest", boom)

    first = _upload(client, headers, name="dup.pdf")
    second = _upload(client, headers, name="dup.pdf")

    assert first.json()["filename"] == "dup.pdf"
    assert second.json()["filename"] == "dup (1).pdf"


def test_list_documents_tenant_scoped(client, headers):
    mine = _seed_document(user_id=1, filename="mine.pdf")
    _seed_document(user_id=2, filename="theirs.pdf")

    response = client.get("/documents", headers=headers)

    assert response.status_code == 200
    listed = [row["id"] for row in response.json()]
    assert listed == [mine]


def test_get_document_404_for_other_tenant(client, headers):
    theirs = _seed_document(user_id=2, filename="theirs.pdf")

    response = client.get(f"/documents/{theirs}", headers=headers)

    assert response.status_code == 404


def test_delete_document_removes_row_and_file(client, headers, monkeypatch):
    monkeypatch.setattr("src.api.documents._ingest", lambda tenant, path, workspace=None: None)
    monkeypatch.setattr("src.api.documents._delete_chunks", lambda tenant, doc: None)
    uploaded = _upload(client, headers, name="gone.pdf")

    response = client.delete(f"/documents/{uploaded.json()['id']}", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"deleted": "gone.pdf"}
    # Raw file was already purged at ingest (PR-3b); delete stays idempotent.
    assert not (Config.DATA_DIR / "raw" / "1" / "gone.pdf").exists()
    assert client.get("/documents", headers=headers).json() == []


def test_delete_document_404_for_other_tenant(client, headers):
    theirs = _seed_document(user_id=2, filename="theirs.pdf")

    response = client.delete(f"/documents/{theirs}", headers=headers)

    assert response.status_code == 404


def test_delete_survives_vectorstore_failure(client, headers, monkeypatch):
    def chunks_boom(tenant, doc_id):
        raise RuntimeError("chroma down")

    # Failed ingest keeps the raw file — the retryable state this test guards.
    def ingest_boom(tenant, path, workspace=None):
        raise RuntimeError("ingest down")

    monkeypatch.setattr("src.api.documents._ingest", ingest_boom)
    monkeypatch.setattr("src.api.documents._delete_chunks", chunks_boom)
    uploaded = _upload(client, headers)

    response = client.delete(f"/documents/{uploaded.json()['id']}", headers=headers)

    assert response.status_code == 502
    # Row and file must survive — delete is retryable.
    assert len(client.get("/documents", headers=headers).json()) == 1
    assert (Config.DATA_DIR / "raw" / "1" / "paper.pdf").is_file()
