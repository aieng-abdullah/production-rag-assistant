"""Document endpoints (PLAN.md PR-3): upload, list, status poll, delete.

Uploads land in `data/raw/{user_id}/` and ingest via FastAPI BackgroundTasks;
`status=processing` → `ready` | `failed` for polling. Quotas and upload
hardening run before any disk write.
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from loguru import logger

from src.api.deps import quota_to_http, require_user
from src.api.uploads import claim_path, is_pdf_magic, sanitize_filename
from src.config import Config
from src.db.database import session_scope
from src.db.models import Document
from src.services import RAGService
from src.services.bm25_cache import invalidate
from src.services.quotas import (
    STORAGE_LIMIT_BYTES,
    QuotaExceeded,
    check_document_quota,
    enforce_storage_quota,
    reserve_document_slot,
)

__all__ = ["router"]

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_CAP = STORAGE_LIMIT_BYTES + 1  # enough to detect over-quota, nothing more


def _raw_dir(user_id: int) -> Path:
    return Config.DATA_DIR / "raw" / str(user_id)


def _ingest(tenant_id: str, path: Path) -> None:
    """Seam: tests monkeypatch this instead of running the embedder."""
    RAGService().ingest(tenant_id, path)


def _delete_chunks(tenant_id: str, doc_id: str) -> None:
    """Seam: tests monkeypatch this instead of touching ChromaDB."""
    RAGService().delete_document(tenant_id, doc_id)


def _set_status(document_id: int, status_value: str) -> None:
    with session_scope() as session:
        row = session.get(Document, document_id)
        if row is not None:
            row.status = status_value


def recover_stale_documents() -> int:
    """Startup sweep: orphaned `processing` rows (worker crash mid-ingest)
    become `failed` so pollers don't hang forever. Returns rows recovered."""
    with session_scope() as session:
        stale = (
            session.query(Document).filter(Document.status == "processing").count()
        )
        if stale:
            (
                session.query(Document)
                .filter(Document.status == "processing")
                .update({"status": "failed"}, synchronize_session=False)
            )
    if stale:
        logger.warning("Recovered {stale} stale processing documents", stale=stale)
    return stale


def process_document(user_id: int, document_id: int, path: Path) -> None:
    """BackgroundTasks callback — ingest, then mark ready/failed and refresh BM25."""
    tenant = str(user_id)
    try:
        _ingest(tenant, path)
    except Exception as exc:
        logger.error(f"Ingest failed doc={path.name} tenant={tenant}: {exc}")
        _set_status(document_id, "failed")
        return
    _set_status(document_id, "ready")
    invalidate(tenant)
    logger.info(f"Document ready doc={path.name} tenant={tenant}")


def _document_payload(row: Document) -> dict:
    return {
        "id": row.id,
        "filename": row.filename,
        "status": row.status,
        "created_at": row.created_at.isoformat(),
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Depends(require_user),
) -> dict:
    """Save PDF, queue ingestion, return `processing` status for polling."""
    try:
        filename = sanitize_filename(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    payload = file.file.read(_READ_CAP)
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
    if not is_pdf_magic(payload[:5]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not a PDF (magic bytes check failed)",
        )

    try:
        with session_scope() as session:
            check_document_quota(session, user_id, len(payload))
    except QuotaExceeded as exc:
        raise quota_to_http(exc) from exc

    directory = _raw_dir(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = claim_path(directory, filename)
    path.write_bytes(payload)

    # Enforcement after the write: shared filesystem state closes the
    # pre-write TOCTOU window; conditional insert enforces the 5-doc cap.
    try:
        enforce_storage_quota(user_id)
        with session_scope() as session:
            document_id = reserve_document_slot(session, user_id, path.name)
    except QuotaExceeded as exc:
        path.unlink(missing_ok=True)
        raise quota_to_http(exc) from exc

    background_tasks.add_task(process_document, user_id, document_id, path)
    logger.info(f"Upload queued doc={path.name} tenant={user_id} bytes={len(payload)}")
    return {"id": document_id, "filename": path.name, "status": "processing"}


@router.get("")
def list_documents(user_id: int = Depends(require_user)) -> list[dict]:
    with session_scope() as session:
        rows = (
            session.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return [_document_payload(row) for row in rows]


@router.get("/{document_id}")
def get_document(document_id: int, user_id: int = Depends(require_user)) -> dict:
    with session_scope() as session:
        row = session.get(Document, document_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return _document_payload(row)


@router.delete("/{document_id}")
def delete_document(document_id: int, user_id: int = Depends(require_user)) -> dict:
    """Remove chunks, file, and DB row — all scoped to this tenant.

    Order matters: chunks first, row last. A vector-store failure leaves
    everything intact (retryable); a row-first order would orphan chunks.
    """
    with session_scope() as session:
        row = session.get(Document, document_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )
        filename = row.filename

    doc_id = Path(filename).stem
    try:
        _delete_chunks(str(user_id), doc_id)
    except Exception as exc:
        logger.error(f"Chunk delete failed doc={doc_id} tenant={user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store delete failed",
        ) from exc

    file_path = _raw_dir(user_id) / Path(filename).name
    file_path.unlink(missing_ok=True)
    with session_scope() as session:
        row = session.get(Document, document_id)
        if row is not None:
            session.delete(row)
    invalidate(str(user_id))
    logger.info(f"Document deleted doc={filename} tenant={user_id}")
    return {"deleted": filename}
