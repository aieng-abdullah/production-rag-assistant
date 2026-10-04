"""Upload hardening (PLAN.md PR-3): magic bytes, filename sanitization.

Framework-free; raises `ValueError` with a user-facing message — the API
layer maps it to 400.
"""

import re
from pathlib import Path

__all__ = ["PDF_MAGIC", "sanitize_filename", "is_pdf_magic", "unique_path"]

PDF_MAGIC = b"%PDF-"
MAX_FILENAME_LEN = 200
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(raw: str) -> str:
    """Strip directories, drop unsafe chars, require `.pdf` suffix.

    Blocks traversal (path separators gone), hidden-dot names, and
    empty/degenerate names. Raises ValueError when nothing usable remains.
    """
    name = Path(raw.replace("\\", "/")).name  # drop any directory components
    name = _UNSAFE.sub("", name).strip()
    name = name.lstrip(".")  # no hidden files
    name = re.sub(r"\s+", "_", name)  # spaces → underscore (clean doc_id)
    if not name:
        raise ValueError("Filename is empty after sanitization")
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only .pdf files are accepted")
    if len(name) > MAX_FILENAME_LEN:
        name = name[: MAX_FILENAME_LEN - 4].rstrip(".") + ".pdf"
    return name


def is_pdf_magic(head: bytes) -> bool:
    """Content check — Content-Type headers are advisory only."""
    return head.startswith(PDF_MAGIC)


def unique_path(directory: Path, name: str) -> Path:
    """Avoid overwriting an existing upload: `report.pdf` → `report (1).pdf`."""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while True:
        candidate = directory / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1
