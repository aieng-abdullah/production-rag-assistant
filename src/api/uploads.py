"""Upload hardening (PLAN.md PR-3): magic bytes, filename sanitization.

Framework-free; raises `ValueError` with a user-facing message — the API
layer maps it to 400.
"""

import os
import re
from pathlib import Path

__all__ = ["PDF_MAGIC", "sanitize_filename", "is_pdf_magic", "claim_path"]

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


def claim_path(directory: Path, name: str) -> Path:
    """Atomically claim a free path (`O_CREAT|O_EXCL`).

    Closes the duplicate-name race: two concurrent uploads of `report.pdf`
    get `report.pdf` and `report (1).pdf` — the kernel arbitrates, not a
    exists()-then-write check. The empty file stays claimed; caller overwrites.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 0
    while True:
        candidate = directory / (name if n == 0 else f"{stem} ({n}){suffix}")
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return candidate
