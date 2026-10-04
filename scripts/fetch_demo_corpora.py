"""Fetch demo corpora for PLAN.md PR-4 (two niches, one engine).

academic/: 3 arXiv papers (Attention, BERT, GPT-3).
legal/:    Indian Evidence Act 1872 (Wikisource) rendered to PDF.

Indian Contract Act 1872 has no clean machine-readable source (absent from
Wikisource; archive.org scans are OCR noise), so the legal demo uses the
Evidence Act — same era and jurisdiction, heavy on exceptions and
illustrations, which exercises the legal profile's proviso rules.

Existing files are skipped; pass --force to re-download.
Run: python3 scripts/fetch_demo_corpora.py [--force]
"""

import argparse
import re
import sys
from pathlib import Path

import httpx
import pymupdf as fitz

DATA_DIR = Path("data/demo")

ARXIV_PAPERS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "bert.pdf": "1810.04805",
    "gpt3.pdf": "2005.11401",
}

WIKISOURCE_URL = (
    "https://en.wikisource.org/w/index.php"
    "?title=Indian_Evidence_Act_1872&action=raw"
)

_HEADERS = {"User-Agent": "rag-assistant-demo/0.4 (+github.com/aieng-abdullah)"}


def _get(url: str, timeout: float = 60.0) -> bytes:
    response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.content


def fetch_academic(force: bool) -> list[Path]:
    out_dir = DATA_DIR / "academic"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, arxiv_id in ARXIV_PAPERS.items():
        path = out_dir / name
        if path.exists() and not force:
            print(f"skip  {path} (exists)")
            saved.append(path)
            continue
        print(f"fetch arXiv {arxiv_id} -> {path}")
        path.write_bytes(_get(f"https://arxiv.org/pdf/{arxiv_id}"))
        saved.append(path)
    return saved


def _wiki_to_text(raw: str) -> str:
    """Strip Wikisource markup enough for readable demo PDF text."""
    text = raw
    text = re.sub(r"<ref[^>/]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    # Templates can nest ({{header|... {{no scan}} ...}}) — strip innermost out.
    while "{{" in text:
        stripped = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"^\s*\{\|.*?^\s*\|\}", "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^==+\s*(.*?)\s*==+$", r"\n\1\n", text, flags=re.MULTILINE)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_to_pdf(text: str, path: Path) -> None:
    doc = fitz.open()
    font, size, margin, leading = "helv", 10, 54, 13
    page = doc.new_page(width=595, height=842)  # A4
    y = margin
    for line in text.splitlines():
        if y > 842 - margin:
            page = doc.new_page(width=595, height=842)
            y = margin
        for wrapped in _wrap(line, 78) or [""]:
            page.insert_text((margin, y), wrapped, fontname=font, fontsize=size)
            y += leading
    doc.save(path)
    doc.close()


def _wrap(line: str, width: int) -> list[str]:
    """Greedy word wrap — never split mid-word (breaks tokenization)."""
    words = line.split()
    if not words:
        return []
    rows: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            rows.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        rows.append(current)
    return rows


def fetch_legal(force: bool) -> list[Path]:
    out_dir = DATA_DIR / "legal"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "indian_evidence_act_1872.pdf"
    if path.exists() and not force:
        print(f"skip  {path} (exists)")
        return [path]
    print(f"fetch Wikisource -> {path}")
    raw = _get(WIKISOURCE_URL).decode("utf-8", errors="replace")
    _text_to_pdf(_wiki_to_text(raw), path)
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch existing files")
    args = parser.parse_args()

    try:
        saved = fetch_academic(args.force) + fetch_legal(args.force)
    except httpx.HTTPError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    print(f"done: {len(saved)} demo PDFs under {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
