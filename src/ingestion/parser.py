from pathlib import Path
from typing import Optional
import fitz
from loguru import logger

MIN_PAGE_CHARS = 100


def extract_pages(pdf_path: str, doc_id: Optional[str] = None) -> list[dict]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    resolved_doc_id = doc_id or path.stem
    doc = fitz.open(str(path))
    total_pages = len(doc)

    logger.info(f"Parsing '{resolved_doc_id}' — {total_pages} pages")

    pages = []
    skipped = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if len(text) < MIN_PAGE_CHARS:
            skipped += 1
            continue

        pages.append(
            {
                "text": text,
                "page_num": page_num,
                "doc_id": resolved_doc_id,
                "source": str(path),
                "total_pages": total_pages,
            }
        )

    doc.close()
    logger.info(f"Extracted {len(pages)} pages ({skipped} skipped)")
    return pages
