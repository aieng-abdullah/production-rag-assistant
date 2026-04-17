"""PDF text extraction with PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

from src.monitoring.logger import get_logger

logger = get_logger("parser")


def parse_pdf(file_path: str | Path) -> List[Dict]:
    """Extract text from PDF preserving page numbers and metadata."""
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        logger.error(f"PDF not found: {file_path}")
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    pages = []
    
    try:
        # Open PDF document
        with fitz.open(str(file_path)) as doc:
            logger.info(f"Parsing PDF: {file_path.name}, pages: {len(doc)}")
            
            for page_num, page in enumerate(doc, start=1):
                # Extract text from page
                text = page.get_text().strip()
                
                # Skip empty pages
                if not text:
                    continue
                
                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "char_count": len(text)
                })
        
        logger.info(f"Extracted {len(pages)} non-empty pages")
        return pages
        
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise ValueError(f"PDF parsing failed: {e}")
