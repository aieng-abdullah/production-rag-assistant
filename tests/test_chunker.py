from src.ingestion.parser import extract_pages
from src.ingestion.chunker import chunk_pages

PDF_PATH = "data/raw/How To Win Friends And Influence People - Carnegie, Dale.pdf"

pages = extract_pages(PDF_PATH)
chunks = chunk_pages(pages)

print(f"Total chunks: {len(chunks)}")
print(f"First chunk doc_id: {chunks[0]['doc_id']}")
print(f"Sample text: {chunks[0]['text'][:80]}")
