from src.ingestion.parser import extract_pages

PDF_PATH = "data/raw/How To Win Friends And Influence People - Carnegie, Dale.pdf"

pages = extract_pages(PDF_PATH)
print(f"✅ Extracted {len(pages)} pages")
print(f"✅ First page doc_id: {pages[0]['doc_id']}")
print(f"✅ First page number: {pages[0]['page_num']}")
print(f"✅ First 100 chars: {pages[0]['text'][:100]}")
