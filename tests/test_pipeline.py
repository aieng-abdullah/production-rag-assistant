from src.ingestion.pipeline import ingest

PDF_PATH = "data/raw/How To Win Friends And Influence People - Carnegie, Dale.pdf"

result = ingest(PDF_PATH)

print(f"✅ Pages extracted: {result['pages']}")
print(f"✅ Chunks saved to ChromaDB: {result['chunks']}")
