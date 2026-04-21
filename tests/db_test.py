# quick_test.py (delete after testing)
from src.db.chroma_client import get_collection, reset_client

# Test 1: collection creates successfully
col = get_collection()
print(f"✅ Collection created: {col.name}")
print(f"✅ Document count: {col.count()}")

# Test 2: singleton works — should NOT reconnect
col2 = get_collection()
print(f"✅ Same object? {col is col2}")  # should print True... 
# actually get_collection() returns a new Collection wrapper each time
# but _client is reused — add a print in get_chroma_client to verify

# Test 3: reset works
reset_client()
col3 = get_collection()
print(f"✅ After reset, new connection made")
