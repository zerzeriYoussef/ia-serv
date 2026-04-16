"""Quick script to check if ChromaDB has any collections / chunks for dataset 3."""
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")

# List all collections
collections = client.list_collections()
print(f"Collections found: {len(collections)}")
for c in collections:
    print(f"  - {c.name}: {c.count()} chunks")

# Try to get dataset_3 specifically
try:
    col = client.get_collection("dataset_3")
    count = col.count()
    print(f"\ndataset_3 collection: {count} chunks")
    if count > 0:
        peek = col.peek(limit=3)
        print("Sample chunk IDs:", peek["ids"])
    else:
        print("Collection is EMPTY — no chunks indexed!")
except Exception as e:
    print(f"\nCould not get dataset_3 collection: {e}")
    print("=> The index was NEVER built. Call POST /datasets/3/index first.")
