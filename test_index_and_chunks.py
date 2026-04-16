"""
Test script: build the Chroma index for dataset 3, then verify chunk retrieval.
"""
import asyncio
import sys

async def main():
    from app.core.database import AsyncSessionLocal
    from app.services.rag.dataset_indexer import index_dataset, retrieve_chunks

    async with AsyncSessionLocal() as db:
        dataset_id = 3

        print(f"[1/3] Building Chroma index for dataset {dataset_id}...")
        try:
            n = await index_dataset(db, dataset_id)
            print(f"       Indexed {n} chunks successfully!")
        except Exception as e:
            print(f"       Index build failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

        print(f"\n[2/3] Testing retrieval: 'coffee vs tea spending'...")
        try:
            chunks = await retrieve_chunks(dataset_id, "coffee vs tea spending", top_k=5)
            print(f"       Retrieved {len(chunks)} chunks")
            for c in chunks:
                cid = c['chunk_id']
                dist = c['distance']
                text = c['text'][:80]
                print(f"         - {cid} (dist={dist:.4f}): {text}...")
        except Exception as e:
            print(f"       Retrieval failed: {e}")
            sys.exit(1)

        print(f"\n[3/3] Verifying collection state...")
        import chromadb
        from app.core.config import settings
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        col = client.get_collection(f"dataset_{dataset_id}")
        print(f"       Collection has {col.count()} chunks total")

    print("\nDONE - chunk_ids should now populate on your next chat message!")

if __name__ == "__main__":
    asyncio.run(main())
