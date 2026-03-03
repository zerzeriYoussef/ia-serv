"""Test upload functionality"""
import httpx
import asyncio


async def test_upload():
    """Test file upload"""
    
    # Create a test CSV file
    test_csv = "test_data.csv"
    with open(test_csv, 'w') as f:
        f.write("name,age,salary\n")
        f.write("Alice,25,50000\n")
        f.write("Bob,30,60000\n")
        f.write("Charlie,35,70000\n")
    
    # Upload file
    async with httpx.AsyncClient() as client:
        with open(test_csv, 'rb') as f:
            response = await client.post(
                "http://localhost:8000/api/v1/upload",
                files={"file": (test_csv, f, "text/csv")}
            )
        
        print(f"Upload Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 201:
            dataset_id = response.json()["id"]
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Get dataset details
            details = await client.get(f"http://localhost:8000/api/v1/datasets/{dataset_id}")
            print(f"\nDataset Details: {details.json()}")
    
    # Clean up
    import os
    os.remove(test_csv)


if __name__ == "__main__":
    asyncio.run(test_upload())