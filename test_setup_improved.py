"""
Fresh test setup for AI Service - Clean and Simple
Run this to verify your entire environment is ready
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🔍 AI SERVICE - FRESH SETUP TEST")
print("=" * 70)

# Test 1: Check Python environment
print("\n📌 [1] Python Environment:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   Path: {sys.executable}")
print(f"   Encoding: {sys.getdefaultencoding()}")

# Test 2: Check if we can import settings
print("\n📌 [2] Loading Configuration:")
try:
    from app.core.config import settings
    print("   ✅ Settings loaded successfully")
    print(f"   📁 Project root: {Path(__file__).parent}")
    print(f"   🗄️  Database: {settings.DATABASE_URL}")
    print(f"   📦 Redis: {settings.REDIS_URL}")
except Exception as e:
    print(f"   ❌ Failed to load settings: {e}")
    sys.exit(1)

# Test 3: Test PostgreSQL connection (simple)
print("\n📌 [3] Testing PostgreSQL Connection:")
try:
    import psycopg2
    
    # Simple direct connection
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ai_service",
        user="postgres",
        password="admin"
    )
    
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()
    cur.close()
    conn.close()
    
    print("   ✅ PostgreSQL connected successfully!")
    print(f"   📊 Version: {version[0][:60]}...")
    
except Exception as e:
    print(f"   ❌ PostgreSQL connection failed: {e}")
    print("\n   🔧 Quick fix:")
    print("   1. Open pgAdmin")
    print("   2. Create database: CREATE DATABASE ai_service;")
    print("   3. Run this test again")

# Test 4: Test Redis connection
print("\n📌 [4] Testing Redis Connection:")
try:
    import redis
    
    r = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    
    # Test ping
    r.ping()
    
    # Test read/write
    r.set("test:setup", "working")
    result = r.get("test:setup")
    r.delete("test:setup")
    
    print("   ✅ Redis connected successfully!")
    print(f"   🔴 Read/Write test: {result}")
    
except Exception as e:
    print(f"   ❌ Redis connection failed: {e}")
    print("\n   🔧 Quick fix:")
    print("   1. Make sure Redis is running")
    print("   2. Check Windows Services for 'Redis'")

# Test 5: Test Async SQLAlchemy (what your app actually uses)
print("\n📌 [5] Testing Async Database (SQLAlchemy):")
try:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    
    # Your exact database URL from settings
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False
    )
    
    async def test_async_db():
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar()
    
    result = asyncio.run(test_async_db())
    print(f"   ✅ Async SQLAlchemy connected!")
    print(f"   🔢 Test query result: {result}")
    
    asyncio.run(engine.dispose())
    
except Exception as e:
    print(f"   ❌ Async SQLAlchemy failed: {e}")

# Test 6: Test required directories
print("\n📌 [6] Checking Required Directories:")
try:
    # Check if upload directory exists
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    print(f"   ✅ Upload directory: {upload_dir.absolute()}")
    
    # Check if logs directory exists
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"   ✅ Logs directory: {logs_dir.absolute()}")
    
    # Check if .env exists
    env_file = Path("./.env")
    if env_file.exists():
        print(f"   ✅ .env file found")
    else:
        print(f"   ⚠️  .env file not found (create from .env.example)")
    
except Exception as e:
    print(f"   ❌ Directory check failed: {e}")

# Test 7: Check for required packages
print("\n📌 [7] Checking Required Packages:")
required_packages = [
    "fastapi", "uvicorn", "sqlalchemy", "psycopg2-binary",
    "redis", "pydantic", "pydantic-settings"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - missing")
        print(f"      Run: pip install {package}")

# Final Summary
print("\n" + "=" * 70)
print("📋 SETUP SUMMARY")
print("=" * 70)

if all([
    'conn' in locals() or 'psycopg2' in sys.modules,  # PostgreSQL check
    'r' in locals() or 'redis' in sys.modules,       # Redis check
]):
    print("\n✅ YOUR ENVIRONMENT IS READY! 🚀")
    print("\n📝 Next steps:")
    print("   1. Create database tables (if needed):")
    print("      python scripts/create_tables.py")
    print("\n   2. Start the application:")
    print("      python -m app.main")
    print("\n   3. Open in browser:")
    print("      http://localhost:8000/docs")
    print("      http://localhost:8000/redoc")
else:
    print("\n⚠️  Some checks failed. Fix the issues above and try again.")
    
print("\n" + "=" * 70)