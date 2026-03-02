# check_my_tables.py
from sqlalchemy import create_engine, inspect, text
from app.core.config import settings

# Use sync connection
sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(sync_url)

print("=" * 60)
print("🔍 DATABASE TABLE CHECK")
print("=" * 60)

with engine.connect() as conn:
    # Check if datasets table exists
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'datasets'
    """))
    
    exists = result.fetchone()
    
    if exists:
        print("✅ 'datasets' table EXISTS!")
        
        # Show table structure
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'datasets'
            ORDER BY ordinal_position
        """))
        
        print("\n📋 Table Structure:")
        print("-" * 50)
        for row in result:
            print(f"  • {row[0]}: {row[1]} (nullable: {row[2]})")
        
        # Count rows
        result = conn.execute(text("SELECT COUNT(*) FROM datasets"))
        count = result.scalar()
        print(f"\n📊 Row count: {count}")
        
    else:
        print("❌ 'datasets' table NOT found")
        
        # List all tables
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        tables = [row[0] for row in result]
        print(f"\nExisting tables: {tables}")

print("\n✅ Check complete!")