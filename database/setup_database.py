"""
setup_database.py - Setup PostgreSQL database without using psql command
Run this script to create the database and tables
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    """Create database and tables"""
    
    # Get password once
    password = input("Enter PostgreSQL password for user 'postgres': ")
    
    # First, connect to default postgres database to create our database
    print("\nStep 1: Creating database...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",  # Connect to default database first
            user="postgres",
            password=password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='license_plate_db'")
        exists = cursor.fetchone()
        
        if exists:
            print("  ✓ Database 'license_plate_db' already exists")
        else:
            cursor.execute("CREATE DATABASE license_plate_db")
            print("  ✓ Created database 'license_plate_db'")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"  ✗ Error connecting to PostgreSQL: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is installed")
        print("2. Check if PostgreSQL service is running (search 'Services' in Windows)")
        print("3. Verify the password is correct")
        return False, None
    
    # Now connect to our new database and create tables
    print("\nStep 2: Creating tables...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="license_plate_db",
            user="postgres",
            password=password
        )
        cursor = conn.cursor()
        
        # Create tables
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS detected_plates (
            id SERIAL PRIMARY KEY,
            plate_number VARCHAR(20) NOT NULL,
            detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            confidence FLOAT,
            camera_id VARCHAR(50) DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_plate_number ON detected_plates(plate_number);
        CREATE INDEX IF NOT EXISTS idx_detected_at ON detected_plates(detected_at);
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        print("  ✓ Created 'detected_plates' table")
        print("  ✓ Created indexes")
        
        # Insert a test record
        cursor.execute("""
            INSERT INTO detected_plates (plate_number, confidence, camera_id)
            VALUES (%s, %s, %s)
            RETURNING id
        """, ("TEST123", 1.0, "setup_test"))
        test_id = cursor.fetchone()[0]
        conn.commit()
        print(f"  ✓ Inserted test record (ID: {test_id})")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM detected_plates")
        count = cursor.fetchone()[0]
        print(f"  ✓ Database has {count} record(s)")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Database setup complete!")
        print("\nNext steps:")
        print("1. Update the password in app_simple.py (around line 20)")
        print("2. Run: python app_simple.py")
        return True, password
        
    except psycopg2.Error as e:
        print(f"  ✗ Error creating tables: {e}")
        return False, password

if __name__ == "__main__":
    print("="*60)
    print("License Plate Detection - Database Setup")
    print("="*60)
    print()
    
    success, password = setup_database()
    
    if success:
        print("\n" + "="*60)
        print("Setup successful! Database is ready to use.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Setup failed. Please check the errors above.")
        print("="*60)
