"""
migrate_add_parking_fields.py - Add floor and spot fields to vehicles table
Run this to add parking location fields to existing vehicles table
"""

import psycopg2
import getpass
import sys

def migrate_database(password):
    """Add floor and spot columns to vehicles table"""
    
    print("\n" + "="*60)
    print("Database Migration: Add Parking Fields")
    print("="*60)
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="license_plate_db",
            user="postgres",
            password=password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("\n1. Adding 'floor' column to vehicles table...")
        cursor.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='vehicles' AND column_name='floor'
                ) THEN
                    ALTER TABLE vehicles ADD COLUMN floor VARCHAR(10);
                    RAISE NOTICE 'Column floor added';
                ELSE
                    RAISE NOTICE 'Column floor already exists';
                END IF;
            END $$;
        """)
        print("   ✓ Floor column ready")
        
        print("\n2. Adding 'spot' column to vehicles table...")
        cursor.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='vehicles' AND column_name='spot'
                ) THEN
                    ALTER TABLE vehicles ADD COLUMN spot VARCHAR(10);
                    RAISE NOTICE 'Column spot added';
                ELSE
                    RAISE NOTICE 'Column spot already exists';
                END IF;
            END $$;
        """)
        print("   ✓ Spot column ready")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*60)
        print("✅ Migration complete!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        return False

if __name__ == "__main__":
    password = getpass.getpass("\nEnter PostgreSQL password: ")
    success = migrate_database(password)
    sys.exit(0 if success else 1)