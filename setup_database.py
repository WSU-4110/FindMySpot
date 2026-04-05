<<<<<<< HEAD
import argparse
import logging
import os
import sys

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


DEFAULT_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "license_plate_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

DDL_STATEMENTS = [
    # ── users ────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS users (
        id            SERIAL PRIMARY KEY,
        email         VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        username      VARCHAR(50)  UNIQUE NOT NULL,
        role          VARCHAR(20)  DEFAULT 'user',
        created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
        last_login    TIMESTAMP
    )
    """,

    # ── cameras ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS cameras (
        id          SERIAL PRIMARY KEY,
        name        VARCHAR(100) NOT NULL,
        location    VARCHAR(255),
        camera_type VARCHAR(50)  DEFAULT 'webcam',
        is_active   BOOLEAN      DEFAULT true,
        created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── user_camera_access ───────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS user_camera_access (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER REFERENCES users(id)   ON DELETE CASCADE,
        camera_id  INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
        granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, camera_id)
    )
    """,

    # ── vehicles ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS vehicles (
        id            SERIAL PRIMARY KEY,
        user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
        license_plate VARCHAR(20) NOT NULL,
        make          VARCHAR(50),
        model         VARCHAR(50),
        color         VARCHAR(30),
        is_primary    BOOLEAN DEFAULT false
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_vehicles_user  ON vehicles(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(license_plate)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_one_primary_per_user
        ON vehicles(user_id) WHERE is_primary = true
    """,

    # ── detected_plates ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS detected_plates (
        id           SERIAL PRIMARY KEY,
        plate_number VARCHAR(20) NOT NULL,
        camera_id    INTEGER REFERENCES cameras(id)  ON DELETE SET NULL,
        vehicle_id   INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
        detected_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        confidence   FLOAT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plate_camera      ON detected_plates(camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_plate_number      ON detected_plates(plate_number)",
    "CREATE INDEX IF NOT EXISTS idx_plate_vehicle     ON detected_plates(vehicle_id)",
    "CREATE INDEX IF NOT EXISTS idx_plate_detected_at ON detected_plates(detected_at)",



=======
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
>>>>>>> bef3ead3623c9edc3503dd54c89a31fbe9e9b6b8
