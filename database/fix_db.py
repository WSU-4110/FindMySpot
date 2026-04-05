import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def fix_everything():
    # We ask for the password once
    db_pw = input("Enter your PostgreSQL 'postgres' user password: ")

    try:
        # STEP 1: Connect to the default 'postgres' system database
        # This is the 'lobby' of the database server
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            host='localhost',
            password=db_pw
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # STEP 2: Create the license_plate_db
        try:
            cur.execute("CREATE DATABASE license_plate_db")
            print("✅ Database 'license_plate_db' created successfully!")
        except psycopg2.errors.DuplicateDatabase:
            print("ℹ️ Database 'license_plate_db' already exists. Moving on...")
        
        cur.close()
        conn.close()

        # STEP 3: Connect to the NEW database to create the table
        conn = psycopg2.connect(
            dbname='license_plate_db',
            user='postgres',
            host='localhost',
            password=db_pw
        )
        cur = conn.cursor()

        # This is the exact table your AuthDatabase class expects
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            username VARCHAR(100) NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        """
        cur.execute(create_table_sql)
        conn.commit()
        print("✅ Table 'users' is ready!")
        
        cur.close()
        conn.close()
        print("\n🚀 All set! You can now run your original test script.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Double-check that your PostgreSQL password is correct and the service is running.")

if __name__ == "__main__":
    fix_everything()