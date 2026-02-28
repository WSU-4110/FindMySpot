import psycopg2
import getpass

password = getpass.getpass("\nEnter PostgreSQL password: ")

print("\nCreating tables...")

conn = psycopg2.connect(
    host='localhost',
    database='license_plate_db',
    user='postgres',
    password=password
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        license_plate VARCHAR(20) NOT NULL,
        make VARCHAR(50),
        model VARCHAR(50),
        color VARCHAR(30),
        year INTEGER,
        nickname VARCHAR(50),
        floor VARCHAR(10),
        spot VARCHAR(10),
        is_primary BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
print("✓ vehicles table created")

cur.close()
conn.close()
print("✅ Done!")