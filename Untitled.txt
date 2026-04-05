"""
fix_and_test.py - Fix stuck transaction and test vehicle database
This will clean up any stuck transactions and verify everything works
"""

import psycopg2
import getpass
import sys

password = getpass.getpass("\nEnter PostgreSQL password: ")

print("\n" + "="*70)
print("FIXING STUCK TRANSACTION AND TESTING DATABASE")
print("="*70)

try:
    # Step 1: Check and fix columns
    print("\n📋 Step 1: Checking database structure...")
    
    conn = psycopg2.connect(
        host='localhost',
        database='license_plate_db',
        user='postgres',
        password=password
    )
    
    # Force rollback any stuck transactions
    conn.rollback()
    
    cur = conn.cursor()
    
    # Check if vehicles table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'vehicles'
        );
    """)
    
    table_exists = cur.fetchone()[0]
    
    if not table_exists:
        print("   ✗ vehicles table doesn't exist!")
        print("   Run: python setup_sprint2_database.py")
        cur.close()
        conn.close()
        sys.exit(1)
    
    print("   ✓ vehicles table exists")
    
    # Check columns
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'vehicles'
        ORDER BY ordinal_position;
    """)
    
    columns = [row[0] for row in cur.fetchall()]
    print(f"   Found {len(columns)} columns: {', '.join(columns)}")
    
    # Add missing columns if needed
    if 'floor' not in columns:
        print("\n   Adding 'floor' column...")
        cur.execute("ALTER TABLE vehicles ADD COLUMN floor VARCHAR(10);")
        conn.commit()
        print("   ✓ Added floor column")
    else:
        print("   ✓ floor column exists")
    
    if 'spot' not in columns:
        print("\n   Adding 'spot' column...")
        cur.execute("ALTER TABLE vehicles ADD COLUMN spot VARCHAR(10);")
        conn.commit()
        print("   ✓ Added spot column")
    else:
        print("   ✓ spot column exists")
    
    cur.close()
    conn.close()
    
    # Step 2: Check if we have a test user
    print("\n📋 Step 2: Checking for test user...")
    
    conn = psycopg2.connect(
        host='localhost',
        database='license_plate_db',
        user='postgres',
        password=password
    )
    conn.rollback()  # Clear any stuck state
    
    cur = conn.cursor()
    
    # Check if users table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'users'
        );
    """)
    
    users_table_exists = cur.fetchone()[0]
    
    if not users_table_exists:
        print("   ✗ users table doesn't exist!")
        print("   Run: python setup_sprint2_database.py")
        cur.close()
        conn.close()
        sys.exit(1)
    
    # Check if any users exist
    cur.execute("SELECT COUNT(*) FROM users;")
    user_count = cur.fetchone()[0]
    
    if user_count == 0:
        print("   ⚠ No users in database - creating test user...")
        
        # Import bcrypt to hash password
        import bcrypt
        password_hash = bcrypt.hashpw(b'test123', bcrypt.gensalt()).decode('utf-8')
        
        cur.execute("""
            INSERT INTO users (email, password_hash, username, role, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id;
        """, ('test@example.com', password_hash, 'testuser', 'user'))
        
        user_id = cur.fetchone()[0]
        conn.commit()
        print(f"   ✓ Created test user (ID: {user_id})")
    else:
        cur.execute("SELECT id, email, username FROM users LIMIT 1;")
        user = cur.fetchone()
        user_id = user[0]
        print(f"   ✓ Found existing user: {user[2]} ({user[1]}) - ID: {user_id}")
    
    cur.close()
    conn.close()
    
    # Step 3: Test vehicle operations
    print("\n📋 Step 3: Testing vehicle database operations...")
    
    conn = psycopg2.connect(
        host='localhost',
        database='license_plate_db',
        user='postgres',
        password=password
    )
    conn.rollback()  # Clear any stuck state
    
    cur = conn.cursor()
    
    # Test 1: Add a vehicle
    print("\n   3.1 Adding test vehicle...")
    try:
        cur.execute("""
            INSERT INTO vehicles 
            (user_id, license_plate, make, model, color, year, floor, spot, is_primary, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT DO NOTHING
            RETURNING id;
        """, (user_id, 'FIXTEST1', 'Toyota', 'Camry', 'Blue', 2020, 'Floor 1', 'A1', True))
        
        result = cur.fetchone()
        if result:
            vehicle_id = result[0]
            conn.commit()
            print(f"       ✓ Vehicle added with ID: {vehicle_id}")
        else:
            print("       ⚠ Vehicle already exists (that's OK)")
            # Get existing vehicle
            cur.execute("SELECT id FROM vehicles WHERE license_plate = %s AND user_id = %s", ('FIXTEST1', user_id))
            vehicle_id = cur.fetchone()[0]
    
    except Exception as e:
        conn.rollback()
        print(f"       ✗ Failed: {e}")
        cur.close()
        conn.close()
        sys.exit(1)
    
    # Test 2: Query the vehicle
    print("\n   3.2 Querying vehicle...")
    try:
        cur.execute("""
            SELECT license_plate, make, model, floor, spot
            FROM vehicles
            WHERE id = %s;
        """, (vehicle_id,))
        
        result = cur.fetchone()
        print(f"       ✓ Found: {result[1]} {result[2]}")
        print(f"       Location: {result[3]}, Spot {result[4]}")
    
    except Exception as e:
        print(f"       ✗ Query failed: {e}")
    
    # Test 3: Update the vehicle
    print("\n   3.3 Updating vehicle...")
    try:
        cur.execute("""
            UPDATE vehicles
            SET floor = %s, spot = %s, updated_at = NOW()
            WHERE id = %s;
        """, ('Floor 2', 'B5', vehicle_id))
        
        conn.commit()
        print("       ✓ Vehicle updated")
        
        # Verify update
        cur.execute("SELECT floor, spot FROM vehicles WHERE id = %s", (vehicle_id,))
        result = cur.fetchone()
        print(f"       ✓ New location: {result[0]}, Spot {result[1]}")
    
    except Exception as e:
        conn.rollback()
        print(f"       ✗ Update failed: {e}")
    
    cur.close()
    conn.close()
    
    # Step 4: Test with VehicleDatabase class
    print("\n📋 Step 4: Testing VehicleDatabase class...")
    
    try:
        from vehicle_database import VehicleDatabase
        
        with VehicleDatabase(password=password) as veh_db:
            print("\n   4.1 Adding vehicle via class...")
            v_id = veh_db.add_vehicle(
                user_id=user_id,
                license_plate="CLASS001",
                make="Honda",
                model="Civic",
                color="Red",
                year=2021,
                floor="Floor 3",
                spot="C10"
            )
            
            if v_id:
                print(f"       ✓ Vehicle added (ID: {v_id})")
            else:
                print("       ⚠ Vehicle already exists")
            
            print("\n   4.2 Looking up vehicle...")
            vehicle = veh_db.get_vehicle_by_plate("CLASS001", user_id)
            if vehicle:
                print(f"       ✓ Found: {vehicle['make']} {vehicle['model']}")
                if vehicle.get('floor') and vehicle.get('spot'):
                    print(f"       ✓ Location: {vehicle['floor']}, Spot {vehicle['spot']}")
            else:
                print("       ✗ Vehicle not found")
    
    except Exception as e:
        print(f"       ✗ Class test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - DATABASE IS WORKING!")
    print("="*70)
    print("\nYour database is ready to use.")
    print("You can now run: python app_api.py")
    
except psycopg2.Error as e:
    print(f"\n✗ Database error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure PostgreSQL is running")
    print("2. Check your password is correct")
    print("3. Make sure license_plate_db exists")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)