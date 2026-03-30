import pytest
from datetime import datetime
from vehicle_database import VehicleDatabase

@pytest.fixture
def db():
    """Fixture to initialize and close the database connection"""
    # Uses environment variables for connection by default
    database = VehicleDatabase()
    yield database
    database.close()

@pytest.fixture
def test_user(db):
    """
    Helper fixture to ensure a test user exists in the users table.
    Assumes your database schema has a users table.
    """
    with db.conn.cursor() as cur:
        # Create a dummy user for foreign key constraints
        cur.execute("""
            INSERT INTO users (email, username, password_hash, role)
            VALUES ('test@example.com', 'testuser', 'hash', 'user')
            ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
            RETURNING id;
        """)
        user_id = cur.fetchone()[0]
        db.conn.commit()
    return user_id

class TestVehicleDatabase:
    
    def test_add_vehicle_success(self, db, test_user):
        """Test adding a valid vehicle to a user"""
        vehicle_id = db.add_vehicle(
            user_id=test_user,
            license_plate="DET-2026",
            make="Ford",
            model="F-150",
            year=2024,
            color="Silver"
        )
        assert vehicle_id is not None
        
        # Verify it was stored
        vehicle = db.get_vehicle_by_plate("DET2026")
        assert vehicle['make'] == "Ford"
        assert vehicle['user_id'] == test_user

    def test_plate_normalization(self, db, test_user):
        """Verify that plates are normalized (caps, no spaces/dashes)"""
        db.add_vehicle(test_user, "abc 123")
        
        # Lookup with different formats
        assert db.get_vehicle_by_plate("ABC123") is not None
        assert db.get_vehicle_by_plate("abc-123") is not None
        assert db.get_vehicle_by_plate("ABC 123") is not None

    def test_duplicate_vehicle_for_user(self, db, test_user):
        """Test that adding the same plate twice returns None"""
        db.add_vehicle(test_user, "UNIQUE1")
        result = db.add_vehicle(test_user, "unique 1")
        assert result is None

    def test_invalid_year_raises_error(self, db, test_user):
        """Test validation for impossible vehicle years"""
        with pytest.raises(ValueError, match="Invalid year"):
            db.add_vehicle(test_user, "PLATE1", year=1850)
            
        with pytest.raises(ValueError, match="Invalid year"):
            db.add_vehicle(test_user, "PLATE1", year=datetime.now().year + 2)

    def test_get_user_vehicles(self, db, test_user):
        """Test retrieving all vehicles owned by a specific user"""
        db.add_vehicle(test_user, "CAR1", is_primary=False)
        db.add_vehicle(test_user, "CAR2", is_primary=True)
        
        vehicles = db.get_user_vehicles(test_user)
        assert len(vehicles) >= 2
        # Primary should be first due to ORDER BY is_primary DESC
        assert vehicles[0]['license_plate'] == "CAR2"

    def test_get_vehicle_owner(self, db, test_user):
        """Test the JOIN query to find user info via license plate"""
        db.add_vehicle(test_user, "OWNER123")
        owner_info = db.get_vehicle_owner("OWNER123")
        
        assert owner_info is not None
        assert owner_info['email'] == 'test@example.com'
        assert owner_info['license_plate'] == 'OWNER123'

    def test_update_vehicle(self, db, test_user):
        """Test updating vehicle attributes"""
        v_id = db.add_vehicle(test_user, "UPDATE1", color="Red")
        success = db.update_vehicle(v_id, color="Blue", nickname="Blueberry")
        
        assert success is True
        updated = db.get_vehicle_by_plate("UPDATE1")
        assert updated['color'] == "Blue"
        assert updated['nickname'] == "Blueberry"

    def test_set_primary_vehicle(self, db, test_user):
        """Test the logic that unsets other primary vehicles"""
        v1 = db.add_vehicle(test_user, "FIRST", is_primary=True)
        v2 = db.add_vehicle(test_user, "SECOND", is_primary=False)
        
        # Switch primary to v2
        db.set_primary_vehicle(v2, test_user)
        
        vehicles = db.get_user_vehicles(test_user)
        v1_data = next(v for v in vehicles if v['id'] == v1)
        v2_data = next(v for v in vehicles if v['id'] == v2)
        
        assert v2_data['is_primary'] is True
        assert v1_data['is_primary'] is False

    def test_delete_vehicle(self, db, test_user):
        """Test vehicle deletion and ownership authorization"""
        v_id = db.add_vehicle(test_user, "GONE")
        
        # Try deleting with wrong user_id (e.g., 999)
        assert db.delete_vehicle(v_id, 999) is False
        
        # Delete with correct user_id
        assert db.delete_vehicle(v_id, test_user) is True
        assert db.get_vehicle_by_plate("GONE") is None