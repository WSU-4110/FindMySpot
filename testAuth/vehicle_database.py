"""
vehicle_database.py - Vehicle and license plate management for ParkDetroit
Sprint 2: Connect users to their vehicles

Handles:
- Adding/removing vehicles for users
- Looking up vehicle owners by license plate
- Managing primary vehicle designation
- Vehicle information (make, model, color, year)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Dict, List
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleDatabase:
    """Handle all vehicle-related database operations"""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        """
        Initialize database connection for vehicle management
        
        Uses environment variables if parameters not provided:
        - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        """
        self.connection_params = {
            'host': host or os.getenv('DB_HOST', 'localhost'),
            'port': port or int(os.getenv('DB_PORT', 5432)),
            'database': database or os.getenv('DB_NAME', 'license_plate_db'),
            'user': user or os.getenv('DB_USER', 'postgres'),
            'password': password or os.getenv('DB_PASSWORD', 'postgres')
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            logger.info("✓ VehicleDatabase connected to PostgreSQL")
        except psycopg2.Error as e:
            logger.error(f"✗ VehicleDatabase connection failed: {e}")
            raise
    
    def ensure_connection(self):
        """Ensure database connection is alive"""
        if self.conn is None or self.conn.closed:
            self.connect()
    
    def _normalize_plate(self, plate: str) -> str:
        """
        Normalize license plate format for consistent storage/lookup
        
        Args:
            plate: Raw license plate string
            
        Returns:
            Normalized plate (uppercase, spaces removed)
        """
        return plate.upper().replace(' ', '').replace('-', '')
    
    def add_vehicle(self,
                   user_id: int,
                   license_plate: str,
                   make: str = None,
                   model: str = None,
                   color: str = None,
                   year: int = None,
                   nickname: str = None,
                   floor: str = None,
                   spot: str = None,
                   is_primary: bool = False) -> Optional[int]:
        """
        Add a new vehicle for a user
        
        Args:
            user_id: Owner's user ID
            license_plate: License plate number
            make: Vehicle make (e.g., "Toyota")
            model: Vehicle model (e.g., "Camry")
            color: Vehicle color (e.g., "Blue")
            year: Vehicle year (e.g., 2020)
            nickname: User's nickname for vehicle (e.g., "My Car")
            floor: Parking floor
            spot: Parking spot
            is_primary: Whether this is user's primary vehicle
            
        Returns:
            vehicle_id if successful, None if vehicle already exists
            
        Raises:
            ValueError: If validation fails
            psycopg2.Error: If database operation fails
        """
        # Validate inputs
        if not license_plate or len(license_plate) < 2:
            raise ValueError("License plate must be at least 2 characters")
        
        if year and (year < 1900 or year > datetime.now().year + 1):
            raise ValueError(f"Invalid year: {year}")
        
        self.ensure_connection()
        
        # Normalize license plate
        normalized_plate = self._normalize_plate(license_plate)
        
        # Check if vehicle already exists for this user
        existing = self.get_vehicle_by_plate(normalized_plate, user_id)
        if existing:
            logger.warning(f"Vehicle {normalized_plate} already exists for user {user_id}")
            return None
        
        query = """
            INSERT INTO vehicles 
            (user_id, license_plate, make, model, color, year, nickname, floor, spot, is_primary, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (
                    user_id,
                    normalized_plate,
                    make,
                    model,
                    color,
                    year,
                    nickname,
                    floor,
                    spot,
                    is_primary,
                    datetime.now(),
                    datetime.now()
                ))
                vehicle_id = cur.fetchone()[0]
                self.conn.commit()
                
                logger.info(
                    f"✓ Added vehicle '{normalized_plate}' for user {user_id} "
                    f"(ID: {vehicle_id})"
                )
                return vehicle_id
                
        except psycopg2.IntegrityError as e:
            self.conn.rollback()
            if 'idx_one_primary_per_user' in str(e):
                raise ValueError("User already has a primary vehicle. Remove primary status from other vehicle first.")
            logger.error(f"✗ Failed to add vehicle: {e}")
            raise
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"✗ Failed to add vehicle: {e}")
            raise
    
    def get_user_vehicles(self, user_id: int) -> List[Dict]:
        """
        Get all vehicles for a specific user
        
        Args:
            user_id: User's database ID
            
        Returns:
            List of vehicle dictionaries
        """
        self.ensure_connection()
        
        query = """
            SELECT 
                id,
                user_id,
                license_plate,
                make,
                model,
                color,
                year,
                nickname,
                floor,
                spot,
                is_primary,
                created_at,
                updated_at
            FROM vehicles
            WHERE user_id = %s
            ORDER BY is_primary DESC, created_at DESC;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (user_id,))
                vehicles = cur.fetchall()
                
                logger.debug(f"Found {len(vehicles)} vehicle(s) for user {user_id}")
                return vehicles
                
        except psycopg2.Error as e:
            logger.error(f"✗ Failed to fetch user vehicles: {e}")
            return []
    
    def get_vehicle_by_plate(self, 
                            license_plate: str, 
                            user_id: int = None) -> Optional[Dict]:
        """
        Find vehicle by license plate number
        
        Args:
            license_plate: License plate to search for
            user_id: Optional - restrict search to specific user
            
        Returns:
            Vehicle dictionary if found, None otherwise
        """
        self.ensure_connection()
        
        normalized_plate = self._normalize_plate(license_plate)
        
        if user_id:
            query = """
                SELECT 
                    id,
                    user_id,
                    license_plate,
                    make,
                    model,
                    color,
                    year,
                    nickname,
                    floor,
                    spot,
                    is_primary,
                    created_at,
                    updated_at
                FROM vehicles
                WHERE license_plate = %s AND user_id = %s;
            """
            params = (normalized_plate, user_id)
        else:
            query = """
                SELECT 
                    id,
                    user_id,
                    license_plate,
                    make,
                    model,
                    color,
                    year,
                    nickname,
                    floor,
                    spot,
                    is_primary,
                    created_at,
                    updated_at
                FROM vehicles
                WHERE license_plate = %s;
            """
            params = (normalized_plate,)
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                vehicle = cur.fetchone()
                
                if vehicle:
                    logger.debug(f"Found vehicle: {normalized_plate}")
                else:
                    logger.debug(f"Vehicle not found: {normalized_plate}")
                
                return vehicle
                
        except psycopg2.Error as e:
            logger.error(f"✗ Failed to search vehicle: {e}")
            return None
    
    def get_vehicle_owner(self, license_plate: str) -> Optional[Dict]:
        """
        Get the owner (user) of a vehicle by license plate
        
        Args:
            license_plate: License plate to search for
            
        Returns:
            Dictionary with user info and vehicle info, or None if not found
        """
        self.ensure_connection()
        
        normalized_plate = self._normalize_plate(license_plate)
        
        query = """
            SELECT 
                u.id as user_id,
                u.email,
                u.username,
                u.role,
                v.id as vehicle_id,
                v.license_plate,
                v.make,
                v.model,
                v.color,
                v.year,
                v.nickname,
                v.is_primary
            FROM vehicles v
            JOIN users u ON v.user_id = u.id
            WHERE v.license_plate = %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (normalized_plate,))
                result = cur.fetchone()
                
                if result:
                    logger.info(f"✓ Found owner for plate {normalized_plate}: {result['email']}")
                else:
                    logger.info(f"⚠ No owner found for plate: {normalized_plate}")
                
                return result
                
        except psycopg2.Error as e:
            logger.error(f"✗ Failed to get vehicle owner: {e}")
            return None
    
    def update_vehicle(self,
                      vehicle_id: int,
                      make: str = None,
                      model: str = None,
                      color: str = None,
                      year: int = None,
                      nickname: str = None,
                      floor: str = None,
                      spot: str = None,
                      is_primary: bool = None) -> bool:
        """
        Update vehicle information
        
        Args:
            vehicle_id: Vehicle's database ID
            make: Updated make
            model: Updated model
            color: Updated color
            year: Updated year
            nickname: Updated nickname
            floor: Updated floor
            spot: Updated spot
            is_primary: Updated primary status
            
        Returns:
            True if updated successfully, False otherwise
        """
        self.ensure_connection()
        
        # Build dynamic update query based on provided parameters
        updates = []
        params = []
        
        if make is not None:
            updates.append("make = %s")
            params.append(make)
        
        if model is not None:
            updates.append("model = %s")
            params.append(model)
        
        if color is not None:
            updates.append("color = %s")
            params.append(color)
        
        if year is not None:
            updates.append("year = %s")
            params.append(year)
        
        if nickname is not None:
            updates.append("nickname = %s")
            params.append(nickname)
        
        if floor is not None:
            updates.append("floor = %s")
            params.append(floor)
        
        if spot is not None:
            updates.append("spot = %s")
            params.append(spot)
        
        if is_primary is not None:
            updates.append("is_primary = %s")
            params.append(is_primary)
        
        if not updates:
            logger.warning("No fields to update")
            return False
        
        updates.append("updated_at = %s")
        params.append(datetime.now())
        
        params.append(vehicle_id)
        
        query = f"""
            UPDATE vehicles
            SET {', '.join(updates)}
            WHERE id = %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                updated = cur.rowcount > 0
                self.conn.commit()
                
                if updated:
                    logger.info(f"✓ Updated vehicle ID: {vehicle_id}")
                else:
                    logger.warning(f"⚠ Vehicle ID {vehicle_id} not found")
                
                return updated
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"✗ Failed to update vehicle: {e}")
            return False
    
    def delete_vehicle(self, vehicle_id: int, user_id: int) -> bool:
        """
        Delete a vehicle (ensures user owns it)
        
        Args:
            vehicle_id: Vehicle's database ID
            user_id: User's ID (for authorization check)
            
        Returns:
            True if deleted successfully, False otherwise
        """
        self.ensure_connection()
        
        query = """
            DELETE FROM vehicles
            WHERE id = %s AND user_id = %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (vehicle_id, user_id))
                deleted = cur.rowcount > 0
                self.conn.commit()
                
                if deleted:
                    logger.info(f"✓ Deleted vehicle ID: {vehicle_id}")
                else:
                    logger.warning(
                        f"⚠ Vehicle ID {vehicle_id} not found or doesn't belong to user {user_id}"
                    )
                
                return deleted
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"✗ Failed to delete vehicle: {e}")
            return False
    
    def set_primary_vehicle(self, vehicle_id: int, user_id: int) -> bool:
        """
        Set a vehicle as user's primary vehicle (unsets others)
        
        Args:
            vehicle_id: Vehicle to set as primary
            user_id: User's ID
            
        Returns:
            True if successful, False otherwise
        """
        self.ensure_connection()
        
        try:
            with self.conn.cursor() as cur:
                # First, unset all primary vehicles for this user
                cur.execute("""
                    UPDATE vehicles
                    SET is_primary = false
                    WHERE user_id = %s;
                """, (user_id,))
                
                # Then set the specified vehicle as primary
                cur.execute("""
                    UPDATE vehicles
                    SET is_primary = true, updated_at = %s
                    WHERE id = %s AND user_id = %s;
                """, (datetime.now(), vehicle_id, user_id))
                
                updated = cur.rowcount > 0
                self.conn.commit()
                
                if updated:
                    logger.info(f"✓ Set vehicle {vehicle_id} as primary for user {user_id}")
                else:
                    logger.warning(f"⚠ Vehicle {vehicle_id} not found for user {user_id}")
                
                return updated
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"✗ Failed to set primary vehicle: {e}")
            return False
    
    def get_vehicle_count(self, user_id: int) -> int:
        """
        Get total number of vehicles for a user
        
        Args:
            user_id: User's database ID
            
        Returns:
            Number of vehicles
        """
        self.ensure_connection()
        
        query = "SELECT COUNT(*) FROM vehicles WHERE user_id = %s;"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (user_id,))
                count = cur.fetchone()[0]
                return count
                
        except psycopg2.Error as e:
            logger.error(f"✗ Failed to count vehicles: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("✓ VehicleDatabase connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Test/Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing VehicleDatabase")
    print("="*60)
    
    # Get password from user
    import getpass
    db_password = getpass.getpass("Enter PostgreSQL password: ")
    
    try:
        with VehicleDatabase(password=db_password) as vehicle_db:
            
            # Assume we have a user with ID 1 (create one first if needed)
            test_user_id = 1
            
            # Test 1: Add a vehicle
            print("\n1. Adding test vehicle...")
            vehicle_id = vehicle_db.add_vehicle(
                user_id=test_user_id,
                license_plate="ABC123",
                make="Toyota",
                model="Camry",
                color="Blue",
                year=2020,
                nickname="My Car",
                is_primary=True
            )
            
            if vehicle_id:
                print(f"   ✓ Vehicle added with ID: {vehicle_id}")
            else:
                print("   ⚠ Vehicle already exists (this is okay for testing)")
            
            # Test 2: Get user's vehicles
            print("\n2. Getting user's vehicles...")
            vehicles = vehicle_db.get_user_vehicles(test_user_id)
            print(f"   Found {len(vehicles)} vehicle(s):")
            for v in vehicles:
                primary = "⭐ PRIMARY" if v['is_primary'] else ""
                print(f"   - {v['license_plate']}: {v['year']} {v['make']} {v['model']} {primary}")
            
            # Test 3: Look up vehicle by plate
            print("\n3. Looking up vehicle by plate...")
            vehicle = vehicle_db.get_vehicle_by_plate("ABC123")
            if vehicle:
                print(f"   ✓ Found: {vehicle['year']} {vehicle['make']} {vehicle['model']}")
            
            # Test 4: Get vehicle owner
            print("\n4. Finding owner of plate 'ABC123'...")
            owner = vehicle_db.get_vehicle_owner("ABC123")
            if owner:
                print(f"   ✓ Owner: {owner['username']} ({owner['email']})")
                print(f"   Vehicle: {owner['year']} {owner['make']} {owner['model']}")
            
            # Test 5: Add another vehicle
            print("\n5. Adding second vehicle...")
            vehicle_id_2 = vehicle_db.add_vehicle(
                user_id=test_user_id,
                license_plate="XYZ789",
                make="Honda",
                model="Civic",
                color="Red",
                year=2019,
                nickname="Work Car"
            )
            
            if vehicle_id_2:
                print(f"   ✓ Second vehicle added with ID: {vehicle_id_2}")
            
            # Test 6: Get vehicle count
            print("\n6. Counting user's vehicles...")
            count = vehicle_db.get_vehicle_count(test_user_id)
            print(f"   User has {count} vehicle(s)")
            
            # Test 7: Update vehicle
            if vehicle_id_2:
                print("\n7. Updating vehicle color...")
                success = vehicle_db.update_vehicle(
                    vehicle_id=vehicle_id_2,
                    color="Black"
                )
                if success:
                    print("   ✓ Vehicle updated")
            
            # Test 8: Test plate normalization
            print("\n8. Testing plate normalization...")
            # These should all match the same vehicle
            test_plates = ["ABC123", "abc123", "ABC 123", "ABC-123"]
            for plate in test_plates:
                result = vehicle_db.get_vehicle_by_plate(plate)
                if result:
                    print(f"   ✓ '{plate}' matched → {result['license_plate']}")
        
        print("\n" + "="*60)
        print("✓ All vehicle tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        logger.exception("Test suite failed")