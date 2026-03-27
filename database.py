import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PlateDatabase:
    """Handle all database operations for license plates"""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        """
        Initialize database connection from environment variables or parameters
        
        Priority: Parameters > Environment Variables > Defaults
        
        Environment variables:
            DB_HOST - Database host (default: localhost)
            DB_PORT - Database port (default: 5432)
            DB_NAME - Database name (default: license_plate_db)
            DB_USER - Database user (default: postgres)
            DB_PASSWORD - Database password (default: postgres)
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
            logger.info(f"Connected to PostgreSQL database: {self.connection_params['database']}")
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def ensure_connection(self, max_retries: int = 3):
        """
        Ensure database connection is alive with retry logic
        
        Args:
            max_retries: Maximum number of connection attempts
        """
        for attempt in range(max_retries):
            try:
                if self.conn is None or self.conn.closed:
                    self.connect()
                
                # Test connection with a simple query
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return  # Connection successful
                
            except psycopg2.Error as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    raise
    
    def _validate_plate_number(self, plate_number: str):
        """Validate plate number format"""
        if not plate_number:
            raise ValueError("plate_number cannot be empty")
        
        if not isinstance(plate_number, str):
            raise TypeError(f"plate_number must be string, got {type(plate_number)}")
        
        if len(plate_number) < 4 or len(plate_number) > 20:
            raise ValueError(f"plate_number '{plate_number}' must be 4-20 characters")
        
        # Check for only alphanumeric characters
        if not plate_number.replace(' ', '').isalnum():
            raise ValueError(f"plate_number '{plate_number}' contains invalid characters")
    
    def _validate_confidence(self, confidence: Optional[float]):
        """Validate confidence score"""
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise TypeError(f"confidence must be numeric, got {type(confidence)}")
            
            if confidence < 0 or confidence > 1:
                raise ValueError(f"confidence {confidence} must be between 0 and 1")
    
    def _validate_camera_id(self, camera_id):
        """Validate camera ID"""
        if camera_id is None:
            raise ValueError("camera_id cannot be None")
        
        # If your DB uses Integers, ensure it's an int
        if not isinstance(camera_id, int):
            try:
                # Try to see if it's a string that looks like a number
                int(camera_id)
            except (ValueError, TypeError):
                raise TypeError(f"camera_id must be an integer, got {type(camera_id)}")
    
    def save_plate(self, 
                   plate_number: str, 
                   confidence: float = None,
                   camera_id: str = 'default',
                   detected_at: datetime = None,
                   dedup_window_seconds: int = 30) -> Optional[int]:
        """
        Save a detected plate to the database with deduplication
        
        Args:
            plate_number: The license plate number detected
            confidence: OCR confidence score (0-1)
            camera_id: Identifier for the camera
            detected_at: Timestamp of detection (defaults to now)
            dedup_window_seconds: Don't save if same plate detected within this window
        
        Returns:
            id: The database ID of the inserted record, or None if duplicate skipped
        
        Raises:
            ValueError: If input validation fails
            psycopg2.Error: If database operation fails
        """

        # 1. Validate inputs
        self._validate_plate_number(plate_number)
        self._validate_confidence(confidence)
        
        self.ensure_connection()
        
        if detected_at is None:
            detected_at = datetime.now()
        
        # 2. Setup Deduplication Query
        dedup_query = """
            SELECT id, detected_at 
            FROM detected_plates
            WHERE plate_number = %s 
            AND camera_id = %s
            AND detected_at > %s
            ORDER BY detected_at DESC
            LIMIT 1;
        """
        
        cutoff_time = detected_at - timedelta(seconds=dedup_window_seconds)

        try:
            with self.conn.cursor() as cur:
                # Check for recent duplicate
                cur.execute(dedup_query, (plate_number, camera_id, cutoff_time))
                recent_detection = cur.fetchone()
                
                if recent_detection:
                    recent_id, recent_time = recent_detection
                    seconds_ago = (detected_at - recent_time).total_seconds()
                    logger.info(f"Skipping duplicate plate '{plate_number}' ({seconds_ago:.1f}s ago)")
                    return None
                
                # 3. Save new record
                insert_query = """
                    INSERT INTO detected_plates (plate_number, detected_at, confidence, camera_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """
                cur.execute(insert_query, (plate_number, detected_at, confidence, camera_id))
                plate_id = cur.fetchone()[0]
                self.conn.commit()
                
                logger.info(f"Saved plate '{plate_number}' (ID: {plate_id})")
                return plate_id
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to save plate '{plate_number}': {e}")
            raise
    
    def get_recent_plates(self, limit: int = 50) -> List[Dict]:
        """Get most recently detected plates"""
        self.ensure_connection()
        
        query = """
            SELECT id, plate_number, detected_at, confidence, camera_id
            FROM detected_plates
            ORDER BY detected_at DESC
            LIMIT %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                results = cur.fetchall()
                logger.debug(f"Fetched {len(results)} recent plates")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to fetch recent plates: {e}")
            return []
    
    def get_plates_today(self) -> List[Dict]:
        """Get all plates detected today"""
        self.ensure_connection()
        
        query = """
            SELECT id, plate_number, detected_at, confidence, camera_id
            FROM detected_plates
            WHERE DATE(detected_at) = CURRENT_DATE
            ORDER BY detected_at DESC;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                logger.debug(f"Fetched {len(results)} plates from today")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to fetch today's plates: {e}")
            return []
    
    def get_plates_by_timerange(self, 
                                start_time: datetime, 
                                end_time: datetime) -> List[Dict]:
        """Get plates detected within a time range"""
        self.ensure_connection()
        
        query = """
            SELECT id, plate_number, detected_at, confidence, camera_id
            FROM detected_plates
            WHERE detected_at BETWEEN %s AND %s
            ORDER BY detected_at DESC;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (start_time, end_time))
                results = cur.fetchall()
                logger.debug(f"Fetched {len(results)} plates in time range")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to fetch plates by time range: {e}")
            return []
    
    def search_plate(self, plate_number: str) -> List[Dict]:
        """
        Search for specific plate number (supports partial match)
        
        Args:
            plate_number: Full or partial plate number to search
            
        Returns:
            List of matching plate detections
        """
        self.ensure_connection()
        
        query = """
            SELECT id, plate_number, detected_at, confidence, camera_id
            FROM detected_plates
            WHERE plate_number ILIKE %s
            ORDER BY detected_at DESC;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (f'%{plate_number}%',))
                results = cur.fetchall()
                logger.debug(f"Search for '{plate_number}' found {len(results)} results")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to search plates: {e}")
            return []
    
    def get_plate_stats(self) -> Dict:
        """Get statistics about detected plates"""
        self.ensure_connection()
        
        query = """
            SELECT 
                COUNT(*) as total_detections,
                COUNT(DISTINCT plate_number) as unique_plates,
                MAX(detected_at) as last_detection,
                MIN(detected_at) as first_detection,
                AVG(confidence) as avg_confidence
            FROM detected_plates;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                stats = cur.fetchone()
                logger.debug(f"Fetched database stats: {stats['total_detections']} detections")
                return stats
        except psycopg2.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def get_most_seen_plates(self, limit: int = 10) -> List[Dict]:
        """Get plates seen most frequently"""
        self.ensure_connection()
        
        query = """
            SELECT 
                plate_number,
                COUNT(*) as times_seen,
                MAX(detected_at) as last_seen,
                MIN(detected_at) as first_seen,
                AVG(confidence) as avg_confidence
            FROM detected_plates
            GROUP BY plate_number
            ORDER BY times_seen DESC
            LIMIT %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                results = cur.fetchall()
                logger.debug(f"Fetched top {len(results)} most seen plates")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to get most seen plates: {e}")
            return []
    
    def get_plate_detection_count(self, 
                                  plate_number: str,
                                  start_time: datetime = None,
                                  end_time: datetime = None) -> int:
        """
        Count how many times a specific plate was detected in time range
        
        Args:
            plate_number: The plate to count
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            Number of detections
        """
        self.ensure_connection()
        
        if start_time and end_time:
            query = """
                SELECT COUNT(*) as count
                FROM detected_plates
                WHERE plate_number = %s
                AND detected_at BETWEEN %s AND %s;
            """
            params = (plate_number, start_time, end_time)
        else:
            query = """
                SELECT COUNT(*) as count
                FROM detected_plates
                WHERE plate_number = %s;
            """
            params = (plate_number,)
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                count = result['count'] if result else 0
                logger.debug(f"Plate '{plate_number}' detected {count} times")
                return count
        except psycopg2.Error as e:
            logger.error(f"Failed to get detection count: {e}")
            return 0
    
    def get_hourly_stats(self, date: datetime = None) -> List[Dict]:
        """
        Get detection counts grouped by hour for a specific date
        
        Args:
            date: Date to analyze (defaults to today)
            
        Returns:
            List of {hour, detections} for each hour
        """
        self.ensure_connection()
        
        if date is None:
            date = datetime.now().date()
        
        query = """
            SELECT 
                EXTRACT(HOUR FROM detected_at)::INTEGER as hour,
                COUNT(*) as detections
            FROM detected_plates
            WHERE DATE(detected_at) = %s
            GROUP BY hour
            ORDER BY hour;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (date,))
                results = cur.fetchall()
                logger.debug(f"Fetched hourly stats for {date}: {len(results)} hours")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to get hourly stats: {e}")
            return []
    
    def get_plates_by_camera(self, camera_id: str, limit: int = 50) -> List[Dict]:
        """
        Get plates detected by a specific camera
        
        Args:
            camera_id: The camera identifier
            limit: Maximum number of results
            
        Returns:
            List of plate detections from this camera
        """
        self.ensure_connection()
        
        query = """
            SELECT id, plate_number, detected_at, confidence, camera_id
            FROM detected_plates
            WHERE camera_id = %s
            ORDER BY detected_at DESC
            LIMIT %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (camera_id, limit))
                results = cur.fetchall()
                logger.debug(f"Fetched {len(results)} plates from camera '{camera_id}'")
                return results
        except psycopg2.Error as e:
            logger.error(f"Failed to get plates by camera: {e}")
            return []
    
    def update_plate_confidence(self, plate_id: int, new_confidence: float) -> bool:
        """
        Update confidence score for an existing detection
        
        Args:
            plate_id: Database ID of the detection
            new_confidence: New confidence value (0-1)
            
        Returns:
            True if updated successfully, False otherwise
        """
        self._validate_confidence(new_confidence)
        self.ensure_connection()
        
        query = """
            UPDATE detected_plates
            SET confidence = %s
            WHERE id = %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (new_confidence, plate_id))
                updated = cur.rowcount > 0
                self.conn.commit()
                
                if updated:
                    logger.info(f"Updated confidence for plate ID {plate_id} to {new_confidence}")
                else:
                    logger.warning(f"No plate found with ID {plate_id}")
                
                return updated
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to update confidence: {e}")
            return False
    
    def delete_old_plates(self, days: int = 30) -> int:
        """
        Delete plates older than specified days
        
        Args:
            days: Delete plates older than this many days
            
        Returns:
            Number of plates deleted
        """
        self.ensure_connection()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        query = """
            DELETE FROM detected_plates
            WHERE detected_at < %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (cutoff_date,))
                deleted_count = cur.rowcount
                self.conn.commit()
                logger.info(f"Deleted {deleted_count} plates older than {days} days")
                return deleted_count
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to delete old plates: {e}")
            return 0
    
    def get_connection_info(self) -> Dict:
        """
        Get information about the current database connection
        
        Returns:
            Dictionary with connection details (password masked)
        """
        return {
            'host': self.connection_params['host'],
            'port': self.connection_params['port'],
            'database': self.connection_params['database'],
            'user': self.connection_params['user'],
            'connected': self.conn is not None and not self.conn.closed
        }
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing PlateDatabase - Sprint 2 Version")
    print("="*60)
    
    try:
        # Test with context manager
        with PlateDatabase() as db:

            print("\n0. Ensuring test camera exists...")
            with db.conn.cursor() as cur:
                # This adds Camera ID 1 if it doesn't already exist
                cur.execute("""
                    INSERT INTO cameras (id, name, location)
                    VALUES (1, 'Main Entrance', 'Front Gate')
                    ON CONFLICT (id) DO NOTHING;
                """)
                db.conn.commit()

            print("\n1. Testing connection info:")
            info = db.get_connection_info()
            print(f"   Connected to: {info['database']} at {info['host']}:{info['port']}")
            print(f"   Connection status: {'✓ Active' if info['connected'] else '✗ Closed'}")
            
            print("\n2. Testing plate save with validation:")
            # This should work
            plate_id = db.save_plate("ABC123", confidence=0.95, camera_id=1)
            if plate_id:
                print(f"   ✓ Saved plate (ID: {plate_id})")
            
            print("\n3. Testing deduplication (saving same plate immediately):")
            # This should be skipped (duplicate)
            duplicate_id = db.save_plate("ABC123", confidence=0.92, camera_id=1)
            if duplicate_id is None:
                print("   ✓ Duplicate correctly skipped")
            
            print("\n4. Testing validation (should fail):")
            try:
                db.save_plate("", confidence=0.5)  # Empty plate
                print("   ✗ Validation failed - empty plate was accepted!")
            except ValueError as e:
                print(f"   ✓ Validation working: {e}")
            
            print("\n5. Getting recent plates:")
            recent = db.get_recent_plates(limit=5)
            print(f"   Found {len(recent)} recent detections")
            for plate in recent[:3]:
                print(f"   - {plate['plate_number']} at {plate['detected_at']}")
            
            print("\n6. Getting statistics:")
            stats = db.get_plate_stats()
            print(f"   Total detections: {stats.get('total_detections', 0)}")
            print(f"   Unique plates: {stats.get('unique_plates', 0)}")
            if stats.get('avg_confidence'):
                print(f"   Average confidence: {stats['avg_confidence']:.2%}")
            
            print("\n7. Testing new methods:")
            count = db.get_plate_detection_count("ABC123")
            print(f"   Plate 'ABC123' detected {count} time(s)")
            
            hourly = db.get_hourly_stats()
            if hourly:
                print(f"   Hourly stats: {len(hourly)} hours with activity")
            
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        logger.exception("Test suite failed")








