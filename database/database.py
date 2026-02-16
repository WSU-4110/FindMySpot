"""
database.py - PostgreSQL database operations for license plate detection
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os


class PlateDatabase:
    """Handle all database operations for license plates"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "license_plate_db",
                 user: str = "postgres",
                 password: str = "postgres"):
        """Initialize database connection"""
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            print(f"✓ Connected to PostgreSQL database: {self.connection_params['database']}")
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def ensure_connection(self):
        """Ensure database connection is alive"""
        try:
            if self.conn is None or self.conn.closed:
                self.connect()
        except:
            self.connect()
    
    def save_plate(self, 
                   plate_number: str, 
                   confidence: float = None,
                   camera_id: str = 'default',
                   detected_at: datetime = None) -> int:
        """
        Save a detected plate to the database
        
        Args:
            plate_number: The license plate number detected
            confidence: OCR confidence score (0-1)
            camera_id: Identifier for the camera
            detected_at: Timestamp of detection (defaults to now)
        
        Returns:
            id: The database ID of the inserted record
        """
        self.ensure_connection()
        
        if detected_at is None:
            detected_at = datetime.now()
        
        query = """
            INSERT INTO detected_plates (plate_number, detected_at, confidence, camera_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (plate_number, detected_at, confidence, camera_id))
                plate_id = cur.fetchone()[0]
                self.conn.commit()
                print(f"✓ Saved plate '{plate_number}' to database (ID: {plate_id})")
                return plate_id
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Failed to save plate: {e}")
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
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Failed to fetch recent plates: {e}")
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
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Failed to fetch today's plates: {e}")
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
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Failed to fetch plates by time range: {e}")
            return []
    
    def search_plate(self, plate_number: str) -> List[Dict]:
        """Search for specific plate number (supports partial match)"""
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
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Failed to search plates: {e}")
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
                return cur.fetchone()
        except psycopg2.Error as e:
            print(f"✗ Failed to get stats: {e}")
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
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Failed to get most seen plates: {e}")
            return []
    
    def delete_old_plates(self, days: int = 30) -> int:
        """Delete plates older than specified days"""
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
                print(f"✓ Deleted {deleted_count} plates older than {days} days")
                return deleted_count
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Failed to delete old plates: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("✓ Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Example usage
if __name__ == "__main__":
    # Test database operations
    with PlateDatabase() as db:
        # Save a test plate
        plate_id = db.save_plate("ABC123", confidence=0.95)
        
        # Get recent plates
        recent = db.get_recent_plates(limit=10)
        print(f"\nRecent plates: {len(recent)}")
        for plate in recent:
            print(f"  {plate['plate_number']} - {plate['detected_at']}")
        
        # Get stats
        stats = db.get_plate_stats()
        print(f"\nDatabase stats:")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Unique plates: {stats['unique_plates']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
