import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import bcrypt
import jwt
import os
import logging
from typing import Optional, Dict
from exceptions import DatabaseUnavailableError

logger = logging.getLogger(__name__)

class AuthDatabase:
    def __init__(self, **params):
        self.params = params
        self.conn = None
        self.SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-key-123')

    def connect(self):
        try:
            # Added a timeout so it doesn't hang if Postgres is down
            self.conn = psycopg2.connect(connect_timeout=3, **self.params)
        except psycopg2.OperationalError:
            self.conn = None

    def ensure_connection(self):
        if not self.conn or self.conn.closed:
            self.connect()
            if not self.conn:
                raise DatabaseUnavailableError()

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def _verify_password_hash(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, email, password, username, role='user'):
        self.ensure_connection()
        password_hash = self._hash_password(password)
        
        query = """
            INSERT INTO users (email, password_hash, username, role, created_at)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (email.lower(), password_hash, username, role, datetime.now()))
                user_id = cur.fetchone()[0]
                self.conn.commit()
                logger.info(f"Created user: {username} (ID: {user_id})")
                return user_id
        except psycopg2.IntegrityError:
            self.conn.rollback()
            logger.warning(f"User creation failed: Email {email} already exists")
            return None

    def verify_password(self, email: str, password: str) -> bool:
        self.ensure_connection()
        query = "SELECT password_hash FROM users WHERE email = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (email.lower(),))
            res = cur.fetchone()
            if res and self._verify_password_hash(password, res[0]):
                return True
        return False
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Get user information by email
        
        Args:
            email: User's email address
            
        Returns:
            Dictionary with user info (without password_hash), or None if not found
        """
        self.ensure_connection()
        
        query = """
            SELECT id, email, username, role, created_at, last_login
            FROM users
            WHERE email = %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (email.lower(),))
                user = cur.fetchone()
                
                if user:
                    logger.debug(f"Found user by email: {email}")
                else:
                    logger.debug(f"User not found by email: {email}")
                
                return user
                
        except psycopg2.Error as e:
            logger.error(f"Error fetching user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Get user information by ID
        
        Args:
            user_id: User's database ID
            
        Returns:
            Dictionary with user info (without password_hash), or None if not found
        """
        self.ensure_connection()
        
        query = """
            SELECT id, email, username, role, created_at, last_login
            FROM users
            WHERE id = %s;
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (user_id,))
                user = cur.fetchone()
                
                if user:
                    logger.debug(f"Found user by ID: {user_id}")
                else:
                    logger.debug(f"User not found by ID: {user_id}")
                
                return user
                
        except psycopg2.Error as e:
            logger.error(f"Error fetching user by ID: {e}")
            return None
    
    def update_last_login(self, user_id: int) -> bool:
        """
        Update user's last login timestamp
        
        Args:
            user_id: User's database ID
            
        Returns:
            True if updated successfully, False otherwise
        """
        self.ensure_connection()
        
        query = """
            UPDATE users
            SET last_login = %s
            WHERE id = %s;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (datetime.now(), user_id))
                updated = cur.rowcount > 0
                self.conn.commit()
                
                if updated:
                    logger.info(f"Updated last login for user ID: {user_id}")
                else:
                    logger.warning(f"Failed to update last login: User ID {user_id} not found")
                
                return updated
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Error updating last login: {e}")
            return False
    
    def generate_jwt_token(self, user_id: int) -> str:
        """
        Generate a JWT token for user
        
        Args:
            user_id: User's database ID
            
        Returns:
            JWT token string
        """
        try:
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
                'iat': datetime.utcnow()  # issued at
            }
            
            token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
            
            logger.info(f"Generated JWT token for user ID: {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT token: {e}")
            raise
    
    def verify_jwt_token(self, token: str) -> Optional[int]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            user_id if token is valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            logger.debug(f"JWT token verified for user ID: {user_id}")
            return user_id
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            return None
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """
        Complete authentication flow: verify credentials and generate token
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Dictionary with {user_id, token, user_info} if successful, None otherwise
        """
        # Verify password
        if not self.verify_password(email, password):
            return None
        
        # Get user info
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        # Update last login
        self.update_last_login(user['id'])
        
        # Generate token
        token = self.generate_jwt_token(user['id'])
        
        return {
            'user_id': user['id'],
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'username': user['username'],
                'role': user['role']
            }
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


# Test/Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing AuthDatabase")
    print("="*60)

    # Get password from user
    import getpass
    db_password = getpass.getpass("Enter PostgreSQL password: ")
    
    
    try:
        with AuthDatabase(password=db_password) as auth_db:
            
            # Test 1: Create user
            print("\n1. Creating test user...")
            user_id = auth_db.create_user(
                email="test@example.com",
                password="password123",
                username="testuser"
            )
            
            if user_id:
                print(f"   ✓ User created with ID: {user_id}")
            else:
                print("   ⚠ User already exists (this is okay for testing)")
            
            # Test 2: Verify correct password
            print("\n2. Testing password verification...")
            is_valid = auth_db.verify_password("test@example.com", "password123")
            print(f"   Correct password: {'✓ Valid' if is_valid else '✗ Invalid'}")
            
            # Test 3: Verify wrong password
            is_valid = auth_db.verify_password("test@example.com", "wrongpassword")
            print(f"   Wrong password: {'✗ Rejected' if not is_valid else '✓ Accepted (BAD!)'}")
            
            # Test 4: Get user by email
            print("\n3. Getting user by email...")
            user = auth_db.get_user_by_email("test@example.com")
            if user:
                print(f"   ✓ Found user: {user['username']} ({user['email']})")
            
            # Test 5: Generate JWT token
            print("\n4. Testing JWT token generation...")
            token = auth_db.generate_jwt_token(user['id'])
            print(f"   ✓ Token generated: {token[:30]}...")
            
            # Test 6: Verify JWT token
            print("\n5. Testing JWT token verification...")
            decoded_user_id = auth_db.verify_jwt_token(token)
            print(f"   ✓ Token verified, user_id: {decoded_user_id}")
            
            # Test 7: Complete authentication flow
            print("\n6. Testing complete authentication flow...")
            auth_result = auth_db.authenticate_user("test@example.com", "password123")
            if auth_result:
                print(f"   ✓ Authentication successful")
                print(f"   User: {auth_result['user']['username']}")
                print(f"   Token: {auth_result['token'][:30]}...")
            
            # Test 8: Try invalid token
            print("\n7. Testing invalid token...")
            fake_token = "invalid.token.string"
            decoded = auth_db.verify_jwt_token(fake_token)
            print(f"   Invalid token rejected: {'✓ Yes' if decoded is None else '✗ No (BAD!)'}")
            
        print("\n" + "="*60)
        print("✓ All authentication tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        logger.exception("Test suite failed")

 