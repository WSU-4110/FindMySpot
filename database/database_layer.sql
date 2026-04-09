
from auth_database import AuthDatabase

db = AuthDatabase()

# Test user creation
user_id = db.create_user("test@gmail.com", "password123", "testuser")
print(f"Created user: {user_id}")

# Test login
is_valid = db.verify_password("test@email.com", "password123")
print(f"Password valid: {is_valid}")

# Test wrong password
is_valid = db.verify_password("test@email.com", "wrongpassword")
print(f"Password rejected: {not is_valid}")

# Test JWT
token = db.generate_jwt_token(user_id)
print(f"Generated token: {token[:20]}...")

decoded_user_id = db.verify_jwt_token(token)
print(f"Token decoded to user_id: {decoded_user_id}")

