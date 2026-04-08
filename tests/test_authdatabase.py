import pytest
from auth_database import AuthDatabase
# 1. Setup a fixture to handle the database lifecycle
@pytest.fixture(scope="module")
def auth_db():
    # Setup: Initialize connection (Uses your env vars or defaults)
    db = AuthDatabase()
    yield db
    # Teardown: Close connection after all tests in this file run
    db.close()

# 2. Test User Creation
def test_create_user(auth_db):
    email = "pytest_user@example.com"
    # Ensure a clean slate (optional, depends on your DB state)
    user_id = auth_db.create_user(
        email=email,
        password="securepassword123",
        username="PytestUser"
    )
    assert user_id is not None
    assert isinstance(user_id, int)

# 3. Test Duplicate User Creation Handling
def test_create_duplicate_user(auth_db):
    email = "pytest_user@example.com"
    user_id = auth_db.create_user(
        email=email,
        password="anotherpassword",
        username="Duplicate"
    )
    assert user_id is None  # Should return None per your implementation
# 4. Test Password Verification
@pytest.mark.parametrize("email, password, expected", [
    ("pytest_user@example.com", "securepassword123", True),
    ("pytest_user@example.com", "wrong_pass", False),
    ("nonexistent@example.com", "any_pass", False),
])
def test_verify_password(auth_db, email, password, expected):
    assert auth_db.verify_password(email, password) == expected


