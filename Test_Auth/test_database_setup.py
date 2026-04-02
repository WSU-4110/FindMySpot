import pytest
import psycopg2
from unittest.mock import patch
from setup_database import create_database_if_not_exists, run_ddl_statements


@pytest.fixture  # <--- Add this decorator!
def mock_config():
    return {
        "host": "localhost",
        "port": 5432,
        "dbname": "test_spot_db",
        "user": "postgres",
        "password": "password"
    }

# 2. TEST CASE: Database Creation Success
@patch('psycopg2.connect')
def test_create_database_success(mock_connect, mock_config):
    # Setup mock to simulate DB doesn't exist yet
    mock_cursor = mock_connect.return_value.cursor.return_value
    mock_cursor.fetchone.return_value = None  # Simulate DB not existing

    # Actually call the function
    result = create_database_if_not_exists(mock_config)
    
    assert mock_cursor.execute.called

# 3. TEST CASE: Database Already Exists
@patch('psycopg2.connect')
def test_create_database_already_exists(mock_connect, mock_config):
    mock_cursor = mock_connect.return_value.cursor.return_value
    mock_cursor.fetchone.return_value = (1,)  # Simulate DB exists

    assert create_database_if_not_exists(mock_config) is True
    # Should not attempt to CREATE DATABASE if it exists
    # (Checking if 'CREATE DATABASE' was in any call arguments)
    calls = [call[0][0] for call in mock_cursor.execute.call_args_list if isinstance(call[0][0], str)]
    assert not any("CREATE DATABASE" in s for s in calls)

# 4. TEST CASE: Connection Failure Handling
@patch('psycopg2.connect')
def test_database_connection_error(mock_connect, mock_config):
    mock_connect.side_effect = Exception("Connection Refused")
    assert create_database_if_not_exists(mock_config) is False

# 5. TEST CASE: DDL Execution Success
@patch('psycopg2.connect')
def test_run_ddl_statements_success(mock_connect, mock_config):
    mock_cursor = mock_connect.return_value.cursor.return_value
    assert run_ddl_statements(mock_config) is True
    assert mock_connect.return_value.commit.called

# 6. TEST CASE: DDL Execution Failure & Rollback
@patch('psycopg2.connect')
def test_run_ddl_statements_rollback(mock_connect, mock_config):
    mock_cursor = mock_connect.return_value.cursor.return_value
    mock_cursor.execute.side_effect = Exception("Syntax Error in SQL")
    
    assert run_ddl_statements(mock_config) is False
    # Ensure rollback was called on error
    assert mock_connect.return_value.rollback.called

# 7. TEST CASE: User Table Schema Verification (Bonus/Requirement check)
@patch('psycopg2.connect')
def test_ddl_contains_users_table(mock_connect, mock_config):
    from setup_database import DDL_STATEMENTS
    # Verify the DDL list contains the essential table
    user_ddl = any("create table in stmt.lower()" and "users" in stmt.lower() for stmt in DDL_STATEMENTS
    )
    assert user_ddl is True, "Could not find a CREATE TABLE statement for 'users' in DDL_STATEMENTS"