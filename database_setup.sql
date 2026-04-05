-- database_setup.sql
-- PostgreSQL schema for license plate detection system

-- Create database (run this separately as postgres user)
-- CREATE DATABASE license_plate_db;

-- Connect to the database and run the rest:
-- \c license_plate_db;


-- Users
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Cameras
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(255),
    camera_type VARCHAR(50) DEFAULT 'webcam',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User <-> Camera access
CREATE TABLE IF NOT EXISTS user_camera_access (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, camera_id)
);

-- Vehicles  (fixed: users -> user, added is_primary, removed trailing comma)
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

CREATE INDEX IF NOT EXISTS idx_vehicles_user ON vehicles(user_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(license_plate);
-- Ensure one primary vehicle per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_primary_per_user
    ON vehicles(user_id) WHERE is_primary = true;

-- Detected plates  (single definition, added vehicle_id, removed trailing comma)
CREATE TABLE IF NOT EXISTS detected_plates (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_plate_camera ON detected_plates(camera_id);
CREATE INDEX IF NOT EXISTS idx_plate_number ON detected_plates(plate_number);
CREATE INDEX IF NOT EXISTS idx_plate_vehicle ON detected_plates(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_plate_detected_at ON detected_plates(detected_at);

-- Parking sessions
CREATE TABLE IF NOT EXISTS parking_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    spot_number VARCHAR(10),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    duration_minutes INTEGER,
    rate_per_hour DECIMAL(10, 2),
    amount_charged DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_parking_user ON parking_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_vehicle ON parking_sessions(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON parking_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_entry ON parking_sessions(entry_time);

-- Seed cameras
INSERT INTO cameras (name, location, camera_type) VALUES
    ('Main Entrance', '123 Main St', 'webcam'),
    ('Parking Lot A', '456 Elm St', 'ip_camera'),
    ('Parking Lot B', '789 Oak St', 'ip_camera');

-- View: 100 most recent detections
CREATE OR REPLACE VIEW recent_plates AS
SELECT

-- Create plates table
CREATE TABLE IF NOT EXISTS detected_plates (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    camera_id VARCHAR(50) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX idx_plate_number ON detected_plates(plate_number);
CREATE INDEX idx_detected_at ON detected_plates(detected_at);

-- Create a view for recent detections
CREATE OR REPLACE VIEW recent_plates AS
SELECT 
    id,
    plate_number,
    detected_at,
    confidence,
    camera_id
FROM detected_plates
ORDER BY detected_at DESC
LIMIT 100;

-- Useful sample queries:

-- All plates detected today:
-- SELECT * FROM detected_plates WHERE DATE(detected_at) = CURRENT_DATE;

-- Unique plates with detection count:
-- SELECT plate_number, COUNT(*) AS times_seen, MAX(detected_at) AS last_seen
-- FROM detected_plates
-- GROUP BY plate_number
-- ORDER BY times_seen DESC;

-- Plates seen in the last hour:
-- SELECT * FROM detected_plates WHERE detected_at > NOW() - INTERVAL '1 hour';-- database_setup.sql
-- PostgreSQL schema for license plate detection system

-- Create database (run this separately as postgres user)
-- CREATE DATABASE license_plate_db;

-- Connect to the database and run the rest:
-- \c license_plate_db;

-- Users
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Cameras
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(255),
    camera_type VARCHAR(50) DEFAULT 'webcam',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User <-> Camera access
CREATE TABLE IF NOT EXISTS user_camera_access (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, camera_id)
);

-- Vehicles  (fixed: users -> user, added is_primary, removed trailing comma)
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

CREATE INDEX IF NOT EXISTS idx_vehicles_user ON vehicles(user_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(license_plate);
-- Ensure one primary vehicle per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_primary_per_user
    ON vehicles(user_id) WHERE is_primary = true;

-- Detected plates  (single definition, added vehicle_id, removed trailing comma)
CREATE TABLE IF NOT EXISTS detected_plates (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_plate_camera ON detected_plates(camera_id);
CREATE INDEX IF NOT EXISTS idx_plate_number ON detected_plates(plate_number);
CREATE INDEX IF NOT EXISTS idx_plate_vehicle ON detected_plates(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_plate_detected_at ON detected_plates(detected_at);

-- Parking sessions
CREATE TABLE IF NOT EXISTS parking_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    spot_number VARCHAR(10),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    duration_minutes INTEGER,
    rate_per_hour DECIMAL(10, 2),
    amount_charged DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_parking_user ON parking_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_vehicle ON parking_sessions(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON parking_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_entry ON parking_sessions(entry_time);

-- Seed cameras
INSERT INTO cameras (name, location, camera_type) VALUES
    ('Main Entrance', '123 Main St', 'webcam'),
    ('Parking Lot A', '456 Elm St', 'ip_camera'),
    ('Parking Lot B', '789 Oak St', 'ip_camera');

-- View: 100 most recent detections
CREATE OR REPLACE VIEW recent_plates AS
SELECT
    id,
    plate_number,
    detected_at,
    confidence,
    camera_id
FROM detected_plates
ORDER BY detected_at DESC
LIMIT 100;

-- Useful sample queries:

-- All plates detected today:
-- SELECT * FROM detected_plates WHERE DATE(detected_at) = CURRENT_DATE;

-- Unique plates with detection count:
-- SELECT plate_number, COUNT(*) AS times_seen, MAX(detected_at) AS last_seen
-- FROM detected_plates
-- GROUP BY plate_number
-- ORDER BY times_seen DESC;

-- Plates seen in the last hour:

-- Sample queries you can use:

-- Get all plates detected today
-- SELECT * FROM detected_plates WHERE DATE(detected_at) = CURRENT_DATE;

-- Get unique plates with count
-- SELECT plate_number, COUNT(*) as times_seen, MAX(detected_at) as last_seen 
-- FROM detected_plates 
-- GROUP BY plate_number 
-- ORDER BY times_seen DESC;

-- Get plates in last hour
-- SELECT * FROM detected_plates WHERE detected_at > NOW() - INTERVAL '1 hour';
