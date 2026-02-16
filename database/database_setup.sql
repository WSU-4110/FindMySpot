-- database_setup.sql
-- PostgreSQL schema for license plate detection system

-- Create database (run this separately as postgres user)
-- CREATE DATABASE license_plate_db;

-- Connect to the database and run the rest:
-- \c license_plate_db;

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
