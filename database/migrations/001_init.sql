CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(64) NOT NULL,
  token VARCHAR(64),
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehicles (
  id SERIAL PRIMARY KEY,
  plate VARCHAR(16) UNIQUE NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parking_spots (
  id SERIAL PRIMARY KEY,
  floor INTEGER NOT NULL,
  lot INTEGER NOT NULL,
  occupied BOOLEAN DEFAULT false,
  vehicle_plate VARCHAR(16),
  check_in_time TIMESTAMP,
  CONSTRAINT uniq_floor_lot UNIQUE (floor, lot)
);

CREATE TABLE IF NOT EXISTS parking_sessions (
  id SERIAL PRIMARY KEY,
  vehicle_plate VARCHAR(16) NOT NULL,
  floor INTEGER NOT NULL,
  lot INTEGER NOT NULL,
  check_in_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  check_out_time TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vehicle_plate ON vehicles(plate);
CREATE INDEX IF NOT EXISTS idx_session_vehicle ON parking_sessions(vehicle_plate);
CREATE INDEX IF NOT EXISTS idx_session_active ON parking_sessions(vehicle_plate, check_out_time);

-- Initialize parking spots if not already present
INSERT INTO parking_spots (floor, lot, occupied) 
SELECT f.floor, l.lot, false
FROM generate_series(1, 5) f(floor)
CROSS JOIN generate_series(1, 5) l(lot)
ON CONFLICT (floor, lot) DO NOTHING;
