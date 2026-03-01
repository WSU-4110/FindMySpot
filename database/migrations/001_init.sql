CREATE TABLE IF NOT EXISTS vehicles (
  id SERIAL PRIMARY KEY,
  plate VARCHAR(16) UNIQUE NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parking_spots (
  id SERIAL PRIMARY KEY,
  floor VARCHAR(16) NOT NULL,
  spot VARCHAR(16) NOT NULL,
  CONSTRAINT uniq_floor_spot UNIQUE (floor, spot)
);

CREATE TABLE IF NOT EXISTS parking_sessions (
  id SERIAL PRIMARY KEY,
  vehicle_id INTEGER NOT NULL REFERENCES vehicles(id),
  spot_id INTEGER NOT NULL REFERENCES parking_spots(id),
  check_in_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  check_out_time TIMESTAMP NULL
);

CREATE INDEX IF NOT EXISTS idx_vehicle_plate ON vehicles(plate);
CREATE INDEX IF NOT EXISTS idx_session_vehicle ON parking_sessions(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_session_spot ON parking_sessions(spot_id);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_spot
  ON parking_sessions(spot_id)
  WHERE check_out_time IS NULL;
