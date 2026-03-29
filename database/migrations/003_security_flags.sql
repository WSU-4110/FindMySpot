CREATE TABLE IF NOT EXISTS security_flags (
  id SERIAL PRIMARY KEY,
  session_id INTEGER,
  vehicle_plate VARCHAR(20) NOT NULL,
  flag_type VARCHAR(64) NOT NULL,
  reason VARCHAR(500) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES parking_sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_security_flags_status ON security_flags(status);
CREATE INDEX IF NOT EXISTS idx_security_flags_vehicle ON security_flags(vehicle_plate);
CREATE INDEX IF NOT EXISTS idx_security_flags_session ON security_flags(session_id);
