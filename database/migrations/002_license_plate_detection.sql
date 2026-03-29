-- Add push notification support to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS push_notification_enabled BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN IF NOT EXISTS push_token VARCHAR(255);

-- Create user_vehicles table (users register their license plates)
CREATE TABLE IF NOT EXISTS user_vehicles (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  license_plate VARCHAR(20) NOT NULL,
  vehicle_name VARCHAR(255),
  make_model VARCHAR(255),
  color VARCHAR(50),
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  UNIQUE(user_id, license_plate)
);

-- Create detected_plates table (real-time detection records)
CREATE TABLE IF NOT EXISTS detected_plates (
  id SERIAL PRIMARY KEY,
  license_plate VARCHAR(20) NOT NULL,
  floor INTEGER,
  lot INTEGER,
  location_description VARCHAR(255),
  confidence FLOAT,
  detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  camera_id VARCHAR(50),
  latitude FLOAT,
  longitude FLOAT
);

-- Create notifications table (notification history)
CREATE TABLE IF NOT EXISTS notifications (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  vehicle_id INTEGER,
  detected_plate_id INTEGER,
  title VARCHAR(255) NOT NULL,
  message VARCHAR(500) NOT NULL,
  location_description VARCHAR(255),
  detected_at TIMESTAMP,
  sent_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  read_at TIMESTAMP,
  is_read BOOLEAN DEFAULT false,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (vehicle_id) REFERENCES user_vehicles(id) ON DELETE SET NULL,
  FOREIGN KEY (detected_plate_id) REFERENCES detected_plates(id) ON DELETE SET NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_vehicles_user_id ON user_vehicles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_vehicles_plate ON user_vehicles(license_plate);
CREATE INDEX IF NOT EXISTS idx_detected_plates_plate ON detected_plates(license_plate);
CREATE INDEX IF NOT EXISTS idx_detected_plates_timestamp ON detected_plates(detected_at);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(sent_at);
