-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    role VARCHAR(20) DEFAULT 'user', -- 'admin' or 'user'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Cameras table
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(255),
    camera_type VARCHAR(50) DEFAULT 'webcam',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User-Camera access mapping
CREATE TABLE user_camera_access (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE CASCADE,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(id), -- who gave access
    UNIQUE(user_id, camera_id) -- prevent duplicate access grants
);

-- Detected plates (NO user_id - belongs to camera)
CREATE TABLE detected_plates (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast queries
CREATE INDEX idx_plate_camera ON detected_plates(camera_id);
CREATE INDEX idx_user_camera ON user_camera_access(user_id);
```

---

## 🔄 How This Works in Practice

### **Scenario 1: User Logs In and Views Plates**
```
1. User A logs in
   ↓
2. Backend queries: "Which cameras does User A have access to?"
   SELECT camera_id FROM user_camera_access WHERE user_id = A
   Result: [1, 2, 5]
   ↓
3. Backend queries: "Get plates from those cameras"
   SELECT * FROM detected_plates 
   WHERE camera_id IN (1, 2, 5)
   ORDER BY detected_at DESC
   LIMIT 50
   ↓
4. Return plates to User A
```

### **Scenario 2: Camera Detects New Plate**
```
1. Camera 2 detects plate "ABC123"
   ↓
2. Backend saves to database:
   INSERT INTO detected_plates (plate_number, camera_id, ...)
   VALUES ('ABC123', 2, ...)
   ↓
3. Who can see this?
   Query: SELECT user_id FROM user_camera_access WHERE camera_id = 2
   Result: [User A, User B] both have access to Camera 2
   ↓
4. When User A or User B refreshes, they both see "ABC123"
```

### **Scenario 3: Admin Grants Camera Access**
```
1. Admin assigns Camera 3 to User C
   ↓
2. Backend inserts:
   INSERT INTO user_camera_access (user_id, camera_id, granted_by)
   VALUES (C, 3, admin_id)
   ↓
3. Now User C can see plates from Camera 3