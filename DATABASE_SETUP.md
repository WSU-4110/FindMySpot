# PostgreSQL Database Setup Guide

## 📦 Prerequisites

1. **Install PostgreSQL**
   - Windows: Download from https://www.postgresql.org/download/windows/
   - Mac: `brew install postgresql`
   - Linux: `sudo apt-get install postgresql postgresql-contrib`

2. **Install Python dependencies**
   ```bash
   pip install psycopg2-binary
   ```

## 🚀 Setup Steps

### Step 1: Install PostgreSQL

**Windows:**
1. Download PostgreSQL installer from postgresql.org
2. Run installer, set password for `postgres` user (remember this!)
3. Default port is 5432
4. pgAdmin 4 will be installed for GUI management

**Mac/Linux:**
```bash
# Mac
brew install postgresql
brew services start postgresql

# Linux
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Step 2: Create Database

**Option A - Using psql command line:**
```bash
# Connect to PostgreSQL
psql -U postgres

# In psql prompt:
CREATE DATABASE license_plate_db;
\c license_plate_db
\i database_setup.sql
\q
```

**Option B - Using pgAdmin (GUI):**
1. Open pgAdmin 4
2. Right-click "Databases" → "Create" → "Database"
3. Name: `license_plate_db`
4. Click "Save"
5. Right-click the new database → "Query Tool"
6. Open `database_setup.sql` and execute it

**Option C - Quick command line:**
```bash
# Windows
psql -U postgres -c "CREATE DATABASE license_plate_db;"
psql -U postgres -d license_plate_db -f database_setup.sql

# Mac/Linux  
createdb license_plate_db
psql license_plate_db < database_setup.sql
```

### Step 3: Configure Database Connection

Edit `app_simple.py` and update these lines with your PostgreSQL credentials:

```python
db = PlateDatabase(
    host="localhost",
    port=5432,
    database="license_plate_db",
    user="postgres",
    password="YOUR_PASSWORD_HERE"  # ← Change this!
)
```

### Step 4: Test Database Connection

```bash
# Test the database module
python database.py
```

You should see:
```
✓ Connected to PostgreSQL database: license_plate_db
✓ Saved plate 'ABC123' to database (ID: 1)
```

### Step 5: Run the App

```bash
python app_simple.py
```

Look for this message:
```
✓ Connected to PostgreSQL database: license_plate_db
Starting SIMPLIFIED Flask server on http://localhost:5000
```

## 📊 Using the Database

### API Endpoints

Once the app is running, you can access these endpoints:

**Check database status:**
```bash
curl http://localhost:5000/api/database/status
```

**Get recent detections:**
```bash
curl http://localhost:5000/api/database/recent?limit=10
```

**Get today's detections:**
```bash
curl http://localhost:5000/api/database/today
```

**Search for a specific plate:**
```bash
curl http://localhost:5000/api/database/search/10L493
```

**Get statistics:**
```bash
curl http://localhost:5000/api/database/stats
```

**Get most frequently seen plates:**
```bash
curl http://localhost:5000/api/database/most-seen?limit=5
```

### Direct Database Queries

**Connect to database:**
```bash
psql -U postgres -d license_plate_db
```

**Useful queries:**

```sql
-- View all plates
SELECT * FROM detected_plates ORDER BY detected_at DESC LIMIT 10;

-- Plates detected today
SELECT * FROM detected_plates 
WHERE DATE(detected_at) = CURRENT_DATE;

-- Unique plates with count
SELECT 
    plate_number, 
    COUNT(*) as times_seen,
    MAX(detected_at) as last_seen 
FROM detected_plates 
GROUP BY plate_number 
ORDER BY times_seen DESC;

-- Plates in last hour
SELECT * FROM detected_plates 
WHERE detected_at > NOW() - INTERVAL '1 hour';

-- Average confidence by plate
SELECT 
    plate_number,
    COUNT(*) as detections,
    AVG(confidence) as avg_confidence,
    MAX(confidence) as max_confidence
FROM detected_plates
GROUP BY plate_number
ORDER BY detections DESC;
```

## 🔧 Database Schema

```sql
detected_plates:
├── id (PRIMARY KEY, auto-increment)
├── plate_number (VARCHAR(20))
├── detected_at (TIMESTAMP)
├── confidence (FLOAT)
├── camera_id (VARCHAR(50))
└── created_at (TIMESTAMP)
```

## 🐛 Troubleshooting

### "Database connection failed"
- Check PostgreSQL is running: `pg_ctl status` or Task Manager (Windows)
- Verify credentials in `app_simple.py`
- Check if port 5432 is open: `netstat -an | findstr 5432`

### "relation detected_plates does not exist"
- Run the `database_setup.sql` script
- `psql -U postgres -d license_plate_db -f database_setup.sql`

### "password authentication failed"
- Reset postgres password:
  ```bash
  # Windows (as admin)
  psql -U postgres
  ALTER USER postgres PASSWORD 'newpassword';
  
  # Linux
  sudo -u postgres psql
  ALTER USER postgres PASSWORD 'newpassword';
  ```

### App runs without database
The app will work even if database connection fails - it just won't save plates.
Check console for:
```
⚠ Database not available: ...
  App will run without database features
```

## 📈 Data Management

### Export data to CSV
```bash
psql -U postgres -d license_plate_db -c "\COPY (SELECT * FROM detected_plates) TO 'plates.csv' CSV HEADER"
```

### Backup database
```bash
pg_dump -U postgres license_plate_db > backup.sql
```

### Restore database
```bash
psql -U postgres license_plate_db < backup.sql
```

### Clean old data
```python
from database import PlateDatabase

with PlateDatabase() as db:
    # Delete plates older than 30 days
    db.delete_old_plates(days=30)
```

## 🎯 Next Steps

1. **Add user interface** - Create a dashboard to view database records
2. **Add alerts** - Notify when specific plates are detected
3. **Add reporting** - Generate daily/weekly reports
4. **Add authentication** - Secure the API endpoints
5. **Add plate watchlist** - Flag specific plates of interest
