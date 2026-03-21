# FindMySpot - Parking Management System

**FindMySpot** is a smart parking management system that uses license plate detection to help users locate their parked vehicles. The system integrates camera-based OCR with a web application to provide real-time vehicle location tracking.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Security](#security)
- [Testing](#testing)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

---

## ✨ Features

- 🔐 **Secure Authentication** - JWT-based authentication with bcrypt password hashing
- 🚗 **Vehicle Management** - Track and manage vehicle parking locations
- 📷 **Camera Integration** - Automatic license plate detection using EasyOCR
- 🗺️ **Location Tracking** - Real-time parking spot updates
- 📊 **PostgreSQL Database** - Persistent data storage with ACID compliance
- 🔒 **Security** - SQL injection prevention, password hashing, JWT tokens

---

## 🛠️ Tech Stack

### Backend
- **Python 3.11+** - Main backend language
- **Flask 3.0.0** - Web framework
- **PostgreSQL 14+** - Database
- **psycopg2** - PostgreSQL adapter
- **bcrypt** - Password hashing
- **PyJWT** - JWT token generation
- **python-dotenv** - Environment configuration

### Camera System
- **OpenCV** - Image processing
- **EasyOCR** - License plate recognition
- **NumPy** - Array operations

### Frontend
- **HTML5/CSS3** - User interface
- **JavaScript (Vanilla)** - Client-side logic
- **Fetch API** - HTTP requests

---

##  Architecture
```
┌─────────────┐
│   Camera    │ ──── Detects License Plate
└──────┬──────┘
       │ POST /api/plate/detected
       ▼
┌─────────────────────┐
│   Flask Backend     │
│  (Port 5000)        │
├─────────────────────┤
│ • Authentication    │
│ • Vehicle Management│
│ • Database Layer    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   PostgreSQL DB     │
│  (Port 5432)        │
├─────────────────────┤
│ • users             │
│ • vehicles          │
│ • parking_sessions  │
└─────────────────────┘
       ▲
       │
┌──────┴──────┐
│  Frontend   │ ──── User Interface
│ (Port 8000) │
└─────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 14 or higher
- pip (Python package manager)
- Git

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/WSU-4110/FindMySpot.git
cd FindMySpot
```

#### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the root directory:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=license_plate_db
DB_USER=postgres
DB_PASSWORD=your_password_here

# JWT Secret (generate with: python -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET=your_secret_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

**⚠️ Important:** Never commit the `.env` file to version control!

#### 5. Set Up Database
```bash
# Run database setup script
python setup_database.py --password your_password

# Or set environment variable and run
$env:DB_PASSWORD="your_password"
python setup_database.py
```

This will:
- Create the `license_plate_db` database
- Create all required tables
- Set up indexes for performance

#### 6. Run the Backend Server
```bash
python app_api.py
```

The server will start on `http://localhost:5000`

#### 7. Run the Frontend (Optional)
```bash
# Simple HTTP server
python -m http.server 8000

# Or just open index.html in your browser
start index.html
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Authentication Endpoints

#### 1. **Sign Up**

Create a new user account.
```http
POST /api/auth/signup
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "email": "john@example.com",
    "username": "john_doe"
  }
}
```

---

#### 2. **Login**

Authenticate existing user.
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "email": "john@example.com",
    "username": "john_doe"
  }
}
```

---

### Vehicle Management Endpoints

**Note:** All vehicle endpoints require authentication. Include JWT token in header:
```http
Authorization: Bearer <your_jwt_token>
```

---

#### 3. **Check-In Vehicle**

Record vehicle arrival and parking location.
```http
POST /api/vehicle/checkin
Authorization: Bearer <token>
Content-Type: application/json

{
  "plate": "ABC123",
  "floor": "Floor 2",
  "spot": "A5"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Vehicle checked in successfully",
  "vehicle": {
    "id": 1,
    "license_plate": "ABC123",
    "floor": "Floor 2",
    "spot": "A5",
    "checked_in_at": "2026-03-14T13:30:00Z"
  }
}
```

---

#### 4. **Find Vehicle**

Locate a parked vehicle.
```http
POST /api/vehicle/find
Authorization: Bearer <token>
Content-Type: application/json

{
  "plate": "ABC123"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "vehicle": {
    "license_plate": "ABC123",
    "floor": "Floor 2",
    "spot": "A5",
    "parked_since": "2026-03-14T13:30:00Z"
  }
}
```

---

### Camera Integration Endpoint

#### 5. **Plate Detected**

Camera system reports detected license plate.
```http
POST /api/plate/detected
Content-Type: application/json

{
  "plate_number": "ABC123",
  "camera_id": 1,
  "floor": "Floor 2",
  "spot": "A5",
  "confidence": 0.95,
  "detected_at": "2026-03-14T13:30:00Z"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "is_registered": true,
  "user_email": "john@example.com",
  "location_updated": true
}
```

---

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username      VARCHAR(50) UNIQUE NOT NULL,
    role          VARCHAR(20) DEFAULT 'user',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login    TIMESTAMP
);
```

### Vehicles Table
```sql
CREATE TABLE vehicles (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    license_plate VARCHAR(20) NOT NULL,
    make          VARCHAR(50),
    model         VARCHAR(50),
    color         VARCHAR(30),
    floor         VARCHAR(20),
    spot          VARCHAR(20),
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_vehicles_user ON vehicles(user_id);
CREATE INDEX idx_vehicles_plate ON vehicles(license_plate);
```

### Cameras Table
```sql
CREATE TABLE cameras (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    location    VARCHAR(255),
    camera_type VARCHAR(50) DEFAULT 'webcam',
    is_active   BOOLEAN DEFAULT true,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Detected Plates Table
```sql
CREATE TABLE detected_plates (
    id           SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    camera_id    INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
    vehicle_id   INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    detected_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence   FLOAT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_plate_number ON detected_plates(plate_number);
CREATE INDEX idx_detected_at ON detected_plates(detected_at);
```

---

## 🔒 Security

### Password Security

- **bcrypt hashing** with salt rounds (cost factor: 10)
- Passwords are **never** stored in plain text
- One-way encryption prevents password recovery
```python
import bcrypt

# Hashing
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Verification
is_valid = bcrypt.checkpw(password.encode('utf-8'), stored_hash)
```

### JWT Authentication

- **Tokens expire after 24 hours**
- Signed with HMAC-SHA256
- Contains user ID claim
- Stateless authentication
```python
import jwt
from datetime import datetime, timedelta

payload = {
    'user_id': user_id,
    'exp': datetime.utcnow() + timedelta(days=1)
}
token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
```

### SQL Injection Prevention

All queries use **parameterized statements**:
```python
# ✅ SAFE
cur.execute("SELECT * FROM users WHERE email = %s", (email,))

# ❌ UNSAFE (never do this!)
cur.execute(f"SELECT * FROM users WHERE email = '{email}'")
```

### Best Practices

- ✅ Environment variables for secrets
- ✅ HTTPS in production (add SSL certificates)
- ✅ Input validation on all endpoints
- ✅ Rate limiting (add in production)
- ✅ CORS configuration
- ✅ Database connection pooling

---

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_auth.py
```

### Manual API Testing

#### Using PowerShell (Windows):
```powershell
# Signup
Invoke-RestMethod -Uri http://localhost:5000/api/auth/signup `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"name":"Test User","email":"test@example.com","password":"test123"}'

# Login
$response = Invoke-RestMethod -Uri http://localhost:5000/api/auth/login `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"email":"test@example.com","password":"test123"}'

# Save token
$token = $response.token

# Check-in vehicle
Invoke-RestMethod -Uri http://localhost:5000/api/vehicle/checkin `
  -Method Post `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body '{"plate":"ABC123","floor":"Floor 1","spot":"A1"}'
```

#### Using curl (Linux/macOS):
```bash
# Signup
curl -X POST http://localhost:5000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"test123"}'

# Login
TOKEN=$(curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}' | jq -r '.token')

# Check-in vehicle
curl -X POST http://localhost:5000/api/vehicle/checkin \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"plate":"ABC123","floor":"Floor 1","spot":"A1"}'
```

---

## 🤝 Contributing

### Branch Strategy
```
main           - Production-ready code
varun          - Backend development
mirza          - Frontend development
<name>_homework4 - Individual homework branches
```

### Workflow

1. **Create a branch** from `main`:
```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
```

2. **Make changes** and commit:
```bash
   git add .
   git commit -m "Add feature: description"
```

3. **Push to GitHub**:
```bash
   git push -u origin feature/your-feature-name
```

4. **Create Pull Request** on GitHub

5. **Code Review** - Wait for team review

6. **Merge** after approval

### Commit Message Guidelines
```
✅ Good:
- "Add JWT authentication to login endpoint"
- "Fix SQL injection vulnerability in user search"
- "Update README with API documentation"

❌ Bad:
- "Fixed stuff"
- "Update"
- "asdf"
```

---

## 👥 Team

### Spring 2026 - CSC 4110

- **Varun** - Backend & Database Architecture
- **Mirza** - Frontend Development
- **Asmita** - Authentication & Security
- **Team Leader** - Camera Integration
- **Jennifer** - [Role]
- **Tristan** - [Role]

---

## 📂 Project Structure
```
FindMySpot/
├── backend/                    # Node.js backend (alternative)
│   ├── config/
│   ├── controllers/
│   ├── models/
│   └── routes/
├── static/                     # Frontend assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/                  # HTML templates
├── app_api.py                 # Flask API (main backend)
├── auth_database.py           # Authentication module
├── vehicle_database.py        # Vehicle management module
├── setup_database.py          # Database initialization
├── camera_detection.py        # License plate detection
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🐛 Known Issues

- Camera detection accuracy varies in low light
- JWT tokens don't support refresh tokens yet
- No rate limiting on API endpoints (add in production)
- Frontend needs responsive design improvements

See [GitHub Issues](https://github.com/WSU-4110/FindMySpot/issues) for full list.

---

## 🔮 Future Enhancements

- [ ] Mobile app (iOS/Android)
- [ ] Real-time notifications (WebSocket)
- [ ] Payment integration for parking fees
- [ ] Admin dashboard for parking management
- [ ] Multi-language support
- [ ] Dark mode for frontend
- [ ] Parking spot reservations
- [ ] Historical parking analytics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/WSU-4110/FindMySpot/issues)
- **Email**: [your-email@wayne.edu](mailto:your-email@wayne.edu)
- **Slack**: #findmyspot-dev

---

## 🙏 Acknowledgments

- Wayne State University - CSC 4110 Course
- EasyOCR - License plate recognition
- Flask Documentation
- PostgreSQL Community

---

**Built with ❤️ by the FindMySpot Team**

*Last Updated: March 14, 2026*
