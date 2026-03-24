# INSTALLATION.md — FindMySpot

This document describes the system requirements and step-by-step instructions to build and install FindMySpot.

---

## System Requirements

| Requirement | Minimum Version |
|---|---|
| Node.js | v16+ |
| PostgreSQL | v13+ |
| Python | 3.8+ |
| Redis | Latest stable |
| npm | v8+ |
| pip | v21+ |
| Camera hardware | Required for deployment |

**Operating System:** macOS, Linux, or Windows (PowerShell recommended on Windows)

---

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/FindMySpot.git
cd FindMySpot
```

---

### Step 2 — Set Up the Database

**On macOS / Linux:**

```bash
psql -U postgres -c "CREATE DATABASE findmyspot;"
```

**On Windows (run setup-database.bat or PowerShell script):**

```powershell
.\setup-database.ps1
```

This runs all three migrations automatically:
- `001_init.sql` — users, vehicles, parking spots, sessions
- `002_license_plate_detection.sql` — detected plates, notifications, user vehicles
- `003_security_flags.sql` — security flag tracking

---

### Step 3 — Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Open `.env` and fill in your values:

```
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=findmyspot
PORT=3000
```

---

### Step 4 — Install Backend Dependencies

```bash
cd backend
npm install
```

---

### Step 5 — Install AI Service Dependencies

```bash
cd ../ai-service
pip install -r requirements.txt
```

Key packages installed:
- `easyocr` — OCR for license plate text
- `opencv-python` — image processing
- `ultralytics` — YOLO object detection
- `flask` / `flask-cors` — AI service API
- `requests` — backend communication

---

### Step 6 — Start the Backend Server

```bash
cd ../backend
npm run dev
```

The API will be available at `http://localhost:3000`.  
Visit `http://localhost:3000` in your browser to see the full API endpoint list.

---

### Step 7 — Start the AI Detection Service

In a separate terminal:

```bash
cd ai-service
python app.py
```

Make sure a camera is connected. Press `q` to quit the camera window.

---

### Step 8 — Run the Mobile App

**Install dependencies:**

```bash
cd ../mobile-app
npm install
```

**Run on iOS:**

```bash
npx react-native run-ios
```

**Run on Android:**

```bash
npx react-native run-android
```

Alternatively, serve the web version locally:

```bash
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

---

## Verify the Installation

Run the test suite to confirm everything is connected:

```bash
python test_system.py
```

Expected output: all tests passing with a registered user, vehicles, and detection records.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to API` | Make sure `npm run dev` is running in `/backend` |
| `DB_PASSWORD missing` | Ensure `.env` file exists in `/backend` with correct credentials |
| `Camera not found` | Check that a webcam is connected before running `app.py` |
| `pip install fails` | Use `pip install -r requirements.txt --break-system-packages` |
| `psql: command not found` | Add PostgreSQL `bin/` directory to your system PATH |

---

## Contributors

- Asmita Bhandari
- Mirza Sneha
- Jennifer Lopez
- Varun Kodikal
- Tristan Mejia
