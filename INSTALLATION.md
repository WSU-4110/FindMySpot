# FindMySpot Installation Guide

This guide provides a complete step-by-step setup for running the FindMySpot system locally. It reflects the current repository structure and ensures all components (backend, database, AI pipeline, and frontend) are correctly configured for full functionality.

It is written to match the repository as it exists now, including:
- the Express/PostgreSQL backend
- the Python OCR/camera script at the repository root
- the auxiliary AI integration service in `ai-service/`
- the static HTML/CSS frontend pages in `mobile-app/`

---
## Quick Start (Recommended)
For a fast setup:
1. Clone the repository  
2. Create the PostgreSQL database  
3. Run database migrations  
4. Configure backend `.env`  
5. Start backend (`npm run dev`)  
6. Install Python dependencies  
7. Run `app.py` or simulator  
8. Serve frontend  

See detailed steps below for full setup.

## 1. Prerequisites

Install the following first:
### Required software
- **Git**
- **Node.js** 16+
- **npm** 8+
- **PostgreSQL** 13+
- **Python** 3.8+
- **pip** 21+

### Recommended
- **PowerShell** on Windows
- **VS Code** or another editor
- A **webcam** for running the live OCR pipeline in `app.py`

---

## 2. Clone the Repository

```bash
git clone https://github.com/WSU-4110/FindMySpot.git
cd FindMySpot
```

---

## 3. Review the Important Runtime Pieces

Before installing, know which parts of the repo are actually used:
### Backend
Location: `backend/`
- Express server
- PostgreSQL connection
- route/controller/model structure
- local API on port `3000` by default

### Database migrations
Location: `database/migrations/`
- `001_init.sql`
- `002_license_plate_detection.sql`
- `003_security_flags.sql`

### Python detection code
There are two Python-related components:
1. **Root `app.py`**
   - live OCR/camera pipeline
   - OpenCV + EasyOCR
   - reads `camera_config.json`
   - posts detections to backend

2. **`ai-service/detection_service.py`**
   - helper class for reporting detections
   - simulation/testing utility

### Frontend
Location: `mobile-app/`
- current repo content is static HTML/CSS pages
- serve them locally with a simple HTTP server

---

## 4. Set Up PostgreSQL

### Create the database
Open PostgreSQL and create a database named `findmyspot`.
Example:

```bash
psql -U postgres -c "CREATE DATABASE findmyspot;"
```

If the database already exists, PostgreSQL will warn you. That is fine.

---

## 5. Run the Database Migrations

The repo contains three migration files in `database/migrations/`.
### Option A: use the provided Windows helper
From the project root:

```powershell
.\setup-database.ps1
```

or

```bat
setup-database.bat
```

### Option B: run migrations manually
If you prefer to run them yourself, apply these SQL files in order:
1. `database/migrations/001_init.sql`
2. `database/migrations/002_license_plate_detection.sql`
3. `database/migrations/003_security_flags.sql`

Example:
```bash
psql -U postgres -d findmyspot -f database/migrations/001_init.sql
psql -U postgres -d findmyspot -f database/migrations/002_license_plate_detection.sql
psql -U postgres -d findmyspot -f database/migrations/003_security_flags.sql
```

### Verify migration success
Run a quick query:

```sql
\dt
```

Or inspect tables in pgAdmin.

---

## 6. Configure Backend Environment Variables

Go into the backend directory:
```bash
cd backend
```

Create your local environment file from the template:
### macOS / Linux
```bash
cp .env.example .env
```

### Windows PowerShell
```powershell
Copy-Item .env.example .env
```

Edit `.env` so it matches local PostgreSQL credentials:
```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=findmyspot
PORT=3000
```

Recommended for easier debugging:
```env
NODE_ENV=development
```

---

## 7. Install Backend Dependencies

Inside `backend/`:
```bash
npm install
```

The backend package file currently defines:
- `express`
- `cors`
- `dotenv`
- `pg`
- `nodemon` for development

---

## 8. Start the Backend Server

From `backend/`:

```bash
npm run dev
```

Or for a non-nodemon run:

```bash
npm start
```

### What should happen
The server should:
- load environment variables
- initialize the database connection
- register all route groups
- start listening on port `3000`

### Quick backend checks
Open these in a browser or test with curl/Postman:

#### Health endpoint
```text
http://localhost:3000/health
```

Expected result: JSON showing the API is running.
#### Root endpoint
```text
http://localhost:3000/
```

Expected result: JSON listing grouped endpoints for:
- auth
- vehicles
- detection
- notifications
- parking

---

## 9. Install Python Dependencies

Return to the repository root if needed, then install the Python requirements used by the AI-related code.
```bash
pip install -r ai-service/requirements.txt
```
The current requirements file includes:
- `flask`
- `flask-cors`
- `opencv-python`
- `easyocr`
- `numpy`
- `sqlalchemy`
- `psycopg2-binary`
- `ultralytics`
- `requests`

### Windows tip
If Python package installation fails because of environment restrictions, use a virtual environment.

#### Create a virtual environment
```bash
python -m venv .venv
```

#### Activate it on Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

#### Activate it on macOS / Linux
```bash
source .venv/bin/activate
```

Then rerun:

```bash
pip install -r ai-service/requirements.txt
```

---

## 10. Configure Camera Mapping

The root OCR pipeline reads `camera_config.json`.
This file maps camera IDs `0` through `9` to floors and lot numbers.

Examples:
- camera `0` -> floor 1, lot 1
- camera `1` -> floor 1, lot 2
- camera `2` -> floor 2, lot 1
- camera `9` -> floor 5, lot 2

### Choose the camera ID before launching `app.py`

#### Windows PowerShell
```powershell
$env:CAMERA_ID=0
```

#### macOS / Linux
```bash
export CAMERA_ID=0
```

If you do not set `CAMERA_ID`, the script defaults to `0`.

---

## 11. Run the Live OCR / Detection Script

From the **repository root**:

```bash
python app.py
```

### Important note
Use the root-level `app.py` for the live camera flow. The repository also contains `ai-service/detection_service.py`, but that file is an integration helper and simulator, not the main live camera script.

### What the script does
- opens the selected camera with OpenCV
- loads `camera_config.json`
- assigns the correct floor and lot based on `CAMERA_ID`
- runs OCR on frames using EasyOCR
- filters plate-like text
- posts detections to:

```text
http://localhost:3000/api/detection/record
```

### Exit the script
Press `q` in the OpenCV window.

---

## 12. Run the Detection Service Simulator (Optional)

If you want to simulate plate detections instead of using a live camera, run:
```bash
python ai-service/detection_service.py
```

This utility sends example detection payloads to the backend detection endpoint.
Use it when:
- you want to test backend detection ingestion quickly
- you do not have camera hardware available
- you want simple end-to-end API testing

---

## 13. Serve the Frontend Pages

The `mobile-app/` directory currently contains HTML/CSS files, so the easiest way to view them is to serve them locally.

From the project root:

```bash
python -m http.server 8080
```

Then open pages such as:
```text
http://localhost:8080/mobile-app/index.html
http://localhost:8080/mobile-app/login.html
http://localhost:8080/mobile-app/signup.html
http://localhost:8080/mobile-app/vehicles.html
http://localhost:8080/mobile-app/notifications.html
http://localhost:8080/mobile-app/history.html
```

You can also `cd mobile-app` first and serve directly from there.

---

## 14. Run the System Test Script

Once the backend and database are ready, run:
```bash
python test_system.py
```

This test script is configured to target:

```text
http://localhost:3000
```

It exercises flows including:
- health check
- user registration
- vehicle-related requests
- detection recording
- notification checks

### Before running tests, confirm:
- PostgreSQL is running
- backend is running on port `3000`
- migrations finished successfully
- `.env` values are correct

---

## 15. Recommended Startup Order

For the smoothest local setup, start things in this order:

### Terminal 1: backend
```bash
cd backend
npm run dev
```

### Terminal 2: optional frontend server
```bash
python -m http.server 8080
```

### Terminal 3: live OCR pipeline or simulator
For live camera:
```bash
python app.py
```

For simulated detections:
```bash
python ai-service/detection_service.py
```

### Terminal 4: verification
```bash
python test_system.py
```

---

## 16. Basic Verification Checklist

Use this checklist after setup:
### Database
- [ ] `findmyspot` database exists
- [ ] all three migration files were applied
- [ ] expected tables appear in pgAdmin or `\dt`

### Backend
- [ ] `npm run dev` starts without crashing
- [ ] `http://localhost:3000/health` responds correctly
- [ ] `http://localhost:3000/` shows endpoint documentation JSON

### Python
- [ ] `pip install -r ai-service/requirements.txt` succeeds
- [ ] `python app.py` opens camera successfully, or simulator runs successfully
- [ ] detections can be posted to the backend

### Frontend
- [ ] static files load from `http://localhost:8080`
- [ ] login/signup/vehicle pages render correctly

### Testing
- [ ] `python test_system.py` reaches the API
- [ ] registration and detection flows work

---

## 17. Troubleshooting

### Problem: `psql: command not found`
Fix:
- install PostgreSQL command-line tools
- add PostgreSQL `bin` directory to your system PATH
- or use pgAdmin instead

### Problem: backend cannot connect to PostgreSQL
Fix:
- verify PostgreSQL service is running
- recheck `.env` values
- make sure database name is `findmyspot`
- confirm migrations were run against the correct database

### Problem: `DB_PASSWORD` or other env values seem missing
Fix:
- ensure `backend/.env` exists
- verify it was created from `.env.example`
- restart the backend after changing `.env`

### Problem: `npm run dev` fails
Fix:
- rerun `npm install`
- confirm Node.js and npm versions are modern enough
- inspect `backend/package.json` scripts

### Problem: `python app.py` fails to open the camera
Fix:
- check camera permissions
- make sure another program is not using the camera
- verify `CAMERA_ID`
- confirm the camera is physically connected

### Problem: `camera_config.json` not found or invalid
Fix:
- run `python app.py` from the repository root
- confirm the file exists at the top level of the repo
- check that the JSON syntax is valid

### Problem: OCR dependencies install slowly or fail
Fix:
- use a virtual environment
- update pip
- install Visual C++ build tools if required on Windows
- retry after ensuring Python version compatibility

### Problem: frontend pages do not load correctly when opened by double-clicking
Fix:
- serve them with `python -m http.server 8080` instead of opening files directly

### Problem: system tests fail
Fix:
- make sure backend is already running
- verify the API is on `http://localhost:3000`
- ensure the database schema exists
- inspect console output for the first failing request

---

## 18. Clean Local Development Recommendations

Although the repository includes certain development-related files for ease of setup, it is recommended that local environments follow standard best practices such as:
- use `backend/.env.example` as the template for your own `.env`
- do not commit real passwords
- avoid committing `node_modules`
- use a Python virtual environment
- keep backend and OCR processes in separate terminals

---

## 19. Quick Command Reference

### Clone repo
```bash
git clone https://github.com/WSU-4110/FindMySpot.git
cd FindMySpot
```

### Create DB
```bash
psql -U postgres -c "CREATE DATABASE findmyspot;"
```

### Backend env
```bash
cd backend
cp .env.example .env
```

### Install backend deps
```bash
npm install
```

### Start backend
```bash
npm run dev
```

### Install Python deps
```bash
pip install -r ai-service/requirements.txt
```

### Run live OCR script
```bash
python app.py
```

### Run simulator
```bash
python ai-service/detection_service.py
```

### Serve frontend
```bash
python -m http.server 8080
```

### Run system tests
```bash
python test_system.py
```

---
