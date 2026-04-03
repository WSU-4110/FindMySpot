# FindMySpot

FindMySpot is an intelligent parking management system designed to help users locate their vehicles within parking structures using AI-powered license plate recognition. The system automatically scans and records a vehicle’s location when it is parked, allowing users to quickly retrieve their parking spot and directions by searching their plate number. The project integrates a Node.js/Express backend, a PostgreSQL database, a Python-based computer vision pipeline, and a lightweight browser/mobile-style frontend to deliver a seamless and efficient parking experience.

## Overview

When a vehicle enters or moves through a monitored parking area, the camera pipeline reads the plate, associates it with a configured camera location, and reports the detection to the backend. Users can then:

- register and log in
- register one or more vehicles
- search for a parked vehicle by license plate
- view parking history and notifications
- request directions to the last detected parking location
- fall back to manual workflows when automatic recognition is unavailable

The backend also exposes parking statistics, occupancy-related endpoints, security scanning endpoints, and notification APIs.

## Repository Status Notes

This repository includes several implementation details to support development and testing:

- The backend is implemented with **Express** and the committed dependency list includes `express`, `cors`, `dotenv`, and `pg`.
- The checked-in `mobile-app/` folder is currently a **static HTML/CSS frontend** rather than a React Native codebase.
- The root-level `app.py` is the active camera/OCR script in the repository. It uses **OpenCV** and **EasyOCR** and posts detection data to the backend.
- The `ai-service/` directory contains a reusable Python integration service (`detection_service.py`) plus its own requirements file and tests.
- The repository includes a tracked `backend/.env` file and `backend/node_modules/`, which is not typical for production repositories. For portability and security, local setups should still rely on `backend/.env.example` and a fresh dependency install.

## Core Features

### User and account features
- User registration and login
- Token-based profile access
- Vehicle registration and management
- Vehicle history retrieval

### Parking features
- Parking spot lookup
- Available and occupied spot queries
- Parking statistics and usage reporting
- Vehicle locate endpoint
- Directions endpoint
- Auto-checkout timer recovery after server restart

### Detection and AI features
- License plate detection reporting endpoint
- Detection history, count, and recent detections
- OCR-based plate extraction in the root Python pipeline
- Camera-to-floor/lot mapping using `camera_config.json`
- Simulated detection generation through the AI integration service

### Notification and security features
- Notification retrieval and unread counts
- Mark-one and mark-all-as-read flows
- Notification deletion
- Security flag retrieval
- Security scan endpoint for suspicious or overstayed vehicles

## High-Level Architecture

```
Camera / CV Pipeline (Python)
        |
        v
Detection POST -> Node.js / Express API -> PostgreSQL
        |                    |
        |                    +-> business logic/controllers/models
        |
        +-> static frontend pages consume backend endpoints
```

### Main runtime pieces

#### 1. Backend API (`backend/`)
The backend is the system’s core service. It:
- initializes the database
- registers all API routes
- exposes health and root documentation endpoints
- restores active parking-session auto-checkout timers on startup
- handles authentication, parking, vehicles, detections, and notifications

#### 2. Database (`database/migrations/`)
The database schema is created through three migration files:
- `001_init.sql`
- `002_license_plate_detection.sql`
- `003_security_flags.sql`

#### 3. Computer vision / AI side
There are two Python-related pieces:

- **Root `app.py`**: live camera/OCR script using `cv2`, `easyocr`, and `camera_config.json`
- **`ai-service/detection_service.py`**: helper/integration service for posting detections to the backend and simulating camera events

#### 4. Frontend (`mobile-app/`)
Despite the folder name, this part of the repo currently contains browser-deliverable UI pages such as:
- `index.html`
- `login.html`
- `signup.html`
- `vehicles.html`
- `vehicles-new.html`
- `notifications.html`
- `history.html`
- `terms.html`
- `style.css`

## Project Structure

```

FindMySpot/
├── backend/              # Node.js API server
│   ├── routes/           # API endpoints
│   ├── controllers/      # Business logic
│   ├── models/           # Database models
│   └── middleware/       # Auth, validation, etc.
├── ai-service/           # Python AI/ML service
│   ├── models/           # YOLO model files
│   ├── ocr/              # License plate OCR
│   └── api/              # Flask API
├── mobile-app/           # Static HTML/CSS/JS frontend
│   ├── screens/          # App screens
│   ├── components/       # Reusable components
│   └── services/         # API calls
├── database/             # Database schemas & migrations
└── docs/                 # Documentation
```


## Backend API Summary

The backend root endpoint documents the API groups below.

### Authentication
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/profile/:token`

### Vehicles
- `POST /api/vehicles/register/:token`
- `GET /api/vehicles/:token`
- `GET /api/vehicles/:token/history`
- `PUT /api/vehicles/:token/:vehicleId`
- `DELETE /api/vehicles/:token/:vehicleId`

### Detection
- `POST /api/detection/record`
- `GET /api/detection/history/:licensePlate`
- `GET /api/detection/count/:licensePlate`
- `GET /api/detection/recent`

### Notifications
- `GET /api/notifications/:token`
- `GET /api/notifications/unread/:token`
- `GET /api/notifications/count/:token`
- `PUT /api/notifications/:token/:notificationId/read`
- `PUT /api/notifications/:token/mark-all-read`
- `DELETE /api/notifications/:token/:notificationId`

### Parking
- `GET /api/parking/spots`
- `GET /api/parking/spots/available`
- `GET /api/parking/spots/occupied`
- `GET /api/parking/stats`
- `GET /api/parking/locate/:plate`
- `GET /api/parking/directions/:plate`
- `GET /api/parking/reports/usage?hoursBack=24`
- `GET /api/parking/security/flags`
- `POST /api/parking/security/scan?maxDurationHours=24`

## How Detection Works

### Live camera path
1. `app.py` opens the configured camera using `CAMERA_ID`.
2. The script loads `camera_config.json`.
3. The selected camera is mapped to a fixed floor and lot.
4. Frames are processed with OpenCV.
5. OCR is performed with EasyOCR.
6. A cleaned plate string and metadata are posted to `http://localhost:3000/api/detection/record`.

### Simulated path
`ai-service/detection_service.py` can simulate detections for test plates and submit them to the same backend detection endpoint.

## Camera Configuration

The repository includes `camera_config.json` with camera IDs `0` through `9`. Each camera is mapped to a specific floor and lot, covering floors 1 through 5 with two lot positions per floor. This lets the OCR pipeline assign real parking metadata based on which camera is running, instead of using random floor/lot values.

## Environment Configuration
The backend `.env.example` defines:
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=findmyspot
PORT=3000

Recommended additions for local development:
NODE_ENV=development

For camera-specific local runs, you can also set a Python-side environment variable before launching `app.py`:
CAMERA_ID=0
On Windows PowerShell:
$env:CAMERA_ID=0

## Local Development Workflow
### Backend
cd backend
npm install
npm run dev

### Database
- Create PostgreSQL database `findmyspot`
- Run repository migrations
- Confirm tables were created successfully

### Python CV pipeline
pip install -r ai-service/requirements.txt
python app.py

### Static frontend
From the repo root or `mobile-app/` directory, serve files locally:
python -m http.server 8080
Then open the appropriate page in a browser.

### End-to-end verification
python test_system.py

## Testing
The repository includes:
- `test_system.py` for API-level integration/system checks
- `ai-service/test_detection_service.py` for detection service testing

The system test script is configured to use:
- API base URL: `http://localhost:3000`
- user registration flow
- vehicle retrieval flow
- detection recording flow
- notification-related flows

Before running tests, make sure:
- PostgreSQL is running
- the backend server is running on port 3000
- required schema exists

## Team
- Asmita Bhandari
- Mirza Sneha
- Jennifer Lopez
- Varun Kodikal
- Tristan Mejia




