# FindMySpot

FindMySpot is a smart parking management system that uses AI-powered license plate recognition to automatically track your vehicle's location in parking garages. When you park, the system scans your plate and saves your spot. When you're ready to leave, just search your plate number and get directions right back to your car.

---

## Features

- **Automatic Plate Detection** — Cameras scan your license plate when you park (no manual input needed!)
- **Quick Vehicle Search** — Find your car by entering your license plate
- **Turn-by-Turn Directions** — Get guided back to your parking spot
- **Manual Check-in** — Backup option if the camera can't read your plate
- **Real-Time Occupancy** — See which spots are open or taken
- **Analytics Dashboard** — Track parking patterns and peak times (for facility managers)
- **Security Alerts** — Flag unauthorized or overstayed vehicles

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React Native (iOS & Android) |
| Backend | Node.js / Express |
| Database | PostgreSQL |
| AI/ML | YOLO (license plate detection) |
| OCR | EasyOCR / Custom model |
| Caching | Redis |
| Image Processing | OpenCV |

---

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
├── mobile-app/           # React Native app
│   ├── screens/          # App screens
│   ├── components/       # Reusable components
│   └── services/         # API calls
├── database/             # Database schemas & migrations
└── docs/                 # Documentation
```

---

## Quick Start

See the installation.md file for full setup instructions.

---

## Team

- Asmita Bhandari
- Mirza Sneha
- Jennifer Lopez
- Varun Kodikal
- Tristan Mejia
