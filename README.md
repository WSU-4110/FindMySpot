# FindMySpot
FindMySpot is a smart parking management system that uses AI-powered license plate recognition to automatically track your vehicle's location in parking garages. When you park, the system scans your plate and saves your spot. When you're ready to leave, just search your plate number and get directions right back to your car.

---
## Features
Automatic Plate Detection - Cameras scan your license plate when you park (no manual input needed!)
Quick Vehicle Search - Find your car by entering your license plate
Turn-by-Turn Directions - Get guided back to your parking spot
Manual Check-in - Backup option if the camera can't read your plate
Real-Time Occupancy - See which spots are open or taken
Analytics Dashboard - Track parking patterns and peak times (for facility managers)
Security Alerts - Flag unauthorized or overstayed vehicles

---

## Tech Stack 
Frontend: React Native (iOS & Android)  
Backend: Node.js / Express  
Database: PostgreSQL  
AI/ML: YOLO (for license plate detection)  
OCR: Tesseract / Custom model  
Caching: Redis  
Image Processing: OpenCV  

## Getting Started 
Prerequisites 
- Node.js (v16+)
- PostgreSQL (v13+)
- Python 3.8+ (for AI components)
- Redis
- Camera hardware (for deployment)
  
---
## Installation and Setup
Clone the repo  

bash   git clone https://github.com/yourusername/FindMySpot.git  
cd FindMySpot  

Install backend dependencies  

bash   cd backend  
npm install  

Install AI dependencies  

bash   cd ai-service  
pip install -r requirements.txt  

Set up the database  

bash   psql -U postgres  
CREATE DATABASE findmyspot;  
\q  
   
   # Run migrations
   npm run migrate

Configure environment variables

bash   cp .env.example .env

Start the backend  

bash   npm run dev  

Install mobile app dependencies  

bash   cd ../mobile-app
   npm install

Run the mobile app

bash   # For iOS
   npx react-native run-ios
   
   npx react-native run-android


---

## Project Structure
FindMySpot/
├── backend/              # Node.js API server  
│   ├── routes/          # API endpoints  
│   ├── controllers/     # Business logic  
│   ├── models/          # Database models  
│   └── middleware/      # Auth, validation, etc.  
├── ai-service/          # Python AI/ML service  
│   ├── models/          # YOLO model files  
│   ├── ocr/            # License plate OCR  
│   └── api/            # Flask API  
├── mobile-app/          # React Native app  
│   ├── screens/        # App screens  
│   ├── components/     # Reusable components  
│   └── services/       # API calls  
├── database/           # Database schemas & migrations  
└── docs/              # Documentation  

---

## Team 
Asmita Bhandari  
Mirza Sneha  
Jennifer Lopez   
Varun Kodikal   
Tristan Mejia  

---



