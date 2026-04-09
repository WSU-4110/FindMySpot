import os
import cv2
import time
import re
import logging
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import easyocr

# Import your custom database classes
from auth_database import AuthDatabase
from vehicle_database import VehicleDatabase

# ============================================================================
# INITIALIZE APP & LOGGING
# ============================================================================

app = Flask(__name__)
auth_db = AuthDatabase(
    dbname="findmyspot",
    user="postgres",
    password=os.getenv('DB_PASSWORD'),
    host="localhost"
)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
auth_db = AuthDatabase(password=DB_PASSWORD)
vehicle_db = VehicleDatabase(password=DB_PASSWORD)

# ============================================================================
# CAMERA & OCR SETUP
# ============================================================================

print("Initializing camera...")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera index 0 failed. Trying index 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.5) 

print("Loading OCR model...")
reader = easyocr.Reader(['en'])
print("OCR ready!")

# Plate Detection State
last_detected_plate = None
last_detection_time = 0
plate_candidates = {}
CONFIRM_COUNT = 3

# ============================================================================
# HELPERS
# ============================================================================

def is_license_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
    blocklist = {'MICHIGAN', 'LAKES', 'GREATLAKES', 'PUREMICHIGAN', 'WATERWONDERLAND', 'REGISTRATION'}
    
    if text in blocklist or len(text) < 4:
        return False
    
    has_numbers = bool(re.search(r'[0-9]', text))
    return has_numbers

def get_user_from_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    try:
        token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
        return auth_db.verify_jwt_token(token)
    except Exception as e:
        logger.error(f"Token error: {e}")
        return None

def generate_frames():
    global last_detected_plate, last_detection_time, plate_candidates
    frame_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % 15 == 0:
            results = reader.readtext(frame)
            for (bbox, text, confidence) in results:
                if confidence > 0.4 and is_license_plate(text):
                    clean = text.strip().upper().replace(' ', '')
                    plate_candidates[clean] = plate_candidates.get(clean, 0) + 1
                    
                    if plate_candidates[clean] >= CONFIRM_COUNT:
                        last_detected_plate = clean
                        last_detection_time = time.time()
                        plate_candidates.clear()

        if frame_count % 60 == 0:
            plate_candidates.clear()

        if last_detected_plate and (time.time() - last_detection_time < 5):
            cv2.putText(frame, f"PLATE: {last_detected_plate}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/')
def serve_index():
    return send_from_directory('mobile-app', 'index.html')

@app.route('/api/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    result = auth_db.authenticate_user(data['email'], data['password'])
    
    if result:
        return jsonify({
            "token": result['token'],
            "user": result['user']
        }), 200
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.json
    user_id = auth_db.create_user(
        email=data['email'],
        password=data['password'],
        username=data['username']
    )
    
    if user_id:
        return jsonify({"message": "User created", "id": user_id}), 201
    return jsonify({"error": "User already exists"}), 400
   

@app.route('/api/vehicle/checkin', methods=['POST'])
def checkin_vehicle():
    user_id = get_user_from_token()
    if not user_id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    plate = data.get('plate', '').strip().upper()
    floor = data.get('floor', '')
    spot = data.get('spot', '')

    vehicle = vehicle_db.get_vehicle_by_plate(plate, user_id)
    if vehicle:
        vehicle_db.update_vehicle(vehicle_id=vehicle['id'], floor=floor, spot=spot)
    else:
        vehicle_db.add_vehicle(user_id=user_id, license_plate=plate, floor=floor, spot=spot)
    
    return jsonify({'success': True, 'message': f'Checked in {plate}'}), 200

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)