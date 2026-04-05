"""
app_api.py - Flask API for FindMySpot
"""

import cv2
import time
import re
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from auth_database import AuthDatabase
from vehicle_database import VehicleDatabase
import easyocr

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE
# ============================================================================

DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
auth_db = AuthDatabase(password=DB_PASSWORD)
vehicle_db = VehicleDatabase(password=DB_PASSWORD)

# ============================================================================
# CAMERA SETUP
# ============================================================================

print("Initializing camera...")

# 1. Switch to CAP_DSHOW for much better stability on Windows
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 2. Add a fallback to Index 1 (in case Windows re-assigned your webcam)
if not camera.isOpened():
    print("Camera index 0 failed. Trying index 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 3. Set properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 4. CRITICAL: Remove the 30-frame loop and replace with a short pause
time.sleep(1.5) 

if camera.isOpened():
    # Test one single read to confirm hardware response
    success, _ = camera.read()
    if success:
        print("Camera ready and responding!")
    else:
        print("Camera opened but failed to return a frame. Check privacy shutter.")
else:
    print("CRITICAL: No camera hardware found.")

# ============================================================================
# OCR SETUP
# ============================================================================

print("Loading OCR model...")
reader = easyocr.Reader(['en'])
print("OCR ready!")

# ============================================================================
# PLATE DETECTION STATE
# ============================================================================

last_detected_plate = None
last_detection_time = 0
plate_candidates = {}
CONFIRM_COUNT = 3

# ============================================================================
# PLATE DETECTION HELPERS
# ============================================================================

def is_license_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())

    # 1. Block everything that isn't the actual plate number
    blocklist = {
        'MICHIGAN', 'LAKES', 'GREATLAKES', 'SPLENDOR', 'MONTH', 
        'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'PUREMICHIGAN',
        'WATERWONDERLAND', 'REGISTRATION', 'GREAT'
    }
    
    if text in blocklist or len(text) < 4:
        return False

    # 2. Add a 'Heuristic': Plate numbers usually have a mix of letters/numbers
    # This helps ignore things like "LAKES" (all letters) 
    # unless it's a vanity plate.
    has_letters = bool(re.search(r'[A-Z]', text))
    has_numbers = bool(re.search(r'[0-9]', text))
    
    # If it's just words, it's likely noise.
    # Let's require at least one number for this specific plate.
    if not has_numbers:
        return False

    return True
    


def generate_frames():
    global last_detected_plate, last_detection_time, plate_candidates

    frame_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to read camera frame")
            time.sleep(0.1)
            continue

        frame_count += 1

        # Run OCR every 15 frames to avoid overloading CPU
        if frame_count % 15 == 0:
            results = reader.readtext(frame)

            for (bbox, text, confidence) in results:
                logger.info(f"OCR: '{text}' conf={confidence:.2f}")

                if confidence > 0.4 and is_license_plate(text):
                    clean = text.strip().upper().replace(' ', '')
                    plate_candidates[clean] = plate_candidates.get(clean, 0) + 1
                    logger.info(f"Candidate '{clean}' count={plate_candidates[clean]}")

                    if plate_candidates[clean] >= CONFIRM_COUNT:
                        last_detected_plate = clean
                        last_detection_time = time.time()
                        plate_candidates.clear()
                        logger.info(f"*** CONFIRMED PLATE: {clean} ***")

            # Reset candidates every 60 frames to avoid stale data
            if frame_count % 60 == 0:
                plate_candidates.clear()

        # Draw overlay on every frame (outside OCR block)
        if last_detected_plate and (time.time() - last_detection_time < 5):
            cv2.putText(
                frame,
                f"PLATE: {last_detected_plate}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============================================================================
# VIDEO / PLATE ENDPOINTS
# ============================================================================

@app.route('/api/video')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/plate', methods=['GET'])
def get_plate():
    """Return last detected plate if detected within 10 seconds"""
    if last_detected_plate and (time.time() - last_detection_time < 10):
        return jsonify({
            'plate': last_detected_plate,
            'detected_at': last_detection_time
        })
    return jsonify({'plate': None})


@app.route('/api/reset', methods=['POST'])
def reset_detection():
    """Reset plate detection state"""
    global last_detected_plate, last_detection_time
    last_detected_plate = None
    last_detection_time = 0
    plate_candidates.clear()
    return jsonify({'success': True, 'message': 'Detection reset successfully'})


@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200


@app.route('/api/camera-test')
def camera_test():
    success, frame = camera.read()
    return jsonify({
        'camera_opened': camera.isOpened(),
        'frame_read': success,
        'frame_shape': str(frame.shape) if success else None,
        'mean_brightness': float(frame.mean()) if success else None
    })


@app.route('/webcam')
def webcam_page():
    return send_from_directory(r'C:\Users\varun\FindMySpot\mobile-app', 'index.html')

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not name or not email or not password:
            return jsonify({'success': False, 'error': 'Please provide name, email, and password'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400

        username = name.lower().replace(' ', '_')
        user_id = auth_db.create_user(email=email, password=password, username=username)

        if not user_id:
            return jsonify({'success': False, 'error': 'Account already exists for this email'}), 400

        token = auth_db.generate_jwt_token(user_id)
        user = auth_db.get_user_by_id(user_id)
        logger.info(f"New user signed up: {email}")

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'username': user['username'],
                'name': name
            }
        }), 201

    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({'success': False, 'error': 'Signup failed. Please try again.'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'error': 'Please enter your email and password'}), 400

        if not auth_db.verify_password(email, password):
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

        user = auth_db.get_user_by_email(email)
        if not user:
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

        token = auth_db.generate_jwt_token(user['id'])
        auth_db.update_last_login(user['id'])
        logger.info(f"User logged in: {email}")

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'username': user['username']
            }
        }), 200

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed. Please try again.'}), 500

# ============================================================================
# VEHICLE ENDPOINTS
# ============================================================================

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


@app.route('/api/vehicle/checkin', methods=['POST'])
def checkin_vehicle():
    try:
        user_id = get_user_from_token()
        if not user_id:
            return jsonify({'success': False, 'error': 'Unauthorized. Please log in.'}), 401

        data = request.get_json()
        plate = data.get('plate', '').strip().upper().replace(' ', '')
        floor = data.get('floor', '').strip()
        spot = data.get('spot', '').strip().upper()

        if not plate or not floor or not spot:
            return jsonify({'success': False, 'error': 'Please provide plate, floor, and spot'}), 400

        vehicle = vehicle_db.get_vehicle_by_plate(plate, user_id)
        if vehicle:
            vehicle_db.update_vehicle(vehicle_id=vehicle['id'], floor=floor, spot=spot)
        else:
            vehicle_id = vehicle_db.add_vehicle(
                user_id=user_id, license_plate=plate,
                floor=floor, spot=spot, is_primary=True
            )
            if not vehicle_id:
                return jsonify({'success': False, 'error': 'Failed to add vehicle'}), 500

        logger.info(f"Checked in {plate} for user {user_id}")
        return jsonify({'success': True, 'message': f'Checked in {plate} successfully'}), 200

    except Exception as e:
        logger.error(f"Checkin error: {e}")
        return jsonify({'success': False, 'error': 'Check-in failed. Please try again.'}), 500


@app.route('/api/vehicle/find', methods=['POST'])
def find_vehicle():
    try:
        user_id = get_user_from_token()
        if not user_id:
            return jsonify({'success': False, 'error': 'Unauthorized. Please log in.'}), 401

        data = request.get_json()
        plate = data.get('plate', '').strip().upper().replace(' ', '')

        if not plate:
            return jsonify({'success': False, 'error': 'Please enter a license plate number'}), 400

        vehicle = vehicle_db.get_vehicle_by_plate(plate, user_id)
        if not vehicle:
            return jsonify({'success': False, 'error': 'Vehicle not found'}), 404

        if not vehicle.get('floor') or not vehicle.get('spot'):
            return jsonify({'success': False, 'error': 'No parking location found for this vehicle'}), 404

        logger.info(f"User {user_id} found vehicle {plate}")
        return jsonify({
            'success': True,
            'floor': vehicle['floor'],
            'spot': vehicle['spot'],
            'plate': vehicle['license_plate'],
            'parkedSince': vehicle['updated_at'].isoformat() if vehicle.get('updated_at') else datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Find vehicle error: {e}")
        return jsonify({'success': False, 'error': 'Lookup failed. Please try again.'}), 500

# ============================================================================
# HEALTH / ERROR HANDLERS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'FindMySpot API is running'}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'FindMySpot API',
        'version': '2.0',
        'endpoints': {
            'auth': {'signup': 'POST /api/auth/signup', 'login': 'POST /api/auth/login'},
            'vehicle': {'checkin': 'POST /api/vehicle/checkin', 'find': 'POST /api/vehicle/find'},
            'camera': {'video': 'GET /api/video', 'plate': 'GET /api/plate', 'reset': 'POST /api/reset'}
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("FindMySpot API Server v2.0")
    print("=" * 70)
    print("  POST http://localhost:5000/api/auth/signup")
    print("  POST http://localhost:5000/api/auth/login")
    print("  POST http://localhost:5000/api/vehicle/checkin")
    print("  POST http://localhost:5000/api/vehicle/find")
    print("  GET  http://localhost:5000/api/video")
    print("  GET  http://localhost:5000/webcam")
    print("=" * 70)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )