import flask
import cv2
import re
import time
import random
import requests
import json
import os
from collections import Counter, deque

import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# BUG FIX #1: Camera Hardcoding (HIGH) & BUG FIX #2: Random Floor/Lot Assignment (CRITICAL)
# PROBLEM #1: cv2.VideoCapture(0) was hardcoded, so all deployments used the same camera
#             even though camera_config.json had 10 cameras mapped to specific floors/lots.
# PROBLEM #2: assigned_floor/lot were randomly assigned (1-5) on each detection,
#             instead of being read from camera_config.json based on deployment location.
#
# SOLUTION: Load camera config at startup, get CAMERA_ID from environment variable,
#           extract floor/lot for that camera, and use them for all detections.

# Load camera configuration
def load_camera_config():
    """Load camera configuration from camera_config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'camera_config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Error: camera_config.json not found at {config_path}", flush=True)
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to parse camera_config.json", flush=True)
        return None

# Backend API configuration
BACKEND_API_URL = "http://localhost:3000/api/detection/record"

# Get camera ID from environment variable (e.g., 'set CAMERA_ID=3' for Floor 2, Lot 2)
# Default to 0 if not set
CAMERA_ID = int(os.environ.get('CAMERA_ID', '0'))
CAMERA_CONFIG = load_camera_config()

# FIXED: These are now loaded from camera_config.json instead of being randomly assigned
assigned_floor = None
assigned_lot = None
camera_name = None

last_printed_plate = ""
last_ocr_time = 0.0
ocr_interval_seconds = 0.9
blocked_words = {
    "PERSON",
    "CELLPHONE",
    "CELL",
    "PHONE",
    "ALABAMA",
    "ALASKA",
    "ARIZONA",
    "ARKANSAS",
    "CALIFORNIA",
    "COLORADO",
    "CONNECTICUT",
    "DELAWARE",
    "FLORIDA",
    "GEORGIA",
    "HAWAII",
    "IDAHO",
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "KANSAS",
    "KENTUCKY",
    "LOUISIANA",
    "MAINE",
    "MARYLAND",
    "MASSACHUSETTS",
    "MICHIGAN",
    "MINNESOTA",
    "MISSISSIPPI",
    "MISSOURI",
    "MONTANA",
    "NEBRASKA",
    "NEVADA",
    "NEWHAMPSHIRE",
    "NEWJERSEY",
    "NEWMEXICO",
    "NEWYORK",
    "NORTHCAROLINA",
    "NORTHDAKOTA",
    "OHIO",
    "OKLAHOMA",
    "OREGON",
    "PENNSYLVANIA",
    "RHODEISLAND",
    "SOUTHCAROLINA",
    "SOUTHDAKOTA",
    "TENNESSEE",
    "TEXAS",
    "UTAH",
    "VERMONT",
    "VIRGINIA",
    "WASHINGTON",
    "WESTVIRGINIA",
    "WISCONSIN",
    "WYOMING",
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}
show_candidate_boxes = True
show_debug_counts = True
max_candidates_per_tick = 3
min_confidence = 0.4
window_seconds = 5.0
cooldown_seconds = 10.0
min_votes = 3
window_start_time = 0.0
window_detections = []
final_plate = ""
finalized_at = 0.0


def send_parking_checkin(plate, floor, lot):
    """Send license plate detection to backend API"""
    try:
        payload = {
            "licensePlate": plate,
            "floor": floor,
            "lot": lot,
            "location": f"Floor {floor}, Lot {lot}",
            "confidence": 0.98,
            "cameraId": f"CAM_{CAMERA_ID}"  # FIXED: Dynamic camera ID instead of hardcoded
        }
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5)
        if response.status_code == 201:
            print(f"✓ Sent to backend: {plate} -> Floor {floor}, Lot {lot} (Camera {CAMERA_ID})", flush=True)
        else:
            print(f"✗ Backend error: {response.status_code}", flush=True)
    except Exception as e:
        print(f"✗ Failed to send to backend: {str(e)}", flush=True)


def find_plate_candidates(gray_frame):
    # Edge-based plate candidate detection
    blur = cv2.bilateralFilter(gray_frame, 11, 17, 17)
    edges = cv2.Canny(blur, 20, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = w / float(h)
        area = w * h
        if area < 800 or area > 90000:
            continue
        if aspect_ratio < 1.6 or aspect_ratio > 7.0:
            continue
        if h < 14:
            continue
        candidates.append((x, y, w, h))

    return candidates

# FIXED: Initialize camera configuration and floor/lot from config file
if CAMERA_CONFIG:
    cameras = CAMERA_CONFIG.get('cameras', [])
    matching_camera = None
    for cam in cameras:
        if cam.get('camera_id') == CAMERA_ID:
            matching_camera = cam
            break
    
    if matching_camera:
        assigned_floor = matching_camera.get('floor')
        assigned_lot = matching_camera.get('lot')
        camera_name = matching_camera.get('name', f'Camera {CAMERA_ID}')
        print(f"✓ Loaded camera config: {camera_name} -> Floor {assigned_floor}, Lot {assigned_lot}", flush=True)
    else:
        print(f"✗ Camera ID {CAMERA_ID} not found in camera_config.json. Available cameras: {[c.get('camera_id') for c in cameras]}", flush=True)
else:
    print("✗ Failed to load camera configuration. Using Camera 0.", flush=True)

# FIXED: Open camera using CAMERA_ID from environment instead of hardcoded 0
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(f"✗ Failed to open camera {CAMERA_ID}. Make sure it's connected and not in use.", flush=True)
    exit(1)

print(f"✓ Camera {CAMERA_ID} ({camera_name}) opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read from camera")
        break
    
    # Run OCR periodically to detect license plate text
    current_time = time.time()
    if current_time - last_ocr_time >= ocr_interval_seconds:
        last_ocr_time = current_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = find_plate_candidates(gray)
        if show_debug_counts:
            print(f"OCR tick: {len(candidates)} candidate(s)", flush=True)

        candidates = sorted(candidates, key=lambda c: c[2] * c[3], reverse=True)
        candidates = candidates[:max_candidates_per_tick]

        for x, y, w, h in candidates:
            roi = gray[y:y + h, x:x + w]
            ocr_results = reader.readtext(
                roi,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                text_threshold=0.3,
                low_text=0.15,
                detail=1,
            )

            for _, text, confidence in ocr_results:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if cleaned in blocked_words:
                    continue
                if not re.fullmatch(r'[A-Z0-9]{4,8}', cleaned):
                    continue
                if confidence < min_confidence:
                    continue
                if window_start_time == 0.0:
                    window_start_time = current_time
                window_detections.append(cleaned)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    cleaned,
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if show_candidate_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 1)

    if window_start_time and (current_time - window_start_time >= window_seconds):
        if window_detections:
            most_common = Counter(window_detections).most_common(1)
            if most_common:
                consensus_plate, hits = most_common[0]
                cooldown_active = (
                    consensus_plate == last_printed_plate
                    and (current_time - finalized_at) < cooldown_seconds
                )
                if hits >= min_votes and not cooldown_active:
                    last_printed_plate = consensus_plate
                    final_plate = consensus_plate
                    finalized_at = current_time
                    
                    # FIXED: Use floor and lot from camera configuration (loaded at startup)
                    # REMOVED: random.randint(1, 5) calls that caused wrong floor/lot assignments
                    # These values are now deterministic and based on camera location
                    
                    timestamp = time.strftime("%H:%M:%S")
                    print(
                        f"[{timestamp}] License plate finalized: {consensus_plate} | Floor {assigned_floor} | Lot {assigned_lot}",
                        flush=True,
                    )
                    
                    # Send parking check-in to backend API
                    send_parking_checkin(consensus_plate, assigned_floor, assigned_lot)
        window_start_time = 0.0
        window_detections = []

    if final_plate:
        cv2.putText(
            frame,
            f"FINAL: {final_plate}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
    
    # Display floor and lot information
    if assigned_floor is not None and assigned_lot is not None:
        cv2.putText(
            frame,
            f"Floor {assigned_floor} | Lot {assigned_lot}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Waiting for vehicle detection...",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")