# test_app.py
import cv2
import re
import time
import random
import requests
from collections import Counter

import easyocr

# -------------------------
# Constants & Config
# -------------------------
BACKEND_API_URL = "http://localhost:3000/api/detection/record"
ocr_interval_seconds = 0.9
blocked_words = {
    "PERSON", "CELLPHONE", "CELL", "PHONE",  # ...etc, keep your full list
}
min_confidence = 0.4
max_candidates_per_tick = 3
window_seconds = 5.0
cooldown_seconds = 10.0
min_votes = 3

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# -------------------------
# Helper functions
# -------------------------

def send_parking_checkin(plate, floor, lot):
    """Send license plate detection to backend API"""
    try:
        payload = {
            "licensePlate": plate,
            "floor": floor,
            "lot": lot,
            "location": f"Floor {floor}, Lot {lot}",
            "confidence": 0.98,
            "cameraId": "CAM_LOCAL"
        }
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5)
        return response.status_code
    except Exception as e:
        return str(e)


def clean_plate_text(text):
    """Cleans OCR text to match license plate rules"""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned


def is_valid_plate(text, confidence):
    """Returns True if text passes blocked words, regex, and confidence"""
    if text in blocked_words:
        return False
    if not re.fullmatch(r'[A-Z0-9]{4,8}', text):
        return False
    if confidence < min_confidence:
        return False
    return True


def find_plate_candidates(gray_frame):
    """Return a list of bounding boxes (x, y, w, h) for candidate plates"""
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


def ocr_plate(roi):
    """Run EasyOCR on a region of interest and return list of valid plates"""
    results = []
    ocr_results = reader.readtext(
        roi,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        text_threshold=0.3,
        low_text=0.15,
        detail=1,
    )
    for _, text, confidence in ocr_results:
        cleaned = clean_plate_text(text)
        if is_valid_plate(cleaned, confidence):
            results.append((cleaned, confidence))
    return results


# -------------------------
# Main loop (only runs if script executed directly)
# -------------------------
if __name__ == "__main__":
    assigned_floor = None
    assigned_lot = None
    last_printed_plate = ""
    final_plate = ""
    finalized_at = 0.0
    window_start_time = 0.0
    window_detections = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        # OCR and detection logic goes here, same as before...
        # (Use the helper functions instead of inline code)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")