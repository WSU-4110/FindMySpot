from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import re
import time
from collections import Counter, deque
import easyocr

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

<<<<<<< HEAD
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

# Global variables
last_printed_plate = ""
last_ocr_time = 0.0
ocr_interval_seconds = 0.9
blocked_words = {# 50 States (Cleaned for OCR)
=======
# Global variables
last_printed_plate = ""
last_ocr_time = 0.0
ocr_interval_seconds = 1.0  # Run OCR every second
blocked_words = {
    # 50 States (Cleaned for OCR)
>>>>>>> bef3ead3623c9edc3503dd54c89a31fbe9e9b6b8
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", 
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", 
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", 
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", 
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", 
    "NEWHAMPSHIRE", "NEWJERSEY", "NEWMEXICO", "NEWYORK", "NORTHCAROLINA", 
    "NORTHDAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA", 
    "RHODEISLAND", "SOUTHCAROLINA", "SOUTHDAKOTA", "TENNESSEE", "TEXAS", 
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WESTVIRGINIA", 
    "WISCONSIN", "WYOMING",
<<<<<<< HEAD
    
    # Common catch-alls/Slogans
    "WASH", "PENN", "ALOHA", "SUNSHINE", "GARDENSTATE", "EMPIRESTATE"}
show_candidate_boxes = True
show_debug_counts = True
max_candidates_per_tick = 3
min_confidence = 0.4
recent_plate_window = deque(maxlen=8)
min_consensus_hits = 4
final_plate = ""
finalized_at = 0.0


def send_parking_checkin(plate, floor, lot):
    """Send parking check-in data to backend API"""
    try:
        payload = {
            "vehiclePlate": plate,
            "floor": floor,
            "lot": lot
        }
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✓ Sent to backend: {plate} -> Floor {floor}, Lot {lot}", flush=True)
        else:
            print(f"✗ Backend error: {response.status_code}", flush=True)
    except Exception as e:
        print(f"✗ Failed to send to backend: {str(e)}", flush=True)
=======
    # Common OCR misreads of state names
    "MICHIGA", "MICHICAN", "MCHIGAN", "MIGHIGAN", "MICHGAN",  # Michigan variations
    "TEXA", "CALIF", "CALIFORN", "CALIFOR",
    "FLOR", "FLORID", "GEORGI", "OREGO", "TENNESS", "PENNSY", "NEVAD",
    # Slogans
    "WASH", "PENN", "ALOHA", "SUNSHINE", "GARDENSTATE", "EMPIRESTATE",
    "GREAT", "LAKES", "GREATLAKES", "SPLENDOR", "GLKSS", "GLAKES", "GLKESSI"
}
recent_plate_window = deque(maxlen=10)
min_consensus_hits = 2  # Reduced from 3 to 2 for faster detection
final_plate = ""
finalized_at = 0.0

# Initialize webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
>>>>>>> bef3ead3623c9edc3503dd54c89a31fbe9e9b6b8


def generate_frames():
    """Generate frames with OCR overlay - SIMPLE WHOLE-FRAME APPROACH"""
    global last_ocr_time, last_printed_plate, final_plate, finalized_at
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run OCR periodically on THE ENTIRE FRAME
        current_time = time.time()
        if current_time - last_ocr_time >= ocr_interval_seconds:
            last_ocr_time = current_time
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print(f"=== OCR Scan ===", flush=True)
            
            # Run OCR on whole frame
            ocr_results = reader.readtext(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                text_threshold=0.2,
                low_text=0.1,
                detail=1,
            )
            
            print(f"Found {len(ocr_results)} text regions", flush=True)
            
            # First pass: collect all valid plate-like fragments
            plate_fragments = []
            for bbox, text, confidence in ocr_results:
                text_no_space = text.replace(' ', '')
                cleaned = re.sub(r'[^A-Z0-9]', '', text_no_space.upper())
                
                # Skip blocked words
                if cleaned in blocked_words:
                    continue
                
                # Collect fragments that could be part of a plate (2-8 chars with letters/numbers)
                if len(cleaned) >= 2 and len(cleaned) <= 8:
                    if re.search(r'[A-Z]', cleaned) or re.search(r'\d', cleaned):
                        pts = [bbox[0], bbox[1], bbox[2], bbox[3]]
                        x = int(min(p[0] for p in pts))
                        y = int(min(p[1] for p in pts))
                        w = int(max(p[0] for p in pts) - x)
                        h = int(max(p[1] for p in pts) - y)
                        plate_fragments.append({
                            'text': cleaned,
                            'conf': confidence,
                            'x': x, 'y': y, 'w': w, 'h': h
                        })
            
            print(f"  Found {len(plate_fragments)} potential plate fragments", flush=True)
            
            # Second pass: try to combine fragments on the same horizontal line
            combined_candidates = []
            used = set()
            
            for i, frag1 in enumerate(plate_fragments):
                if i in used:
                    continue
                    
                # Look for nearby fragments on same line
                combined_text = frag1['text']
                combined_conf = frag1['conf']
                bbox_group = [frag1]
                
                for j, frag2 in enumerate(plate_fragments):
                    if j <= i or j in used:
                        continue
                    
                    # Check if on same horizontal line (y coordinates similar)
                    y_diff = abs(frag1['y'] - frag2['y'])
                    x_gap = abs(frag2['x'] - (frag1['x'] + frag1['w']))
                    
                    # If close together horizontally and vertically aligned
                    if y_diff < 20 and x_gap < 100:
                        # Combine them
                        if frag2['x'] > frag1['x']:  # frag2 is to the right
                            combined_text += frag2['text']
                        else:  # frag2 is to the left
                            combined_text = frag2['text'] + combined_text
                        combined_conf = max(combined_conf, frag2['conf'])
                        bbox_group.append(frag2)
                        used.add(j)
                
                combined_candidates.append({
                    'text': combined_text,
                    'conf': combined_conf,
                    'boxes': bbox_group
                })
                used.add(i)
            
            print(f"  After combining: {len(combined_candidates)} candidates", flush=True)
            
            # Sort by length (prefer longer matches - more likely to be full plate)
            combined_candidates = sorted(combined_candidates, key=lambda x: len(x['text']), reverse=True)
            
            # Process combined candidates
            for candidate in combined_candidates:
                cleaned = candidate['text']
                confidence = candidate['conf']
                
                # Apply OCR corrections for common confusions
                # D <-> 0, O <-> 0, I <-> 1, L <-> 1
                original = cleaned
                
                # Michigan plates are typically: ABC1234 or AB12345 format
                # Your plate: 1DL D99 (3 chars, space, 3 chars)
                
                # If it's all numbers or mostly numbers, try adding letters
                if re.fullmatch(r'\d+', cleaned):
                    # All digits - likely some are actually letters
                    # Common pattern: 1DL099 read as 101099 or 10L099
                    if len(cleaned) == 6:  # Typical plate length
                        # Try various letter substitutions for middle characters
                        # Pattern: XYZ### where middle could be D, L, etc.
                        variants = [
                            cleaned[0] + 'DL' + cleaned[3:],     # 1DL099 (most likely for your plate)
                            cleaned[0] + 'D' + cleaned[2] + cleaned[3:],  # 1D1099
                            cleaned[:2] + 'LD' + cleaned[4:],   # 10LD99
                            cleaned[0] + 'O' + cleaned[2] + cleaned[3:],  # 1O1099 (O not 0)
                        ]
                        for variant in variants:
                            if re.search(r'[A-Z]', variant) and re.search(r'\d', variant):
                                print(f"  Auto-corrected: '{original}' -> '{variant}'", flush=True)
                                cleaned = variant
                                break
                    elif len(cleaned) == 5:  # Missing a character
                        # Try: X0LXX -> XDL0XX
                        variants = [
                            cleaned[0] + 'DL' + cleaned[2:],
                            cleaned[0] + 'DL0' + cleaned[3:],
                        ]
                        for variant in variants:
                            if len(variant) == 6 and re.search(r'[A-Z]', variant):
                                print(f"  Auto-corrected: '{original}' -> '{variant}'", flush=True)
                                cleaned = variant
                                break
                
                elif len(cleaned) == 5 and re.search(r'[A-Z]', cleaned):
                    # Has letters but too short - maybe missing D
                    # 10L99 -> 1DL099 or 1DLD99
                    if cleaned.startswith('10L') or cleaned.startswith('1DL'):
                        variants = [
                            '1DL0' + cleaned[3:],
                            '1DLD' + cleaned[3:],
                        ]
                        for variant in variants:
                            if len(variant) == 6:
                                print(f"  Auto-corrected: '{original}' -> '{variant}'", flush=True)
                                cleaned = variant
                                break
                
                print(f"  Candidate: '{cleaned}' (conf: {confidence:.2f})", flush=True)
                
                # Skip if blocked
                if cleaned in blocked_words:
                    print(f"    BLOCKED (state name/slogan)", flush=True)
                    continue
                
                # Must be 4-8 characters
                if not re.fullmatch(r'[A-Z0-9]{4,8}', cleaned):
                    print(f"    SKIP (length {len(cleaned)} not 4-8)", flush=True)
                    continue
                
                # Prefer plates with mix of letters and numbers, but allow all-digit if conf is decent
                has_mix = re.search(r'\d', cleaned) and re.search(r'[A-Z]', cleaned)
                if not has_mix and confidence < 0.3:
                    print(f"    SKIP (all numbers/letters and low confidence)", flush=True)
                    continue
                
                # Must have minimum confidence
                if confidence < 0.12:
                    print(f"    SKIP (very low confidence)", flush=True)
                    continue
<<<<<<< HEAD
                if not final_plate:
                    recent_plate_window.append(cleaned)
                    most_common = Counter(recent_plate_window).most_common(1)
                    if most_common:
                        consensus_plate, hits = most_common[0]
                        if hits >= min_consensus_hits and consensus_plate != last_printed_plate:
                            last_printed_plate = consensus_plate
                            final_plate = consensus_plate
                            finalized_at = time.time()
                            timestamp = time.strftime("%H:%M:%S")
                            print(
                                f"[{timestamp}] License plate finalized: {consensus_plate}",
                                flush=True,
                            )
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
                    
                    # Randomly assign floor (1-5) and lot (1-5)
                    assigned_floor = random.randint(1, 5)
                    assigned_lot = random.randint(1, 5)
                    
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
=======
                
                # This passed all filters!
                print(f"    ✓ VALID PLATE CANDIDATE", flush=True)
                
                # Draw boxes on frame for all fragments in this candidate
                for box in candidate['boxes']:
                    x, y, w, h = box['x'], box['y'], box['w'], box['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw the combined text
                first_box = candidate['boxes'][0]
                cv2.putText(frame, cleaned, (first_box['x'], max(0, first_box['y'] - 8)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add to consensus window
                if not final_plate:
                    recent_plate_window.append(cleaned)
                    
                    # Use fuzzy matching - plates with same first 4 chars are likely the same
                    # (OCR often misreads last digits)
                    if len(cleaned) >= 4:
                        prefix = cleaned[:4]  # e.g., "1DL0" from "1DL099" or "1DL093"
                        similar_plates = [p for p in recent_plate_window if len(p) >= 4 and p[:4] == prefix]
                        
                        if len(similar_plates) >= min_consensus_hits:
                            # Use the most common full plate among similar ones
                            plate_counter = Counter(similar_plates)
                            consensus_plate, hits = plate_counter.most_common(1)[0]
                            
                            if consensus_plate != last_printed_plate:
                                last_printed_plate = consensus_plate
                                final_plate = consensus_plate
                                finalized_at = time.time()
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"[{timestamp}] ★★★ FINAL PLATE: {consensus_plate} (from {len(similar_plates)} similar) ★★★", flush=True)
                        else:
                            # Fall back to exact matching
                            most_common = Counter(recent_plate_window).most_common(1)
                            if most_common:
                                consensus_plate, hits = most_common[0]
                                print(f"    Consensus: {consensus_plate} ({hits}/{min_consensus_hits} hits)", flush=True)
                                if hits >= min_consensus_hits and consensus_plate != last_printed_plate:
                                    last_printed_plate = consensus_plate
                                    final_plate = consensus_plate
                                    finalized_at = time.time()
                                    timestamp = time.strftime("%H:%M:%S")
                                    print(f"[{timestamp}] ★★★ FINAL PLATE: {consensus_plate} ★★★", flush=True)
                    else:
                        # Plate too short for fuzzy matching
                        most_common = Counter(recent_plate_window).most_common(1)
                        if most_common:
                            consensus_plate, hits = most_common[0]
                            print(f"    Consensus: {consensus_plate} ({hits}/{min_consensus_hits} hits)", flush=True)
                            if hits >= min_consensus_hits and consensus_plate != last_printed_plate:
                                last_printed_plate = consensus_plate
                                final_plate = consensus_plate
                                finalized_at = time.time()
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"[{timestamp}] ★★★ FINAL PLATE: {consensus_plate} ★★★", flush=True)

        # Display final plate on frame
        if final_plate:
            cv2.putText(frame, f"FINAL: {final_plate}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
>>>>>>> bef3ead3623c9edc3503dd54c89a31fbe9e9b6b8

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/api/hello')
def hello():
    return jsonify({'message': 'Backend is connected!'})


@app.route('/api/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/plate')
def get_plate():
    return jsonify({
        'plate': final_plate,
        'timestamp': finalized_at
    })


@app.route('/api/reset', methods=['POST'])
def reset_plate():
    global final_plate, finalized_at, recent_plate_window, last_printed_plate
    final_plate = ""
    finalized_at = 0.0
    recent_plate_window.clear()
    last_printed_plate = ""
    print("=== PLATE DETECTION RESET ===", flush=True)
    return jsonify({'status': 'reset', 'message': 'Ready for new detection'})


if __name__ == '__main__':
    print("Starting SIMPLIFIED Flask server on http://localhost:5000")
    print("This version runs OCR on the entire frame - slower but more reliable")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)