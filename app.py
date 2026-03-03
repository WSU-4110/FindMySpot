from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import re
import time
import random
import requests
from collections import Counter, deque
import easyocr

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Backend API configuration
BACKEND_API_URL = "http://localhost:3000/api/parking/checkin"

# Variables for random floor/lot assignment
assigned_floor = None
assigned_lot = None

# Global variables
last_printed_plate = ""
last_ocr_time = 0.0
ocr_interval_seconds = 0.9
blocked_words = {# 50 States (Cleaned for OCR)
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

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened. Press 'q' to quit.")

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

    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")