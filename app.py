import flask
import cv2
import re
import time
from collections import Counter, deque

import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

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