import cv2
import re
import time

import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

last_printed_plate = ""
last_ocr_time = 0.0
ocr_interval_seconds = 0.7
blocked_words = {"PERSON", "CELLPHONE", "CELL", "PHONE"}
show_candidate_boxes = True
show_debug_counts = True


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
                if confidence < 0.3:
                    continue
                if cleaned != last_printed_plate:
                    last_printed_plate = cleaned
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] License plate detected: {cleaned}", flush=True)
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

    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")