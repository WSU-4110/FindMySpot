# FindMySpot
A real time **license plate detection system** using **YOLOv8** and **EasyOCR**, with webcam integration and a simple web interface.
Detects license plates from live webcam feed and performs OCR to read the plate number.
---
## Features

- Real-time license plate detection using **YOLOv8**.
- OCR via **EasyOCR** to read license plate text.
- Filters invalid or blocked text (e.g., `PERSON`, `CELLPHONE`).
- Finalizes the most likely plate number using consensus from recent frames.
- Web interface for live webcam feed and backend testing.
- Configurable parameters for confidence, candidate boxes, and OCR intervals.

---

## Project Structure
FindMySpot/
-  app.py # Main Python script
-  index.html # Web interface
-   olov8n.pt # Pre-trained YOLOv8 model
-     ADME.md # This documentation
- requirements.txt # Python dependencies

---

## Requirements
- Python 3.10+
- OpenCV (`cv2`)
- EasyOCR
- Ultralytics YOLO (`ultralytics`)
- Flask
