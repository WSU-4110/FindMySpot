"""
License Plate Recognition (LPR) Integration Service
Integrates with cameras and submits detected plates to FindMySpot backend
"""

import requests
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateDetectionService:
    def __init__(self, backend_url='http://localhost:3000'):
        self.backend_url = backend_url
        self.detection_endpoint = f'{backend_url}/api/detection/record'
        
    def report_detection(self, license_plate, floor=None, lot=None, location_description=None, 
                        confidence=0.95, camera_id=None, latitude=None, longitude=None):
        """
        Report a detected license plate to the backend
        
        Args:
            license_plate (str): The detected license plate (e.g., 'ABC123')
            floor (int): Parking floor number
            lot (int): Parking lot number
            location_description (str): Human-readable location
            confidence (float): Detection confidence 0-1
            camera_id (str): Camera identifier
            latitude (float): GPS latitude
            longitude (float): GPS longitude
            
        Returns:
            dict: Response from backend
        """
        
        payload = {
            'licensePlate': license_plate.upper(),
            'floor': floor,
            'lot': lot,
            'location': location_description or f'Floor {floor}, Lot {lot}',
            'confidence': confidence,
            'cameraId': camera_id,
            'latitude': latitude,
            'longitude': longitude
        }
        
        try:
            response = requests.post(
                self.detection_endpoint,
                json=payload,
                timeout=5
            )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Detection reported: {license_plate} at {payload['location']}")
            if data.get('matched'):
                logger.info(f"MATCH FOUND! Notification will be sent to registered user.")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reporting detection: {e}")
            return None

# Example usage for real-time camera integration
def simulate_camera_detections():
    """
    Simulate camera detections for testing
    """
    service = PlateDetectionService()
    
    # Simulated detected plates
    test_detections = [
        {
            'plate': 'ABC123',
            'floor': 2,
            'lot': 5,
            'location': 'Entry Gate - Floor 2',
            'confidence': 0.98,
            'camera_id': 'CAM_01'
        },
        {
            'plate': 'XYZ789',
            'floor': 3,
            'lot': 12,
            'location': 'Main Entrance - Floor 3',
            'confidence': 0.95,
            'camera_id': 'CAM_02'
        },
        {
            'plate': 'DEF456',
            'floor': 1,
            'lot': 8,
            'location': 'Ground Floor North',
            'confidence': 0.92,
            'camera_id': 'CAM_03'
        }
    ]
    
    for detection in test_detections:
        result = service.report_detection(
            license_plate=detection['plate'],
            floor=detection['floor'],
            lot=detection['lot'],
            location_description=detection['location'],
            confidence=detection['confidence'],
            camera_id=detection['camera_id']
        )
        print(f"Detection Result: {json.dumps(result, indent=2)}\n")

if __name__ == '__main__':
    print("License Plate Detection Service")
    print("=" * 50)
    print("\nSimulating camera detections...\n")
    simulate_camera_detections()
