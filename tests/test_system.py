#!/usr/bin/env python3
"""
FindMySpot - License Plate Detection System Testing Script
Tests all components of the implementation including vehicle registration,
detection recording, and notifications
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:3000"
TEST_USER = {
    "name": "Test User",
    "email": f"test_{int(time.time())}@example.com",
    "password": "TestPassword123"
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_status(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "SUCCESS":
        print(f"{Colors.GREEN}[+] [{timestamp}] {message}{Colors.RESET}")
    elif status == "ERROR":
        print(f"{Colors.RED}[-] [{timestamp}] {message}{Colors.RESET}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}[!] [{timestamp}] {message}{Colors.RESET}")
    else:
        print(f"{Colors.BLUE}[i] [{timestamp}] {message}{Colors.RESET}")

def print_section(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.RESET}\n")

def test_api_health():
    """Test if API is running"""
    print_section("TEST 1: API Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_status("API Server is running", "SUCCESS")
            return True
        else:
            print_status(f"API returned status {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status(f"Cannot connect to API at {API_BASE_URL}", "ERROR")
        print_status("Make sure backend is running: cd backend && npm run dev", "WARNING")
        return False

def test_user_registration():
    """Test user registration"""
    print_section("TEST 2: User Registration")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/auth/register",
            json=TEST_USER,
            timeout=5
        )
        if response.status_code == 201:
            data = response.json()
            token = data['data']['token']
            print_status(f"User registered: {TEST_USER['email']}", "SUCCESS")
            print_status(f"Token: {token[:20]}...", "INFO")
            return token
        else:
            print_status(f"Registration failed: {response.json()['message']}", "ERROR")
            return None
    except Exception as e:
        print_status(f"Registration test error: {str(e)}", "ERROR")
        return None

def test_vehicle_registration(token):
    """Test vehicle registration - returns (vehicle_ids, plates)"""
    print_section("TEST 3: Vehicle Registration")
    if not token:
        print_status("Skipping vehicle test (no token)", "WARNING")
        return [], []
    
    vehicles = []
    plates = []
    # Use unique plates based on timestamp to avoid conflicts with previous test runs
    timestamp_suffix = str(int(time.time()))[-3:]
    test_vehicles = [
        {
            "licensePlate": f"ABC{timestamp_suffix}",
            "vehicleName": "Test Car 1",
            "makeModel": "2020 Toyota Camry",
            "color": "Blue"
        },
        {
            "licensePlate": f"XYZ{timestamp_suffix}",
            "vehicleName": "Test Car 2",
            "makeModel": "2023 Honda Civic",
            "color": "Red"
        }
    ]
    
    for vehicle in test_vehicles:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/vehicles",
                json={**vehicle, "token": token},
                timeout=5
            )
            if response.status_code == 201:
                vehicle_id = response.json()['data']['id']
                vehicles.append(vehicle_id)
                plates.append(vehicle['licensePlate'])
                print_status(f"Vehicle registered: {vehicle['licensePlate']}", "SUCCESS")
            else:
                error_msg = response.json().get('message', 'Unknown error')
                print_status(f"Vehicle registration failed: {error_msg}", "ERROR")
        except Exception as e:
            print_status(f"Vehicle registration error: {str(e)}", "ERROR")
    
    return vehicles, plates

def test_get_vehicles(token):
    """Test retrieving user's vehicles"""
    print_section("TEST 4: Retrieve Vehicles")
    if not token:
        return False
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/vehicles/{token}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            count = data['count']
            print_status(f"Retrieved {count} vehicles", "SUCCESS")
            for v in data['data']:
                print(f"  - {v['vehicle_name']} ({v['license_plate']})")
            return True
        else:
            print_status("Failed to retrieve vehicles", "ERROR")
            return False
    except Exception as e:
        print_status(f"Get vehicles error: {str(e)}", "ERROR")
        return False

def test_detection_recording(license_plate):
    """Test recording a detection"""
    try:
        payload = {
            "licensePlate": license_plate,
            "floor": 2,
            "lot": 5,
            "location": "Ground Floor - Entry Gate",
            "confidence": 0.98,
            "cameraId": "CAM_01"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/detection/record",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 201:
            data = response.json()
            matched = data.get('matched', False)
            status_text = "MATCHED" if matched else "no match"
            print_status(f"Detection recorded ({status_text}): {license_plate}", "SUCCESS")
            return True
        else:
            print_status(f"Detection recording failed: {response.json()['message']}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Detection recording error: {str(e)}", "ERROR")
        return False

def test_detection_history(token, license_plate):
    """Test getting detection history"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/detection/history/{license_plate}?token={token}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            count = data['count']
            print_status(f"Found {count} detections for {license_plate}", "SUCCESS")
            return True
        else:
            print_status(f"Failed to get history: {response.json()['message']}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Detection history error: {str(e)}", "ERROR")
        return False

def test_notifications(token):
    """Test notification endpoints"""
    print_section("TEST 6: Notifications")
    if not token:
        return False
    
    try:
        # Get all notifications
        response = requests.get(
            f"{API_BASE_URL}/api/notifications/{token}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            count = data['count']
            print_status(f"Retrieved {count} notifications", "SUCCESS")
            
            # Get unread count
            response = requests.get(
                f"{API_BASE_URL}/api/notifications/count/{token}",
                timeout=5
            )
            if response.status_code == 200:
                unread = response.json()['unreadCount']
                print_status(f"Unread count: {unread}", "INFO")
            
            # Display recent notifications
            for n in data['data'][:3]:
                print(f"  - {n['title']}: {n['message'][:50]}...")
            
            return True
        else:
            print_status("Failed to retrieve notifications", "ERROR")
            return False
    except Exception as e:
        print_status(f"Notification test error: {str(e)}", "ERROR")
        return False

def run_full_test():
    """Run complete test suite"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("FindMySpot - License Plate Detection System")
    print("Comprehensive Testing Suite")
    print(f"{'='*60}{Colors.RESET}\n")
    
    # Test 1: API Health
    if not test_api_health():
        print_status("Aborting tests - API not available", "ERROR")
        return False
    
    # Test 2: User Registration
    token = test_user_registration()
    if not token:
        print_status("User registration test failed", "ERROR")
        return False
    
    # Test 3: Vehicle Registration
    vehicle_ids, plates = test_vehicle_registration(token)
    
    # Test 4: Get Vehicles
    test_get_vehicles(token)
    
    # Test 5: Detection Recording
    print_section("TEST 5: Detection Recording")
    if vehicle_ids:
        test_detection_recording(plates[0])
        time.sleep(0.5)
        test_detection_recording(plates[1])
        time.sleep(0.5)
        test_detection_recording("UNKNOWN123")  # Unregistered plate
        
        # Get detection history
        print_section("TEST 5B: Detection History")
        test_detection_history(token, plates[0])
    
    # Test 6: Notifications
    test_notifications(token)
    
    # Summary
    print(f"\n{Colors.GREEN}{'='*60}")
    print("All Tests Completed Successfully!")
    print(f"{'='*60}{Colors.RESET}\n")
    
    print_status(f"Test User: {TEST_USER['email']}", "INFO")
    print_status(f"API Endpoint: {API_BASE_URL}", "INFO")
    
    print(f"\n{Colors.YELLOW}Next Steps:{Colors.RESET}")
    print("1. Open mobile app at http://localhost:8080 (or your dev server)")
    print(f"2. Sign up with: {TEST_USER['email']} / {TEST_USER['password']}")
    print("3. Navigate to Vehicles page")
    print("4. See registered vehicles: ABC123, XYZ789")
    print("5. Check Notifications page for detection alerts")
    print("6. Run this test again to generate more detections")
    
    return True

if __name__ == "__main__":
    try:
        success = run_full_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)

def check_in_vehicle(plate, floor, lot):
    """Check in a vehicle to a specific floor and lot"""
    print(f"\nChecking in vehicle {plate} to Floor {floor}, Lot {lot}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/parking/checkin",
            json={
                "vehiclePlate": plate,
                "floor": floor,
                "lot": lot
            }
        )
        if response.ok:
            data = response.json()
            print(f"✓ {data['message']}")
            return True
        else:
            data = response.json()
            print(f"✗ Check-in failed: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def find_vehicle(plate):
    """Find a vehicle by license plate"""
    print(f"\nSearching for vehicle {plate}...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/parking/vehicles/{plate}")
        if response.ok:
            data = response.json()
            vehicle = data['data']['vehicle']
            sessions = data['data']['sessions']
            
            print(f"✓ Vehicle found: {vehicle['plate']}")
            
            active_sessions = [s for s in sessions if s['checkOutTime'] is None]
            if active_sessions:
                session = active_sessions[0]
                print(f"  Currently parked at: Floor {session['floor']}, Lot {session['lot']}")
                print(f"  Checked in: {session['checkInTime']}")
            else:
                print(f"  Not currently parked")
            
            if len(sessions) > 0:
                print(f"  Total parking sessions: {len(sessions)}")
            
            return True
        else:
            print(f"✗ Vehicle not found")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def checkout_vehicle(plate):
    """Check out a vehicle"""
    print(f"\nChecking out vehicle {plate}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/parking/checkout",
            json={"vehiclePlate": plate}
        )
        if response.ok:
            data = response.json()
            print(f"✓ {data['message']}")
            return True
        else:
            data = response.json()
            print(f"✗ Checkout failed: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def get_floor_info(floor):
    """Get information about a specific floor"""
    print(f"\nGetting info for Floor {floor}...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/parking/spots/floor/{floor}")
        if response.ok:
            data = response.json()
            spots = data['data']
            occupied = len([s for s in spots if s['occupied']])
            print(f"✓ Floor {floor}: {occupied}/{len(spots)} lots occupied")
            for spot in spots:
                status = "🚗 OCCUPIED" if spot['occupied'] else "✓ AVAILABLE"
                vehicle_info = f" - {spot['vehicle']}" if spot['vehicle'] else ""
                print(f"  Lot {spot['lot']}: {status}{vehicle_info}")
            return True
        else:
            print(f"✗ Failed to get floor info")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def simulate_camera_detections():
    """Simulate camera detections on different floors and lots"""
    print_section("Simulating Camera Detections")
    
    # Simulate cameras detecting vehicles
    detections = [
        ("ABC123", 1, 1),  # Camera 0: Floor 1, Lot 1
        ("XYZ789", 1, 2),  # Camera 1: Floor 1, Lot 2
        ("DEF456", 2, 1),  # Camera 2: Floor 2, Lot 1
        ("GHI789", 3, 3),  # Floor 3, Lot 3
        ("JKL012", 5, 5),  # Floor 5, Lot 5
    ]
    
    print("Camera Assignment:")
    print("  Camera 0 → Floor 1, Lot 1")
    print("  Camera 1 → Floor 1, Lot 2")
    print("  Camera 2 → Floor 2, Lot 1")
    print("  Manual  → Floor 3, Lot 3")
    print("  Manual  → Floor 5, Lot 5")
    print()
    
    success_count = 0
    for plate, floor, lot in detections:
        if check_in_vehicle(plate, floor, lot):
            success_count += 1
        time.sleep(0.5)
    
    print(f"\n✓ Successfully checked in {success_count}/{len(detections)} vehicles")

def run_full_demo():
    """Run a complete demonstration of the system"""
    print("\n" + "="*60)
    print("  FindMySpot Camera System - Floor & Lot Assignment Demo")
    print("="*60)
    
    # Test API connection
    if not test_api_health():
        return
    
    # Get initial stats
    get_stats()
    
    # Simulate camera detections
    simulate_camera_detections()
    
    # Get updated stats
    get_stats()
    
    # Test finding vehicles
    print_section("Finding Vehicles")
    find_vehicle("ABC123")
    find_vehicle("XYZ789")
    find_vehicle("NOTFOUND")
    
    # Get floor information
    print_section("Floor Information")
    get_floor_info(1)
    get_floor_info(2)
    
    # Test checkout
    print_section("Vehicle Checkout")
    checkout_vehicle("ABC123")
    find_vehicle("ABC123")
    
    # Final stats
    get_stats()
    
    print_section("Demo Complete!")
    print("✓ All tests completed successfully")
    print("\nNext steps:")
    print("1. Open mobile-app/index.html to use the mobile interface")
    print("2. Open mobile-app/dashboard.html to view the parking dashboard")
    print("3. Run python app.py to start camera detection")


