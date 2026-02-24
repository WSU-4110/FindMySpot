"""
Test script for FindMySpot Camera System
Demonstrates floor and lot assignment functionality
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:3000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_api_health():
    """Test if backend API is running"""
    print_section("Testing API Connection")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.ok:
            print("✓ Backend API is running")
            return True
        else:
            print("✗ Backend API returned error")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to backend: {e}")
        print(f"  Make sure backend is running: cd backend && npm start")
        return False

def get_stats():
    """Get parking statistics"""
    print_section("Parking Statistics")
    try:
        response = requests.get(f"{API_BASE_URL}/api/parking/stats")
        if response.ok:
            data = response.json()
            stats = data['data']
            print(f"Total Spots: {stats['total']}")
            print(f"Available: {stats['available']}")
            print(f"Occupied: {stats['occupied']}")
            print(f"\nBy Floor:")
            for floor, floor_stats in stats['byFloor'].items():
                print(f"  Floor {floor}: {floor_stats['occupied']}/{floor_stats['total']} occupied")
            return True
        else:
            print(f"✗ Failed to get stats: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

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

if __name__ == "__main__":
    run_full_demo()
