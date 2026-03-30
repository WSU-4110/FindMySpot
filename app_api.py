"""
app_api.py - Flask API for FindMySpot
Connects frontend HTML to backend database

Endpoints:
- POST /api/auth/signup - Create new user account
- POST /api/auth/login - Login user and return token
- POST /api/vehicle/checkin - Save vehicle parking location
- POST /api/vehicle/find - Find vehicle by plate number
"""

from flask import Flask, request, jsonify
from flask import send_from_directory

from flask_cors import CORS
from auth_database import AuthDatabase
from vehicle_database import VehicleDatabase
from datetime import datetime
import os
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_index():
    return send_from_directory('mobile-app', 'index.html')
@app.route('/login')
def serve_login():
    return send_from_directory('mobile-app', 'login.html')

@app.route('/signup')
def serve_signup():
    return send_from_directory('mobile-app', 'signup.html')

@app.route('/vehicles')
def serve_vehicles():
    return send_from_directory('mobile-app', 'vehicles.html')

@app.route('/vehicles-new')
def serve_vehicles_new():
    return send_from_directory('mobile-app', 'vehicles-new.html')

@app.route('/history')
def serve_history():
    return send_from_directory('mobile-app', 'history.html')

@app.route('/notifications')
def serve_notifications():
    return send_from_directory('mobile-app', 'notifications.html')

@app.route('/terms')
def serve_terms():
    return send_from_directory('mobile-app', 'terms.html')
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('mobile-app', filename)
# Database password - should come from environment variable
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Initialize database connections
auth_db = AuthDatabase(password=DB_PASSWORD)
vehicle_db = VehicleDatabase(password=DB_PASSWORD)

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """
    Create new user account
    
    Request body:
    {
        "name": "John Doe",
        "email": "john@example.com",
        "password": "password123"
    }
    
    Response:
    {
        "success": true,
        "token": "jwt_token_here",
        "user": {
            "id": 1,
            "email": "john@example.com",
            "username": "john_doe"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not name or not email or not password:
            return jsonify({
                'success': False,
                'error': 'Please provide name, email, and password'
            }), 400
        
        if len(password) < 8:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 8 characters'
            }), 400
        
        # Create username from name (e.g., "John Doe" -> "john_doe")
        username = name.lower().replace(' ', '_')
        
        # Create user
        user_id = auth_db.create_user(
            email=email,
            password=password,
            username=username
        )
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'Account already exists for this email'
            }), 400
        
        # Generate token
        token = auth_db.generate_jwt_token(user_id)
        
        # Get user info
        user = auth_db.get_user_by_id(user_id)
        
        logger.info(f"New user signed up: {email}")
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'username': user['username'],
                'name': name  # Return original name
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({
            'success': False,
            'error': 'Signup failed. Please try again.'
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """
    Login user and return JWT token
    
    Request body:
    {
        "email": "john@example.com",
        "password": "password123"
    }
    
    Response:
    {
        "success": true,
        "token": "jwt_token_here",
        "user": {
            "id": 1,
            "email": "john@example.com",
            "username": "john_doe"
        }
    }
    """
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Please enter your email and password'
            }), 400
        
        # Verify password
        if not auth_db.verify_password(email, password):
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
        
        # Get user
        user = auth_db.get_user_by_email(email)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
        
        # Generate token
        token = auth_db.generate_jwt_token(user['id'])
        
        # Update last login
        auth_db.update_last_login(user['id'])
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'username': user['username']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'Login failed. Please try again.'
        }), 500


# ============================================================================
# VEHICLE ENDPOINTS
# ============================================================================

def get_user_from_token():
    """
    Extract user ID from JWT token in Authorization header
    
    Returns:
        user_id (int) or None
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    try:
        # Format: "Bearer <token>"
        token = auth_header.split(' ')[1] if ' ' in auth_header else auth_header
        user_id = auth_db.verify_jwt_token(token)
        return user_id
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


@app.route('/api/vehicle/checkin', methods=['POST'])
def checkin_vehicle():
    """
    Check in a vehicle (save parking location)
    
    Request body:
    {
        "plate": "ABC123",
        "floor": "Floor 1",
        "spot": "A1"
    }
    
    Headers:
        Authorization: Bearer <token>
    
    Response:
    {
        "success": true,
        "message": "Checked in ABC123 successfully"
    }
    """
    try:
        # Get user from token
        user_id = get_user_from_token()
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized. Please log in.'
            }), 401
        
        data = request.get_json()
        
        plate = data.get('plate', '').strip().upper().replace(' ', '')
        floor = data.get('floor', '').strip()
        spot = data.get('spot', '').strip().upper()
        
        if not plate or not floor or not spot:
            return jsonify({
                'success': False,
                'error': 'Please provide plate, floor, and spot'
            }), 400
        
        # Check if vehicle exists for this user
        vehicle = vehicle_db.get_vehicle_by_plate(plate, user_id)
        
        if vehicle:
            # Update existing vehicle with parking location
            vehicle_db.update_vehicle(
                vehicle_id=vehicle['id'],
                floor=floor,
                spot=spot
            )
            logger.info(f"Updated vehicle {plate} parking location for user {user_id}")
        else:
            # Add new vehicle with parking location
            vehicle_id = vehicle_db.add_vehicle(
                user_id=user_id,
                license_plate=plate,
                floor=floor,
                spot=spot,
                is_primary=True  # First vehicle is primary
            )
            
            if not vehicle_id:
                return jsonify({
                    'success': False,
                    'error': 'Failed to add vehicle'
                }), 500
            
            logger.info(f"Added new vehicle {plate} for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': f'Checked in {plate} successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Checkin error: {e}")
        return jsonify({
            'success': False,
            'error': 'Check-in failed. Please try again.'
        }), 500


@app.route('/api/vehicle/find/<plate>', methods=['GET'])
def find_vehicle(plate):
    """
    Find vehicle parking location by plate number
    
    Request body:
    {
        "plate": "ABC123"
    }
    
    Headers:
        Authorization: Bearer <token>
    
    Response:
    {
        "success": true,
        "floor": "Floor 1",
        "spot": "A1",
        "plate": "ABC123",
        "parkedSince": "2026-02-21T10:30:00"
    }
    """
    try:
        # Get user from token
        user_id = get_user_from_token()
        if not user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401

        plate = plate.strip().upper().replace(' ', '')

        vehicle = vehicle_db.get_vehicle_by_plate(plate, user_id)

        if not vehicle or not vehicle.get('floor') or not vehicle.get('spot'):
            return jsonify({'success': False, 'error': 'Vehicle not found'}), 404

        return jsonify({
            'success': True,
            'data': {
                'vehiclePlate': vehicle['license_plate'],
                'floor': vehicle['floor'],
                'lot': vehicle['spot'],
                'spotNumber': f"F{vehicle['floor']}-S{vehicle['spot']}",
                'area': f"Floor {vehicle['floor']}",
                'locationDescription': f"Floor {vehicle['floor']}, Spot {vehicle['spot']}",
                'parkedSince': vehicle['updated_at'].isoformat() if vehicle.get('updated_at') else datetime.now().isoformat()
            },
            'preciseSpotAvailable': True,
            'navigationAvailable': False
        }), 200

    except Exception as e:
        logger.error(f"Find vehicle error: {e}")
        return jsonify({'success': False, 'error': 'Lookup failed.'}), 500

        
    except Exception as e:
        logger.error(f"Find vehicle error: {e}")
        return jsonify({
            'success': False,
            'error': 'Lookup failed. Please try again.'
        }), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'FindMySpot API is running'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """API root"""
    return jsonify({
        'message': 'FindMySpot API',
        'version': '1.0',
        'endpoints': {
            'auth': {
                'signup': 'POST /api/auth/signup',
                'login': 'POST /api/auth/login'
            },
            'vehicle': {
                'checkin': 'POST /api/vehicle/checkin',
                'find': 'POST /api/vehicle/find'
            }
        }
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("FindMySpot API Server")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  POST http://localhost:5000/api/auth/signup")
    print("  POST http://localhost:5000/api/auth/login")
    print("  POST http://localhost:5000/api/vehicle/checkin")
    print("  POST http://localhost:5000/api/vehicle/find")
    print("\n" + "="*70)
    print("Starting server on http://localhost:5000")
    print("="*70)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )