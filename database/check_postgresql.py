"""
check_postgresql.py - Diagnose PostgreSQL installation issues
"""
import socket
import subprocess
import sys
import os

def check_port_open(host, port):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_postgresql_service():
    """Check if PostgreSQL service is running on Windows"""
    try:
        # Try to find PostgreSQL service
        result = subprocess.run(
            ['sc', 'query', 'postgresql'],
            capture_output=True,
            text=True
        )
        if 'RUNNING' in result.stdout:
            return True, "Service is running"
        elif 'STOPPED' in result.stdout:
            return False, "Service is installed but stopped"
        else:
            # Try alternative service names
            for name in ['postgresql-x64-16', 'postgresql-x64-15', 'postgresql-x64-14']:
                result = subprocess.run(
                    ['sc', 'query', name],
                    capture_output=True,
                    text=True
                )
                if 'RUNNING' in result.stdout:
                    return True, f"Service '{name}' is running"
                elif 'STOPPED' in result.stdout:
                    return False, f"Service '{name}' is installed but stopped"
            
            return False, "Service not found"
    except Exception as e:
        return False, f"Cannot check service: {e}"

def find_postgresql_install():
    """Try to find PostgreSQL installation"""
    possible_paths = [
        r"C:\Program Files\PostgreSQL",
        r"C:\Program Files (x86)\PostgreSQL",
        r"C:\PostgreSQL",
    ]
    
    found = []
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                version_path = os.path.join(base_path, item)
                if os.path.isdir(version_path):
                    bin_path = os.path.join(version_path, 'bin', 'psql.exe')
                    if os.path.exists(bin_path):
                        found.append(version_path)
    
    return found

def main():
    print("="*70)
    print("PostgreSQL Installation Diagnostic")
    print("="*70)
    print()
    
    # Check 1: Is port 5432 open?
    print("Check 1: PostgreSQL Port (5432)")
    port_open = check_port_open('localhost', 5432)
    if port_open:
        print("  ✓ Port 5432 is open - PostgreSQL might be running")
    else:
        print("  ✗ Port 5432 is closed - PostgreSQL is not running or not installed")
    print()
    
    # Check 2: Is PostgreSQL service running?
    print("Check 2: PostgreSQL Service Status")
    service_running, service_msg = check_postgresql_service()
    if service_running:
        print(f"  ✓ {service_msg}")
    else:
        print(f"  ✗ {service_msg}")
    print()
    
    # Check 3: Can we import psycopg2?
    print("Check 3: Python PostgreSQL Library")
    try:
        import psycopg2
        print("  ✓ psycopg2-binary is installed")
    except ImportError:
        print("  ✗ psycopg2-binary is NOT installed")
        print("     Run: pip install psycopg2-binary")
    print()
    
    # Check 4: Find PostgreSQL installation
    print("Check 4: PostgreSQL Installation Location")
    installations = find_postgresql_install()
    if installations:
        print(f"  ✓ Found {len(installations)} PostgreSQL installation(s):")
        for path in installations:
            print(f"     - {path}")
    else:
        print("  ✗ PostgreSQL not found in common installation directories")
    print()
    
    # Summary and recommendations
    print("="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if port_open and service_running:
        print("✓ PostgreSQL appears to be installed and running!")
        print("\nNext step: Run setup_database.py to create the database")
        
    elif not port_open and not service_running:
        print("✗ PostgreSQL is either not installed or not running")
        print("\nRecommended actions:")
        print("1. Check if PostgreSQL is installed:")
        if installations:
            print("   - Found at:", installations[0])
            print("\n2. Start the PostgreSQL service:")
            print("   - Press Windows + R")
            print("   - Type: services.msc")
            print("   - Find 'postgresql' service")
            print("   - Right-click → Start")
        else:
            print("   - Not found. Download from: https://www.postgresql.org/download/windows/")
            print("   - Install PostgreSQL 16")
            print("   - During installation, set a password for 'postgres' user")
            print("   - Make sure to start the service during installation")
    
    elif installations and not service_running:
        print("⚠ PostgreSQL is installed but the service is not running")
        print("\nRecommended actions:")
        print("1. Start the PostgreSQL service:")
        print("   - Press Windows + R")
        print("   - Type: services.msc")
        print("   - Find 'postgresql' service")
        print("   - Right-click → Start")
        print("\n   OR use command line:")
        print('   - Run PowerShell as Administrator')
        print('   - Run: net start postgresql-x64-16  (adjust version number)')
    
    else:
        print("⚠ Unclear status - PostgreSQL may be partially configured")
        print("\nRecommended actions:")
        print("1. Try to connect manually:")
        print("   - Run: python")
        print("   - import psycopg2")
        print("   - psycopg2.connect(host='localhost', database='postgres', user='postgres', password='YOUR_PASSWORD')")
    
    print()
    print("="*70)

if __name__ == "__main__":
    main()
