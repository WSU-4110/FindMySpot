# FindMySpot Database Setup Script for PowerShell
# This script automatically sets up your database with all migrations

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FindMySpot Database Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# PostgreSQL details
$PSQL_PATH = "C:\Program Files\PostgreSQL\18\bin\psql"
$PGHOST = "localhost"
$PGPORT = 5432
$PGUSER = "postgres"

# Check if psql exists
if (-not (Test-Path $PSQL_PATH)) {
    Write-Host "ERROR: PostgreSQL not found at $PSQL_PATH" -ForegroundColor Red
    Write-Host "Please install PostgreSQL first" -ForegroundColor Yellow
    exit 1
}

# Get current directory
$scriptDir = Get-Location
Write-Host "Working directory: $scriptDir" -ForegroundColor Green
Write-Host ""

# Check if migration files exist
if (-not (Test-Path "database/migrations/001_init.sql")) {
    Write-Host "ERROR: Migration file not found" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Running initial migration (001_init.sql)" -ForegroundColor Yellow
Write-Host ""

# Run initial migration
& $PSQL_PATH -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -f database/migrations/001_init.sql

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Initial migration failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Running license plate detection migration (002_license_plate_detection.sql)" -ForegroundColor Yellow
Write-Host ""

# Run license plate detection migration
& $PSQL_PATH -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -f database/migrations/002_license_plate_detection.sql

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: License plate migration failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Database setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start backend server:" -ForegroundColor White
Write-Host "   cd backend" -ForegroundColor Gray
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "2. In another terminal, start mobile app:" -ForegroundColor White
Write-Host "   cd mobile-app" -ForegroundColor Gray
Write-Host "   python -m http.server 8080" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test the system:" -ForegroundColor White
Write-Host "   python test_system.py" -ForegroundColor Gray
Write-Host ""
