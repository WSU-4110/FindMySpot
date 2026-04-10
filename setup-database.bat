@echo off
setlocal enabledelayedexpansion

set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres
set PGPASSWORD=mir3863

set PSQL="C:\Users\mirza\OneDrive - Wayne State University\CSC 4710\bin\psql.exe"
echo.
echo ========================================
echo FindMySpot Database Setup
echo ========================================
echo.

echo Step 1: Running initial migration (001_init.sql)
echo.
%PSQL% -h %PGHOST% -p %PGPORT% -U %PGUSER% -d postgres -f database/migrations/001_init.sql

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Initial migration failed
    pause
    exit /b 1
)

echo.
echo Step 2: Running license plate detection migration (002_license_plate_detection.sql)
echo.
%PSQL% -h %PGHOST% -p %PGPORT% -U %PGUSER% -d postgres -f database/migrations/002_license_plate_detection.sql

if %errorlevel% neq 0 (
    echo.
    echo ERROR: License plate migration failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Database setup completed successfully!
echo ========================================
echo.
pause