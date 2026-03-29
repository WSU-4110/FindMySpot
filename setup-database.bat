@echo off
REM FindMySpot Database Setup Script for Windows
REM This script sets up the database with all migrations

setlocal enabledelayedexpansion

REM PostgreSQL connection details
set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres
set PGPASSWORD=

REM PostgreSQL path
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql"

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
echo You can now:
echo 1. Start backend: cd backend && npm run dev
echo 2. Test system: python test_system.py
echo.
pause
