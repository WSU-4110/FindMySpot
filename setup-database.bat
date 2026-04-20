@echo off
setlocal

where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0setup_database.py" %*
) else (
  python "%~dp0setup_database.py" %*
)

endlocal