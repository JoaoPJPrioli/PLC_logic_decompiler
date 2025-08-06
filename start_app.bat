@echo off
REM Quick Start Script for PLC Logic Decompiler Web Application
REM This batch file automates the startup process

echo.
echo ========================================
echo  PLC Logic Decompiler - Quick Start
echo ========================================
echo.

REM Navigate to project directory
echo [1/4] Navigating to project directory...
cd /d "c:\Users\jjacominiprioli\OneDrive - North Carolina A&T State University\Articles\PLC logic decompiler"

REM Check if we're in the right directory
if not exist "app.py" (
    echo ERROR: app.py not found! Please check the project path.
    pause
    exit /b 1
)

echo [2/4] Activating conda environment 'pyoccenv'...
call conda activate pyoccenv

echo [3/4] Installing/updating web dependencies...
conda run -n pyoccenv pip install flask werkzeug jinja2 itsdangerous click markupsafe --quiet

echo [4/4] Starting Flask web application...
echo.
echo =========================================
echo  Starting PLC Logic Decompiler Server
echo =========================================
echo.
echo Server will be available at:
echo   http://127.0.0.1:5000
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask application
conda run -n pyoccenv python app.py

echo.
echo ========================================
echo  Server stopped. Press any key to exit.
echo ========================================
pause
