@echo off
echo Starting LaneSegNet Infrastructure Analysis API...
echo.

REM Check if port 8010 is already in use
echo Checking if port 8010 is available...
netstat -ano | findstr :8010 > nul
if %errorlevel% == 0 (
    echo WARNING: Port 8010 is already in use!
    echo Please stop the existing service before starting a new one.
    echo.
    netstat -ano | findstr :8010
    echo.
    pause
    exit /b 1
)

echo Port 8010 is available. Starting server...
echo.

REM Start the FastAPI server with uvicorn
python -m uvicorn app.main:app --reload --port 8010 --host 0.0.0.0

echo.
echo Server stopped.
pause