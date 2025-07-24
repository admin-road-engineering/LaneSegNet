@echo off
echo ===================================
echo TOP 3 MODEL VALIDATION TEST
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Testing our top 3 candidate models...
python scripts/test_top_models.py

pause