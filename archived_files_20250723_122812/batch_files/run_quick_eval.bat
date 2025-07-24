@echo off
echo ===================================
echo QUICK MODEL EVALUATION
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Running quick evaluation of current 85.1%% model...
python scripts/quick_model_test.py

pause