@echo off
echo Running training infrastructure diagnostics...

rem Activate virtual environment
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate the virtual environment.
    pause
    exit /b 1
)

rem Run diagnostics
python scripts/debug_training_issues.py

pause