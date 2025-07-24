@echo off
echo ===================================
echo MODEL TESTING WITH PROGRESS
echo ===================================
echo.

echo System Status Check:
call .venv\Scripts\activate.bat
python scripts/quick_status_check.py
echo.

echo Choose testing option:
echo [1] Single model test (current best)
echo [2] Top 3 models comparison  
echo [3] Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running single model test...
    python scripts/test_single_model.py
) else if "%choice%"=="2" (
    echo.
    echo Running top 3 models comparison...
    python scripts/test_top_models.py
) else if "%choice%"=="3" (
    echo Exiting...
    goto :end
) else (
    echo Invalid choice. Exiting...
)

:end
echo.
pause