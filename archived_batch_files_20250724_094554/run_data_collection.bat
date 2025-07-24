@echo off
setlocal
cls

echo ========================================
echo  LaneSegNet Unlabeled Data Collection
echo ========================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Could not activate the virtual environment.
    echo Please ensure '.venv' exists and is set up correctly.
    pause
    exit /b 1
)
echo.

echo This script will download unlabeled aerial imagery from multiple sources:
echo - OpenStreetMap tiles (continue to 1000 target)
echo - SkyScapes dataset (3000 aerial images)
echo - CARLA synthetic scenes (2000 images) 
echo - Cityscapes aerial transforms (1000 images)
echo.
echo Target: ~7000+ unlabeled images for SSL pre-training
echo.
echo Press any key to begin the data collection process...
pause

set "any_errors=0"

echo.
echo [1/4] Completing OSM tile collection...
echo ====================================================
python scripts/collect_osm_1000.py
if %errorlevel% neq 0 (
    echo [WARNING] OSM collection encountered an error.
    set "any_errors=1"
)

echo.
echo [2/4] Downloading SkyScapes dataset...
echo ====================================
python scripts/download_skyscapes.py
if %errorlevel% neq 0 (
    echo [WARNING] SkyScapes download failed.
    set "any_errors=1"
)

echo.
echo [3/4] Generating CARLA synthetic scenes...
echo ========================================
echo Note: Requires CARLA simulator running on localhost:2000
echo If CARLA is not available, this step will be skipped.
python scripts/generate_carla_aerial.py
if %errorlevel% neq 0 (
    echo [INFO] CARLA not available or generation failed, skipping...
)

echo.
echo [4/4] Creating Cityscapes aerial transforms...
echo ============================================
python scripts/transform_cityscapes_aerial.py
if %errorlevel% neq 0 (
    echo [WARNING] Cityscapes transform failed.
    set "any_errors=1"
)

echo.
echo [FINAL] Consolidating all collected data...
echo =========================================
python scripts/consolidate_unlabeled_data.py
if %errorlevel% neq 0 (
    echo [ERROR] Consolidation failed! The collected data may be disorganized.
    set "any_errors=1"
)

echo.
echo ========================================
echo  Collection Complete!
echo ========================================
if "%any_errors%"=="1" (
    echo.
    echo [ATTENTION] One or more steps failed during the process.
    echo Please review the logs above for details.
    echo.
) else (
    echo.
    echo All steps completed successfully.
    echo.
)
echo Check data/unlabeled_aerial/ for results.
echo Run scripts/check_collection_status.py for a summary.
echo.
pause