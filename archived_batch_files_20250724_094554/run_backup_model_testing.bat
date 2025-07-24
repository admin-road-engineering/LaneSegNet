@echo off
echo =========================================================
echo BACKUP MODEL TESTING - FIND THE TRUE BEST PERFORMER
echo =========================================================
echo.
echo This will test all backup models to identify which one
echo actually performs best, since the reported "85.1%% mIoU"
echo model is only achieving 32.5%% mIoU in actual testing.
echo.
echo The script will:
echo   1. Find all backup model files
echo   2. Test each on a consistent validation set
echo   3. Compare reported vs actual performance
echo   4. Identify the true best performing model
echo.

call .venv\Scripts\activate.bat
python scripts/test_all_backup_models.py

echo.
echo =========================================================
echo BACKUP MODEL TESTING COMPLETED
echo =========================================================
echo Check the generated comparison report for results.
echo.
pause