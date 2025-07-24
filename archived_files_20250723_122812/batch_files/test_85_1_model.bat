@echo off
echo ===================================
echo TEST-TIME AUGMENTATION ENHANCEMENT
echo 85.1%% mIoU Model Performance Boost
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Testing TTA enhancement on 85.1%% model...
python scripts/test_time_augmentation.py

pause