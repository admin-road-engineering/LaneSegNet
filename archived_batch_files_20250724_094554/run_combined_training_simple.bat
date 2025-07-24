@echo off
echo ===================================
echo COMBINED DATASET TRAINING
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting combined dataset training...
python scripts/combined_dataset_train.py --approach combined_training --lr 0.0005 --dice 0.6 --epochs 30

pause