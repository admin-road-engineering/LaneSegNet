@echo off
echo ===================================
echo COMPREHENSIVE MODEL VALIDATION
echo Testing ALL backup models
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting comprehensive validation...
echo This will test ALL models in model_backups/ directory
echo Each model will be evaluated on 500 validation samples
echo.

python scripts/comprehensive_model_validation.py

echo.
echo Validation complete! Check the results file for detailed analysis.
pause