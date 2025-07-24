@echo off
echo ======================================================
echo COMPREHENSIVE MODEL TRAINING & VALIDATION AUDIT
echo ======================================================
echo.
echo This audit will investigate the performance discrepancy:
echo   Reported: 85.1%% mIoU
echo   Actual:   32.5%% mIoU
echo.
echo Audit areas:
echo   1. Model checkpoint integrity verification
echo   2. Dataset split and annotation quality
echo   3. Validation methodology consistency
echo   4. Training log analysis
echo.
echo Starting comprehensive audit...
echo.

call .venv\Scripts\activate.bat
python scripts/comprehensive_audit.py

echo.
echo ======================================================
echo AUDIT COMPLETED
echo ======================================================
echo Check the generated audit report for detailed findings.
echo.
pause