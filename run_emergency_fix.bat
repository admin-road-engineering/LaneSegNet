@echo off
cls
echo ================================================================
echo EMERGENCY TRAINING FIX - Critical Infrastructure Issues
echo ================================================================
echo.
echo CRITICAL ISSUES BEING FIXED:
echo   1. Class collapse (models only predict class 1)
echo   2. Extreme training times (12+ hours without completion)
echo   3. Loss function failures (IoU ~1.1%%)
echo   4. Memory/timeout issues in experiments
echo.
echo FIXES IMPLEMENTED:
echo   • Extreme class weighting (Background: 0.05, Lanes: 50.0+)
echo   • Custom balanced loss with forced class diversity  
echo   • Timeout and checkpointing mechanisms
echo   • Reduced batch size and faster validation
echo   • Curriculum learning approach
echo.
echo Expected Results:
echo   • Training completes in ~30 minutes (vs 12+ hours)
echo   • Model predicts multiple classes (vs only class 1)
echo   • IoU improves to >5%% (vs 1.1%%)
echo   • Enables Phase 4 hyperparameter optimization
echo.

rem Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate the virtual environment.
    pause
    exit /b 1
)

rem Quick dependency check
echo [INFO] Checking PyTorch availability...
python -c "import torch; print(f'PyTorch ready: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch not available
    pause
    exit /b 1
)

echo.
echo ================================================================
echo STARTING EMERGENCY TRAINING FIX
echo ================================================================
echo Timeline: ~30 minutes maximum
echo Progress will be logged in real-time...
echo.

rem Run emergency fix
python scripts/emergency_training_fix.py

set FIX_EXIT_CODE=%errorlevel%

echo.
echo ================================================================
echo EMERGENCY FIX COMPLETE
echo ================================================================

if %FIX_EXIT_CODE% neq 0 (
    echo [ERROR] Emergency fix failed with exit code %FIX_EXIT_CODE%
    echo.
    echo Please check:
    echo   1. Virtual environment is properly activated
    echo   2. All dependencies are installed
    echo   3. Dataset is available
    echo.
    pause
    exit /b 1
)

rem Check results
echo [SUCCESS] Emergency fix completed successfully
echo.

if exist "work_dirs\emergency_fix\emergency_training_results.json" (
    echo Displaying results...
    python -c "
import json
import sys
try:
    with open('work_dirs/emergency_fix/emergency_training_results.json', 'r') as f:
        results = json.load(f)
    
    print('EMERGENCY FIX RESULTS:')
    print('=' * 50)
    print(f'Best IoU achieved: {results[\"best_iou\"]:.1%%}')
    print(f'Issues fixed: {results[\"issues_fixed\"]}')
    print(f'Training successful: {results[\"training_successful\"]}')
    print(f'Epochs completed: {results[\"epochs_completed\"]}')
    
    if results['training_successful']:
        print()
        print('🎉 CRITICAL ISSUES RESOLVED!')
        print('✅ Infrastructure stabilized')
        print('🚀 Ready for Phase 4 optimization')
        print()
        print('Next Steps:')
        print('• Run hyperparameter sweep with fixed infrastructure')
        print('• Target >20%% IoU with systematic optimization')
        print('• Scale to full dataset training')
    else:
        print()
        print('⚠️  Some issues remain - further debugging needed')
        print('📋 Check logs for specific problems')
        
except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"
) else (
    echo [WARNING] Results file not found
)

echo.
echo Press any key to exit...
pause >nul