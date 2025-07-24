@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo OPTION 1: CONTINUE PREMIUM TRAINING
echo Direct approach - same proven parameters
echo ===================================
echo.
echo APPROACH: Load 85.1%% checkpoint and continue with proven parameters
echo - Learning Rate: 5.14e-04 (Bayesian optimized)
echo - Dice Weight: 0.694 (Bayesian optimized)
echo - Continue from Epoch 50 for 30 more epochs
echo - Uses identical architecture and settings
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying checkpoint...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo SUCCESS: 85.1%% checkpoint ready
    python -c "import torch; cp = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu'); print(f'Baseline: Epoch {cp.get(\"epoch\", \"unknown\")}, mIoU {cp.get(\"best_miou\", 0)*100:.1f}%%')"
) else (
    echo ERROR: No checkpoint found!
    pause
    exit /b 1
)
echo.
echo Starting Option 1: Direct Premium Continuation...
echo Loading checkpoint and continuing with proven parameters
echo.
pause

REM Run the premium script with load-optimized flag to use Bayesian parameters
python scripts/premium_gpu_train.py --load-optimized

echo.
echo ===================================
echo OPTION 1 COMPLETE!
echo Check model_backups/ for new improvements
echo ===================================
echo.
pause