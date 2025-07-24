@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo OPTION 1: CONTINUE PREMIUM TRAINING
echo Same proven parameters, 30 more epochs
echo ===================================
echo.
echo APPROACH: Continue with identical settings that achieved 85.1%%
echo - Load 85.1%% mIoU checkpoint
echo - Use same Bayesian-optimized parameters (LR=5.14e-4, Dice=0.694)
echo - Train epochs 51-80 (30 more epochs)
echo - Auto-save any improvements
echo.
echo Expected outcome: Gradual improvement to 86-87%% mIoU
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.

REM First create the continuation script
echo Creating continuation script...
python scripts/continue_premium_training.py

echo.
echo Verifying checkpoint...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo SUCCESS: 85.1%% checkpoint ready
    python -c "import torch; cp = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu'); print(f'Current: Epoch {cp.get(\"epoch\", \"unknown\")}, mIoU {cp.get(\"best_miou\", 0)*100:.1f}%%')"
) else (
    echo ERROR: No checkpoint found!
    pause
    exit /b 1
)

echo.
echo Starting Option 1: Premium Training Continuation...
echo This will continue with the SAME proven parameters
echo.
pause

python scripts/premium_continue_51_80.py

echo.
echo ===================================
echo OPTION 1 COMPLETE!
echo ===================================
echo.
pause