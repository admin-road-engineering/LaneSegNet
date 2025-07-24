@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo FAST OPTION 1: CONTINUE TRAINING
echo Speed-optimized proven parameters
echo ===================================
echo.
echo FAST APPROACH: Continue with proven settings (20 epochs)
echo - Load 85.1%% mIoU checkpoint
echo - Use Bayesian-optimized parameters (LR=5.14e-4, Dice=0.694)
echo - Larger batch size (16 vs 8) for speed
echo - Multi-processing (4 workers) for faster data loading
echo - 20 epochs instead of 50 for quick results
echo.
echo Expected outcome: Quick improvement to 86-87%% mIoU
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
echo Starting Fast Option 1: Speed-optimized training...
echo This uses proven parameters with speed optimizations
echo.
pause

python scripts/fast_gpu_train.py --approach fast_option1 --lr 0.0005144736127521129 --dice 0.694

echo.
echo ===================================
echo FAST OPTION 1 COMPLETE!
echo Check model_backups/ for improvements
echo ===================================
echo.
pause