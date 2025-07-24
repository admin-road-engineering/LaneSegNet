@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo SIMPLE FINE-TUNING APPROACH
echo Continue Training with Lower LR
echo ===================================
echo.
echo APPROACH: Run premium_gpu_train.py again but with:
echo - Very low learning rate (1e-5)
echo - Load optimized parameters
echo - Let it automatically find the best checkpoint
echo.
echo This will continue improving from current best model
echo Target: 85.5-86.0%% mIoU from current 85.1%%
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying best model exists...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo ✓ Best model found: 85.1%% mIoU
) else (
    echo ❌ No best model found!
    pause
    exit /b 1
)
echo.
echo Starting fine-tuning with very low learning rate...
echo This will run premium training but continue from best checkpoint
echo.
pause

REM Run premium training with fine-tuning learning rate
python scripts/premium_gpu_train.py --lr 1e-5 --load-optimized

echo.
echo ===================================
echo FINE-TUNING COMPLETE!
echo ===================================
echo.
echo Check work_dirs/ for improved models
echo Look for mIoU improvements beyond 85.1%%
echo.
pause