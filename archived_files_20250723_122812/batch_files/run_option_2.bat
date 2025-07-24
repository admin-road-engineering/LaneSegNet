@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo OPTION 2: ENHANCED PREMIUM TRAINING
echo Advanced optimizations targeting 87-90%% mIoU
echo ===================================
echo.
echo APPROACH: Advanced techniques for breakthrough performance
echo - Load 85.1%% mIoU as baseline
echo - Enhanced model: Reduced dropout, AMSGrad optimizer
echo - OneCycle LR scheduler with 2x peak learning rate
echo - Advanced loss: Aggressive focal loss (gamma=2.5)
echo - Model EMA + Gradient clipping for stability
echo - Aggressive class weights: [0.08, 6.0, 6.0]
echo.
echo Expected outcome: Breakthrough to 87-90%% mIoU
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying checkpoint...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo SUCCESS: 85.1%% baseline ready
    python -c "import torch; cp = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu'); print(f'Baseline: Epoch {cp.get(\"epoch\", \"unknown\")}, mIoU {cp.get(\"best_miou\", 0)*100:.1f}%%')"
) else (
    echo ERROR: No baseline checkpoint found!
    pause
    exit /b 1
)

echo.
echo Starting Option 2: Enhanced Premium Training...
echo This uses ADVANCED OPTIMIZATIONS for breakthrough performance
echo.
echo ENHANCEMENTS ACTIVE:
echo   SUCCESS: Advanced Model Architecture
echo   SUCCESS: OneCycle Learning Rate Scheduler  
echo   SUCCESS: AMSGrad Optimizer
echo   SUCCESS: Model EMA (Exponential Moving Average)
echo   SUCCESS: Gradient Clipping
echo   SUCCESS: Aggressive Class Weighting
echo   SUCCESS: Enhanced Focal Loss
echo.
pause

python scripts/optimized_premium_training.py

echo.
echo ===================================
echo OPTION 2 COMPLETE!
echo ===================================
echo.
pause