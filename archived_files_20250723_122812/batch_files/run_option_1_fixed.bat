@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo OPTION 1: CONTINUE PREMIUM TRAINING
echo Simple continuation from 85.1%% checkpoint
echo ===================================
echo.
echo APPROACH: Use identical proven parameters
echo - Load 85.1%% mIoU checkpoint automatically
echo - Use same Bayesian-optimized parameters (LR=5.14e-4, Dice=0.694)
echo - Continue training with proven architecture
echo - Auto-save any improvements to model_backups/
echo.
echo Expected outcome: Gradual improvement to 86-87%% mIoU
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying checkpoint...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo SUCCESS: 85.1%% checkpoint ready for continuation
    python -c "import torch; cp = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu'); print(f'Checkpoint: Epoch {cp.get(\"epoch\", \"unknown\")}, mIoU {cp.get(\"best_miou\", 0)*100:.1f}%%')"
) else (
    echo ERROR: No checkpoint found!
    pause
    exit /b 1
)
echo.
echo Starting Option 1: Premium Training Continuation...
echo The premium training function will automatically load the checkpoint
echo and continue with the same proven parameters
echo.
pause

python scripts/simple_continue_training.py

echo.
echo ===================================
echo OPTION 1 COMPLETE!
echo Check model_backups/ for improvements beyond 85.1%%
echo ===================================
echo.
pause