@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo PROPER FINE-TUNING - Fixed Version
echo Starting from 85.1%% mIoU Checkpoint
echo ===================================
echo.
echo ✅ Correct checkpoint: premium_gpu_best_model.pth (85.1%% mIoU)
echo ✅ Fine-tuning parameters: LR=1e-5, Continue from Epoch 51
echo ✅ Target: 85.5-86.0%% mIoU improvement
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Starting premium training with fine-tuning parameters...
echo This will load the 85.1%% checkpoint and continue training
echo.

REM Stop any running training first
echo Press Ctrl+C now if training is still running!
echo Otherwise, press any key to continue...
pause

echo Starting PROPER fine-tuning...
python scripts/premium_gpu_train.py --lr 1e-5 --load-optimized

echo.
echo ===================================
echo FINE-TUNING COMPLETED!
echo ===================================
echo.
pause