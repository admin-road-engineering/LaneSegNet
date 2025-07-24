@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo TRUE FINE-TUNING FROM 85.1%% mIoU
echo Properly loads Epoch 50 checkpoint
echo ===================================
echo.
echo ✅ Loads 85.1%% mIoU checkpoint from Epoch 50
echo ✅ Continues from Epoch 51 (not starting fresh)
echo ✅ Uses fine-tuning LR: 1e-5 (not 5.14e-4)
echo ✅ Saves any improvement > 0.01%%
echo ✅ Target: 85.5-86.0%% mIoU improvement
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying 85.1%% checkpoint exists...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo ✓ Checkpoint ready
) else (
    echo ❌ Checkpoint missing!
    pause
    exit /b 1
)
echo.
echo Starting TRUE fine-tuning (loads checkpoint properly)...
echo This will continue from the actual 85.1%% model
echo.
pause

python scripts/finetune_from_85_1.py

echo.
echo ===================================
echo TRUE FINE-TUNING COMPLETE!
echo ===================================
echo.
echo Check model_backups/ for any improvements beyond 85.1%%
echo.
pause