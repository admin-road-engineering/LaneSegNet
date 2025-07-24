@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo FINE-TUNING PREMIUM MODEL - Option 3
echo Starting from 85.1% mIoU checkpoint
echo ===================================
echo.
echo FINE-TUNING CONFIGURATION:
echo - Base model: Epoch 50 (85.1%% mIoU)
echo - Learning rate: 1e-5 (conservative)
echo - Additional epochs: 30 (51-80)
echo - Target: 85.5-86.0%% mIoU
echo.
echo ENHANCEMENTS ACTIVE:
echo - SWA (Stochastic Weight Averaging) last 10 epochs
echo - EMA with gentle regularization
echo - Automated best model saving (every 0.05%% improvement)
echo - Cosine annealing scheduler
echo - Light DropBlock + reduced augmentations
echo.
echo Expected Timeline: 8-12 hours fine-tuning
echo Auto-saves: Every new best + every 5 epochs
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"
echo.
echo Verifying Epoch 50 checkpoint exists...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo ✓ Epoch 50 checkpoint found (85.1%% mIoU^)
    for %%I in ("work_dirs\premium_gpu_best_model.pth") do set /a size_mb=%%~zI/1024/1024
    echo   File size: !size_mb! MB
    echo   Ready for fine-tuning...
) else (
    echo ❌ Epoch 50 checkpoint not found!
    echo Please ensure premium_gpu_best_model.pth exists in work_dirs/
    pause
    exit /b 1
)
echo.
echo Starting fine-tuning from 85.1%% mIoU baseline...
echo Target: Conservative 85.5%% / Optimistic 86.0%% mIoU
echo Auto-saving every improvement >0.05%% mIoU
echo.
pause
python scripts/finetune_premium_model.py
echo.
echo ===================================
echo FINE-TUNING COMPLETED!
echo ===================================
echo.
echo Check model_backups/ for new best checkpoints
echo Each improvement >0.05%% mIoU was automatically saved
echo.
echo RESULTS LOCATION:
echo - Best models: model_backups/finetune_epoch*_best_*/
echo - Backup models: work_dirs/finetune_epoch*_backup.pth
echo.
echo NEXT STEPS:
echo 1. Review fine-tuning results and best mIoU achieved
echo 2. If satisfied, deploy best checkpoint for production
echo 3. If not, proceed with Option 2 (optimized training)
echo.
echo For extended optimization:
echo   Option 2: python scripts/optimized_premium_train.py
echo   Option 1: python scripts/extended_baseline_train.py
echo.
pause