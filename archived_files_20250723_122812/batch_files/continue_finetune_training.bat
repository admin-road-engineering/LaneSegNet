@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo CONTINUE TRAINING - FINE-TUNING MODE
echo From 85.1%% mIoU Checkpoint
echo ===================================
echo.
echo FINE-TUNING APPROACH:
echo - Continue from existing premium training
echo - Load 85.1%% mIoU checkpoint
echo - Use fine-tuning learning rate (1e-5)
echo - Train additional 30 epochs (51-80)
echo - Target: 85.5-86.0%% mIoU
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"
echo.
echo Verifying checkpoint and data...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo ✓ Checkpoint ready: 85.1%% mIoU model
) else (
    echo ❌ Checkpoint missing!
    pause
    exit /b 1
)

python -c "
import json
try:
    with open('data/train_data.json') as f: train = json.load(f)
    with open('data/val_data.json') as f: val = json.load(f)
    print(f'Training data: {len(train)} samples')
    print(f'Validation data: {len(val)} samples')
    print('✓ Data ready')
except Exception as e:
    print(f'❌ Data error: {e}')
"
echo.
echo STARTING FINE-TUNING CONTINUATION...
echo Using the SAME successful premium training script
echo with fine-tuning parameters applied
echo.
pause

REM Continue training with fine-tuning parameters
python scripts/premium_gpu_train.py ^
  --load-checkpoint work_dirs/premium_gpu_best_model.pth ^
  --learning-rate 1e-5 ^
  --start-epoch 51 ^
  --total-epochs 80 ^
  --save-best-only ^
  --fine-tuning-mode

echo.
echo ===================================
echo FINE-TUNING CONTINUATION COMPLETE!
echo ===================================
echo.
echo Results should be in work_dirs/
echo Check for improved mIoU > 85.1%%
echo.
pause