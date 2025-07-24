@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo QUICK FINE-TUNING - Option 3
echo Using proven 85.1%% mIoU architecture
echo ===================================
echo.
echo FINE-TUNING CONFIGURATION:
echo - Base model: Epoch 50 (85.1%% mIoU)
echo - Learning rate: 1e-5 (conservative)
echo - Additional epochs: 30 (51-80)
echo - Target: 85.5-86.0%% mIoU
echo.
echo APPROACH: Use successful premium_gpu_train.py 
echo with fine-tuning parameters and checkpoint loading
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"
echo.
echo Verifying training data...
python -c "
import json
try:
    with open('data/train_data.json') as f:
        train = json.load(f)
    with open('data/val_data.json') as f:
        val = json.load(f)
    print(f'Training data: {len(train)} samples')
    print(f'Validation data: {len(val)} samples')
    if len(train) > 1000 and len(val) > 100:
        print('Data ready for fine-tuning')
    else:
        print('Insufficient data!')
except Exception as e:
    print(f'Data error: {e}')
"
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
echo Starting QUICK FINE-TUNING using proven architecture...
echo Conservative approach: Load checkpoint + fine-tuning parameters
echo.
pause
python -c "
# Quick fine-tuning approach: Modify the successful script parameters
print('Quick Fine-tuning approach:')
print('1. Use premium_gpu_train.py script (proven to work)')
print('2. Modify key parameters for fine-tuning:')
print('   - Load checkpoint: premium_gpu_best_model.pth')
print('   - Learning rate: 1e-5 (very low)')
print('   - Epochs: 51-80 (30 additional)')
print('   - Gentle augmentations')
print('   - Auto-save improvements')
print()
print('This approach uses the exact same architecture that achieved 85.1%%')
print('Press Ctrl+C to stop and implement this approach properly...')
"
echo.
echo ===================================
echo QUICK APPROACH RECOMMENDED
echo ===================================
echo.
echo Instead of new script, modify premium_gpu_train.py to:
echo 1. Load the 85.1%% checkpoint at start
echo 2. Use fine-tuning learning rate (1e-5)
echo 3. Start from epoch 51
echo 4. Auto-save improvements
echo.
echo This ensures we use the EXACT architecture that worked!
echo.
pause