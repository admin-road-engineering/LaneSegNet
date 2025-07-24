@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo COMBINED DATASET TRAINING
echo Multi-dataset approach for enhanced generalization
echo ===================================
echo.
echo TRAINING APPROACH: Multi-Dataset Fine-tuning
echo - Load 85.1%% mIoU checkpoint as starting point
echo - Train on AEL + SS_Dense + SS_Multi_Lane combined dataset
echo - Total samples: 5491 (AEL: 5471 + SS_Dense: 10 + SS_Multi_Lane: 10)
echo - Fine-tuning learning rate for improved generalization
echo.
echo Expected outcome: Enhanced performance across diverse scenarios
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Verifying combined dataset...
if exist "data\combined_lane_dataset\dataset_info.json" (
    echo SUCCESS: Combined dataset ready
    python -c "import json; info = json.load(open('data/combined_lane_dataset/dataset_info.json')); print(f'Total samples: {info[\"total_samples\"]}')"
) else (
    echo ERROR: Combined dataset not found! Run prepare_combined_datasets.py first
    pause
    exit /b 1
)
echo.
echo Verifying 85.1%% checkpoint...
if exist "work_dirs\premium_gpu_best_model.pth" (
    echo SUCCESS: 85.1%% checkpoint ready for fine-tuning
    python -c "import torch; cp = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu'); print(f'Baseline: Epoch {cp.get(\"epoch\", \"unknown\")}, mIoU {cp.get(\"best_miou\", 0)*100:.1f}%%')"
) else (
    echo ERROR: No checkpoint found!
    pause
    exit /b 1
)
echo.
echo Starting Combined Dataset Training...
echo This will enhance generalization across multiple datasets
echo.
pause

python scripts/combined_dataset_train.py --approach combined_training --lr 0.0005 --dice 0.6 --epochs 30

echo.
echo ===================================
echo COMBINED TRAINING COMPLETE!
echo Check model_backups/ for enhanced models
echo ===================================
echo.
pause