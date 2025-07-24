@echo off
echo =========================================================
echo FULL DATASET PREMIUM TRAINING
echo =========================================================
echo.
echo Training on complete 7,817 sample dataset:
echo   - Training: 5,471 samples (70%%)
echo   - Validation: 782 samples (10%%)  
echo   - Test: 1,564 samples (20%% - holdout)
echo.
echo Target: Achieve 85%%+ mIoU (current baseline: 79.4%%)
echo.
echo Starting premium training with:
echo   - Premium U-Net architecture
echo   - Enhanced loss functions
echo   - Proven hyperparameters
echo   - Class balancing weights
echo.
echo This will take 2-4 hours depending on epochs needed.
echo.

call .venv\Scripts\activate.bat
python scripts/full_dataset_premium_training.py

echo.
echo =========================================================
echo FULL DATASET TRAINING COMPLETED
echo =========================================================
echo Check work_dirs/full_dataset_best_model.pth for results.
echo.
pause