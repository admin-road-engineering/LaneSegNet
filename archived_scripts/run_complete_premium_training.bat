@echo off
echo ===================================
echo COMPLETE PREMIUM TRAINING - Phase 3.2.5
echo Industry-Leading Lane Detection Pipeline
echo ===================================
echo.
echo ENHANCED PIPELINE FEATURES:
echo - Hybrid Loss: DiceFocal + Lovász + Edge + Smoothness
echo - Premium U-Net with Attention and Skip Connections
echo - Advanced Augmentations: MixUp + CutMix + Style Transfer
echo - Bayesian Hyperparameter Optimization (Optional)
echo - Mixed Precision Training for Maximum Efficiency
echo.
echo TRAINING TARGETS:
echo - 80-85%% mIoU: Industry-Leading Performance
echo - All Lane Classes >50%% IoU: Class Balance Resolved
echo - White Lanes >70%% IoU: Production Quality
echo.
echo PIPELINE STEPS:
echo 1. Optional: Bayesian Hyperparameter Optimization
echo 2. Premium GPU Training with Optimized Parameters
echo 3. Comprehensive Model Evaluation
echo.
echo Press any key to start the complete training pipeline...
pause

echo.
echo ===================================
echo STEP 1: BAYESIAN OPTIMIZATION
echo ===================================
echo.
echo Do you want to run Bayesian hyperparameter optimization?
echo This will optimize learning rate and Dice/Focal loss weights.
echo Expected time: 30-60 minutes for 8 trials
echo.
set /p OPTIMIZE="Run optimization? (y/n): "

if /i "%OPTIMIZE%"=="y" (
    echo.
    echo Running Bayesian optimization...
    call .venv\Scripts\activate.bat
    python scripts/bayesian_tuner.py --trials 8
    
    if errorlevel 1 (
        echo ERROR: Bayesian optimization failed, continuing with default parameters...
        set USE_OPTIMIZED=false
    ) else (
        echo Optimization completed successfully!
        set USE_OPTIMIZED=true
    )
) else (
    echo Skipping optimization, using default parameters...
    set USE_OPTIMIZED=false
)

echo.
echo ===================================
echo STEP 2: PREMIUM GPU TRAINING
echo ===================================
echo.
echo GPU Status Check:
call .venv\Scripts\activate.bat
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"

echo.
echo PREMIUM TRAINING CONFIGURATION:
echo - Architecture: Enhanced U-Net with Attention (8.9M parameters)
echo - Loss Function: Hybrid DiceFocal + Lovász + Edge + Smoothness
echo - Augmentations: Advanced geometric + color + scale variations
echo - Training Strategy: Quality-focused with patient early stopping
echo - Expected Duration: 4-8 hours for maximum quality
echo.
echo Starting premium training...

if "%USE_OPTIMIZED%"=="true" (
    echo Using Bayesian-optimized parameters...
    python scripts/premium_gpu_train.py --load-optimized
) else (
    echo Using default parameters...
    python scripts/premium_gpu_train.py
)

if errorlevel 1 (
    echo.
    echo ERROR: Training failed! Check GPU setup and data paths.
    pause
    exit /b 1
)

echo.
echo ===================================
echo STEP 3: MODEL EVALUATION
echo ===================================
echo.
echo Premium training completed! Running comprehensive evaluation...

echo.
echo Detailed Performance Analysis:
python scripts/balanced_eval.py

echo.
echo Quick Performance Monitor:
python scripts/simple_balanced_monitor.py

echo.
echo ===================================
echo PREMIUM TRAINING PIPELINE COMPLETE!
echo ===================================
echo.
echo RESULTS SUMMARY:
echo - Model: work_dirs/premium_gpu_best_model.pth
echo - Results: work_dirs/premium_gpu_results.json
echo - Size: Premium architecture optimized for quality
echo.
echo NEXT STEPS:
echo 1. Review detailed metrics in work_dirs/premium_gpu_results.json
echo 2. If mIoU >= 80%%, model is production-ready for your website
echo 3. If mIoU < 80%%, consider additional training or data augmentation
echo.
echo For API integration, use the saved model:
echo   work_dirs/premium_gpu_best_model.pth
echo.
echo Production deployment commands:
echo   docker build -t lanesegnet-premium .
echo   docker run -p 8010:8010 --gpus all lanesegnet-premium
echo.
pause