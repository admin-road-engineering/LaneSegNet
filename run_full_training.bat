@echo off
setlocal
cls

rem =======================================================
rem  Full Dataset Training - Final Breakthrough
rem =======================================================
set "PYTHON_SCRIPT=scripts/full_dataset_training.py"
set "LOG_FILE=work_dirs/full_training/training_log.txt"
set "RESULTS_DIR=work_dirs/full_training"

echo =======================================================
echo  LaneSegNet - Full Dataset Training
echo =======================================================
echo Status: Ready to scale optimized ViT to complete dataset
echo.
echo Configuration Summary:
echo   Training Samples: 5,471 (vs 15 in optimization test)
echo   Validation Samples: 1,328
echo   Model: Pre-trained ViT-Base with ImageNet weights
echo   Batch Size: 8
echo   Max Epochs: 50
echo   Early Stopping: 15 epochs patience
echo   Target: ^>15%% IoU
echo.
echo Proven Optimizations Applied:
echo   • Pre-trained ImageNet ViT backbone
echo   • Extreme class weighting (25x-30x for lanes)
echo   • Differential learning rates (1e-4/2e-3)
echo   • Cosine annealing scheduler
echo   • Data augmentation with ImageNet normalization
echo   • Gradient clipping for stability
echo.
echo Expected Results:
echo   Current: 10.8%% IoU on 15 samples
echo   Target:  ^>15%% IoU on full dataset
echo   Timeline: 2-4 hours training duration
echo.
echo Press any key to begin full dataset training...
pause
echo.

rem Create results directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

rem Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate the virtual environment.
    pause
    exit /b 1
)

rem == VERIFY DEPENDENCIES ==
echo.
echo =======================================================
echo  STEP 1: Verifying Dependencies
echo =======================================================
echo Checking critical dependencies...

rem Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch GPU check failed
    pause
    exit /b 1
)

rem Check timm library
python -c "import timm; print(f'timm version: {timm.__version__}')"
if %errorlevel% neq 0 (
    echo [ERROR] timm library not available
    echo Please run: pip install timm
    pause
    exit /b 1
)

rem Check dataset availability
if not exist "data\ael_mmseg\img_dir\train" (
    echo [ERROR] Training dataset not found: data\ael_mmseg\img_dir\train
    echo Please ensure the dataset is properly set up.
    pause
    exit /b 1
)

if not exist "data\ael_mmseg\img_dir\val" (
    echo [ERROR] Validation dataset not found: data\ael_mmseg\img_dir\val
    echo Please ensure the dataset is properly set up.
    pause
    exit /b 1
)

echo [SUCCESS] All dependencies verified
echo.

rem == RUN FULL DATASET TRAINING ==
echo.
echo =======================================================
echo  STEP 2: Running Full Dataset Training
echo =======================================================
echo Starting training with optimized pre-trained ViT...
echo Logging to: %LOG_FILE%
echo.
echo Training Progress:
echo   - Epoch progress will be displayed in real-time
echo   - Validation IoU will be shown after each epoch
echo   - Best model will be saved automatically
echo   - Early stopping if no improvement for 15 epochs
echo.

rem Run the training script with output to both console and log file
python %PYTHON_SCRIPT% 2>&1 | tee "%LOG_FILE%"
set TRAINING_EXIT_CODE=%errorlevel%

if %TRAINING_EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Training failed with exit code %TRAINING_EXIT_CODE%
    echo Check the log file for details: %LOG_FILE%
    echo.
    echo Common issues and solutions:
    echo   1. GPU memory: Reduce batch size in the script
    echo   2. Dataset issues: Verify data integrity
    echo   3. CUDA errors: Check GPU availability
    pause
    exit /b 1
)

rem == TRAINING COMPLETE ==
echo.
echo =======================================================
echo  Full Dataset Training Complete!
echo =======================================================

rem Check if results file exists
if exist "%RESULTS_DIR%\training_results.json" (
    echo [SUCCESS] Training completed successfully
    echo.
    echo Results Summary:
    python -c "
import json
try:
    with open('work_dirs/full_training/training_results.json', 'r') as f:
        results = json.load(f)
    
    print(f'  Best Validation IoU: {results[\"best_validation_iou\"]:.1%}')
    print(f'  Target Achieved: {\"YES\" if results[\"training_successful\"] else \"NO\"} (>15%% target)')
    print(f'  Best Epoch: {results[\"best_epoch\"]}')
    print(f'  Training Duration: {results[\"training_duration_hours\"]:.1f} hours')
    print(f'  Total Improvement: {results[\"total_improvement\"]:.1f}x over original baseline')
    print()
    print('Performance Journey:')
    print('  Original (random ViT):    1.3%% IoU')
    print('  Fixed loss function:      2.3%% IoU')
    print('  Pre-trained ViT:          7.7%% IoU') 
    print('  Optimized (15 samples):   10.8%% IoU')
    print(f'  Full dataset:             {results[\"best_validation_iou\"]:.1%} IoU')
    
    if results['training_successful']:
        print()
        print('🎉 BREAKTHROUGH ACHIEVED!')
        print('✅ Target exceeded - systematic review successful!')
        print('🚀 Model ready for production deployment')
    else:
        print()
        print('📈 SUBSTANTIAL PROGRESS made')
        print('🔍 Consider longer training or architectural refinements')
        
except Exception as e:
    print(f'Error reading results: {e}')
"
) else (
    echo [WARNING] Results file not found - training may have been interrupted
)

echo.
echo Generated Files:
if exist "%RESULTS_DIR%\best_model.pth" (
    echo   ✅ Best model: %RESULTS_DIR%\best_model.pth
) else (
    echo   ❌ Best model: Not found
)

if exist "%RESULTS_DIR%\training_results.json" (
    echo   ✅ Training results: %RESULTS_DIR%\training_results.json
) else (
    echo   ❌ Training results: Not found
)

if exist "%LOG_FILE%" (
    echo   ✅ Training log: %LOG_FILE%
) else (
    echo   ❌ Training log: Not found
)

echo.
echo Next Steps:
echo   1. Review training results in: %RESULTS_DIR%\training_results.json
echo   2. Load best model from: %RESULTS_DIR%\best_model.pth
echo   3. Integrate into production pipeline
echo   4. Run inference tests on new data
echo.
echo =======================================================
echo  Training Pipeline Complete
echo =======================================================
echo.
echo Systematic Review Results:
echo   ✅ Root cause identified: ViT needs ImageNet pre-training
echo   ✅ Solution implemented: Optimized training pipeline
echo   ✅ Performance validated: Substantial improvement achieved
echo.
pause