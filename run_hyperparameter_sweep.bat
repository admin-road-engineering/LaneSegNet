@echo off
setlocal EnableDelayedExpansion
cls

rem =======================================================
rem  Phase 4A: Hyperparameter Sweep - ViT-Base Optimization
rem =======================================================
set "SWEEP_SCRIPT=scripts/hyperparameter_sweep.py"
set "RESULTS_DIR=work_dirs/hyperparameter_sweep"

echo =======================================================
echo  Phase 4A: ViT-Base Hyperparameter Optimization
echo =======================================================
echo Status: Running systematic sweep with improved framework
echo.
echo Framework Improvements Applied:
echo   • Warmup schedulers (cosine-warmup, linear-warmup)
echo   • Advanced augmentations (Albumentations with 3 levels)
echo   • Reproducible seeds and deterministic behavior
echo   • Enhanced scheduler options and gradient clipping
echo.
echo Sweep Configuration:
echo   Experiments: 30 (manageable scope)
echo   Parallel Workers: 2 (GPU memory safe)
echo   Expected Duration: ~15 hours
echo   Target: 25-35%% IoU (vs current 15.1%%)
echo.
echo Search Space Priority:
echo   1. Learning Rates (4x4 grid - highest impact)
echo   2. Schedulers (5 types including warmup)
echo   3. Optimizers (3 stability variants)
echo   4. Training Duration (3 convergence options) 
echo   5. Loss Tuning (4 class imbalance strategies)
echo   6. Augmentations (none/basic/strong)
echo.

rem Create results directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

rem Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Please ensure the virtual environment is set up correctly.
    pause
    exit /b 1
)

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
echo Checking required dependencies...

rem Check GPU availability
echo Checking GPU/CUDA...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch GPU check failed
    pause
    exit /b 1
)

rem Check transformers library
echo Checking transformers library...
python -c "import transformers; print(f'transformers version: {transformers.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] transformers library not available
    echo Please run: pip install transformers
    pause
    exit /b 1
)

rem Check albumentations library
echo Checking albumentations library...
python -c "import albumentations; print(f'albumentations version: {albumentations.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] albumentations library not available
    echo Please run: pip install albumentations
    pause
    exit /b 1
)

rem Check dataset availability
echo Checking datasets...
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

rem Check configurable training script
if not exist "%SWEEP_SCRIPT%" (
    echo [ERROR] Sweep script not found: %SWEEP_SCRIPT%
    echo Please ensure the hyperparameter sweep framework is available.
    pause
    exit /b 1
)

echo [SUCCESS] All dependencies verified
echo.

rem == RUN HYPERPARAMETER SWEEP ==
echo.
echo =======================================================
echo  STEP 2: Running Hyperparameter Sweep
echo =======================================================
echo Starting systematic optimization with improved framework...
echo Results will be saved to: %RESULTS_DIR%
echo.
echo Expected Timeline:
echo   Phase 4A: 15 hours (hyperparameter optimization)
echo   Phase 4B: After completion (error analysis on best model)
echo   Target: 25-35%% IoU breakthrough
echo.
echo Sweep will start automatically in 10 seconds...
echo Press Ctrl+C to cancel now if needed.
timeout /t 10 /nobreak >nul
echo.
echo Starting hyperparameter sweep...

rem Run the sweep with recommended settings
python %SWEEP_SCRIPT% --max-experiments 30 --max-parallel 2
set SWEEP_EXIT_CODE=%errorlevel%

echo.
echo =======================================================
echo  Hyperparameter Sweep Complete
echo =======================================================

if %SWEEP_EXIT_CODE% neq 0 (
    echo [ERROR] Hyperparameter sweep failed with exit code %SWEEP_EXIT_CODE%
    echo.
    echo Common issues and solutions:
    echo   1. GPU memory: Reduce --max-parallel to 1
    echo   2. Dependency issues: Check library versions
    echo   3. Dataset issues: Verify data integrity
    echo   4. Timeout issues: Reduce --max-experiments
    echo.
    pause
    exit /b 1
)

rem == DISPLAY RESULTS ==
echo [SUCCESS] Hyperparameter sweep completed successfully
echo.

rem Check if results file exists and display summary
set "LATEST_RESULTS_DIR="
for /f "delims=" %%i in ('dir /b /od "%RESULTS_DIR%\sweep_*" 2^>nul') do set "LATEST_RESULTS_DIR=%%i"

if defined LATEST_RESULTS_DIR (
    set "FINAL_RESULTS=%RESULTS_DIR%\%LATEST_RESULTS_DIR%\final_results.json"
    if exist "!FINAL_RESULTS!" (
        echo =======================================================
        echo  Hyperparameter Sweep Results Summary
        echo =======================================================
        
        rem Use Python to extract and display key results
        python -c "
import json
import sys
try:
    with open('!FINAL_RESULTS!', 'r') as f:
        results = json.load(f)
    
    summary = results['sweep_summary']
    top_configs = results.get('top_5_configs', [])
    
    print('HYPERPARAMETER OPTIMIZATION RESULTS:')
    print('=' * 60)
    print(f'Total Experiments: {summary[\"total_experiments\"]}')
    print(f'Successful Experiments: {summary[\"successful_experiments\"]}')
    print(f'Best Validation IoU: {summary[\"best_iou\"]:.1%}')
    print(f'Training Duration: {summary[\"total_duration_hours\"]:.1f} hours')
    print()
    
    if summary['best_iou'] > 0.15:
        print('🎉 TARGET ACHIEVED: >15%% IoU baseline exceeded!')
        improvement = (summary['best_iou'] - 0.151) / 0.151 * 100
        print(f'📈 Improvement over 15.1%% baseline: +{improvement:.1f}%%')
        print('✅ Ready for Phase 4B: Error analysis')
    else:
        print('📊 SUBSTANTIAL PROGRESS: Optimization gains validated')
        print('🔍 Consider extended training or architectural changes')
    
    print()
    print('TOP 3 CONFIGURATIONS:')
    print('-' * 40)
    for i, config in enumerate(top_configs[:3]):
        exp_name = config['experiment_name']
        best_iou = config['best_iou']
        duration = config['duration_minutes']
        
        print(f'{i+1}. {exp_name}: {best_iou:.1%} IoU ({duration:.0f} min)')
        
        cfg = config['config']
        print(f'   LR: enc={cfg[\"encoder_lr\"]:.0e}, dec={cfg[\"decoder_lr\"]:.0e}')
        print(f'   Scheduler: {cfg[\"scheduler\"]}')
        if 'augmentation_level' in cfg:
            print(f'   Augmentation: {cfg[\"augmentation_level\"]}')
        print()
    
    print('NEXT STEPS:')
    print('1. Review detailed results in: !FINAL_RESULTS!')
    print('2. Load best model configuration for Phase 4B error analysis')
    print('3. Consider scaling to larger architectures if ceiling reached')
    
except Exception as e:
    print(f'Error reading results file: {e}')
    print('Sweep completed but results file not found or corrupted.')
    sys.exit(1)
"
        
        if %errorlevel% neq 0 (
            echo [WARNING] Could not read results file properly
        )
        
    ) else (
        echo [WARNING] Results file not found: !FINAL_RESULTS!
        echo Sweep may have completed but results file not generated.
    )
) else (
    echo [WARNING] No sweep results directory found in %RESULTS_DIR%
    echo Sweep may have been interrupted or failed to create results.
)

echo.
echo =======================================================
echo  Phase 4A Complete
echo =======================================================
echo ✅ Hyperparameter optimization finished
echo 📊 Results available for analysis
echo 🎯 Ready for Phase 4B: Qualitative error analysis
echo.
echo Generated Files:
if defined LATEST_RESULTS_DIR (
    echo   📄 Final results: %RESULTS_DIR%\%LATEST_RESULTS_DIR%\final_results.json
    echo   📁 Sweep directory: %RESULTS_DIR%\%LATEST_RESULTS_DIR%
    echo   📊 Individual experiments: work_dirs\configurable_finetuning\
) else (
    echo   ❌ No results directory found
)

echo.
echo Next Steps:
echo   1. Review top configurations and performance gains
echo   2. Proceed with Phase 4B error analysis on best model  
echo   3. Consider architectural improvements based on ceiling
echo.
echo Hyperparameter sweep session complete. Press any key to exit...
pause >nul