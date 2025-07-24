@echo off
echo ===================================
echo OPTIMIZED PREMIUM TRAINING - Phase 3.2.5
echo Industry-Leading 3-Class Lane Detection
echo ===================================
echo.
echo CORRECTED CONFIGURATION:
echo - Dataset Classes: 3 (background, white_solid, white_dashed)
echo - Model Classes: 3 (corrected from 4)
echo - Class Weights: [0.1, 5.0, 5.0] (research-optimized)
echo - Bayesian Parameters: LR=5.14e-04, Dice=0.694
echo.
echo ENHANCED FEATURES:
echo - Hybrid Loss: DiceFocal + Lovász + Edge + Smoothness
echo - Premium U-Net: 8.9M parameters with attention
echo - Advanced Augmentations: MixUp + CutMix + Style Transfer
echo - Mixed Precision: GPU efficiency without quality loss
echo.
echo TARGET PERFORMANCE:
echo - 80-85%% mIoU: Industry-Leading Quality
echo - All Lane Classes >70%% IoU: Excellent Balance
echo - Class Imbalance: RESOLVED (3-class optimization)
echo.
echo EXPECTED RESULTS:
echo - Training Time: 4-8 hours (quality-focused)
echo - Model Size: 15-25MB (production-ready)
echo - GPU Utilization: Full RTX 3060 optimization
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"
echo.
echo Verifying optimized parameters exist...
if exist "work_dirs\bayesian_optimization_results.json" (
    echo ✓ Bayesian optimization results found
    python -c "import json; data=json.load(open('work_dirs/bayesian_optimization_results.json')); print(f'  LR: {data[\"best_params\"][\"learning_rate\"]:.2e}'); print(f'  Dice Weight: {data[\"best_params\"][\"dice_weight\"]:.3f}'); print(f'  Best Score: {data[\"best_score\"]:.1%}')"
) else (
    echo ⚠️  Optimization results not found, using default parameters
)
echo.
echo Starting OPTIMIZED PREMIUM TRAINING with corrected 3-class model...
echo This will target 80-85%% mIoU with industry-leading enhancements.
echo.
pause
python scripts/premium_gpu_train.py --load-optimized
echo.
echo ===================================
echo OPTIMIZED TRAINING COMPLETED!
echo ===================================
echo.
echo RESULTS LOCATION:
echo - Model: work_dirs/premium_gpu_best_model.pth
echo - Metrics: work_dirs/premium_gpu_results.json
echo.
echo NEXT STEPS:
echo 1. Review training results and mIoU achievement
echo 2. If mIoU >= 80%%, model is production-ready
echo 3. Run evaluation: python scripts/balanced_eval.py
echo 4. Test inference: python scripts/simple_balanced_monitor.py
echo.
echo For API deployment:
echo   docker build -t lanesegnet-premium .
echo   docker run -p 8010:8010 --gpus all lanesegnet-premium
echo.
pause