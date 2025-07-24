@echo off
echo ===================================
echo ULTRA-FAST GPU Training - Phase 3.2.5
echo Maximum Speed Optimizations
echo ===================================
echo.
echo SPEED OPTIMIZATIONS ENABLED:
echo - Mixed Precision Training (FP16)
echo - Larger Batch Size (16 vs 8)
echo - More DataLoader Workers (6 vs 2)
echo - cuDNN Benchmark Mode
echo - OneCycle Learning Rate Schedule
echo - Streamlined Model Architecture
echo - Aggressive Early Stopping
echo.
echo EXPECTED PERFORMANCE:
echo - 2-3x faster than standard GPU training
echo - ~1.5-2 hours total (vs 4.5 hours)
echo - Same or better model quality
echo - Target: 70-85%% mIoU in 10-15 epochs
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}'); print(f'Mixed Precision Support: {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7}')"
echo.
pause
echo Starting ULTRA-FAST GPU training...
python scripts/fast_gpu_train.py
echo.
echo ===================================
echo ULTRA-FAST Training completed!
echo Check results with:
echo   python scripts/simple_balanced_monitor.py
echo   python scripts/balanced_eval.py
echo ===================================
pause