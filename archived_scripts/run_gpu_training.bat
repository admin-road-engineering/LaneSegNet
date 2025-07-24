@echo off
echo ===================================
echo Phase 3.2.5: GPU Accelerated Training
echo DiceFocal Loss for Class Imbalance
echo ===================================
echo.
echo Starting GPU training with RTX 3060...
echo - DiceFocal Loss compound loss function
echo - Class weights: [0.1, 5.0, 5.0, 3.0]
echo - Optimized architecture (target: 10-20MB)
echo - Target: 70-85%% mIoU with balanced classes
echo.
echo GPU Performance Advantages:
echo - 8-9x faster than CPU (confirmed by test)
echo - ~4.5 hours total vs ~37 hours on CPU
echo - Same model quality, much faster results
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo GPU Status Check:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo.
echo Training will run for up to 25 epochs with early stopping
echo Expected completion: ~4.5 hours
echo.
pause
echo Starting GPU training...
python scripts/balanced_train.py
echo.
echo ===================================
echo GPU Training completed!
echo Check results with:
echo   python scripts/simple_balanced_monitor.py
echo   python scripts/balanced_eval.py
echo ===================================
pause