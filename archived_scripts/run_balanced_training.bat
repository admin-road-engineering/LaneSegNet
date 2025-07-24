@echo off
echo ===================================
echo Phase 3.2.5: Balanced Training
echo DiceFocal Loss for Class Imbalance
echo ===================================
echo.
echo Starting balanced training with research-proven solution...
echo - DiceFocal Loss compound loss function
echo - Class weights: [0.1, 5.0, 5.0, 3.0]
echo - Optimized architecture (target: 10-20MB)
echo - Target: 70-85%% mIoU with balanced classes
echo.
echo Training will run for up to 25 epochs with early stopping
echo Expected training time: 2-4 hours
echo.
pause
echo Starting training...
python scripts/balanced_train.py
echo.
echo ===================================
echo Training completed!
echo Check results with:
echo   python scripts/simple_balanced_monitor.py
echo   python scripts/balanced_eval.py
echo ===================================
pause