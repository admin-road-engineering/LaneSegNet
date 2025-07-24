@echo off
echo ===================================
echo GPU Test Training - Phase 3.2.5
echo ===================================
echo.
echo Testing GPU training with RTX 3060...
echo - Same DiceFocal Loss implementation
echo - Same model architecture
echo - Limited to 200 samples for quick test
echo - 3 epochs only (~15 minutes)
echo.
echo This will run ALONGSIDE your CPU training
echo to verify GPU performance before switching.
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Environment check:
python --version
echo.
echo Quick CUDA check:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo.
pause
echo Starting GPU test...
python scripts/gpu_test_train.py
echo.
echo ===================================
echo GPU test completed!
echo If successful, you can stop CPU training
echo and restart with full GPU training.
echo ===================================
pause