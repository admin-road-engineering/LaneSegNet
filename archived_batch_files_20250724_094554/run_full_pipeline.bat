@echo off
setlocal
cls

rem =======================================================
rem  Pipeline Configuration
rem =======================================================
set "SSL_EPOCHS=50"
set "SSL_BATCH_SIZE=16"
set "SSL_SAVE_DIR=work_dirs/mae_pretraining"

set "FINETUNE_EPOCHS=100"
set "FINETUNE_BATCH_SIZE=4"
set "FINETUNE_SAVE_DIR=work_dirs/finetuning"

set "BEST_SSL_MODEL=%SSL_SAVE_DIR%/mae_best_model.pth"
set "BEST_FINETUNED_MODEL=%FINETUNE_SAVE_DIR%/finetuned_best_model.pth"

echo =======================================================
echo  LaneSegNet - Full Advanced Training Pipeline
echo =======================================================
echo This script will execute the entire advanced pipeline:
echo 1. Collect all unlabeled data.
echo 2. Run Self-Supervised Pre-training (MAE).
echo 3. Fine-tune the final model with the pre-trained encoder and OHEM loss.
echo.
echo Configuration:
echo   SSL Epochs: %SSL_EPOCHS%, Batch Size: %SSL_BATCH_SIZE%
echo   Fine-tune Epochs: %FINETUNE_EPOCHS%, Batch Size: %FINETUNE_BATCH_SIZE%
echo.
echo Press any key to begin the full process...
pause
echo.

rem Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate the virtual environment.
    pause
    exit /b 1
)

rem == STEP 1: DATA COLLECTION ==
echo.
echo =======================================================
echo  STEP 1: Running Unlabeled Data Collection
echo =======================================================
call run_data_collection.bat
if %errorlevel% neq 0 (
    echo [ERROR] Data collection failed. Aborting pipeline.
    pause
    exit /b 1
)
echo [SUCCESS] Data collection complete.
echo.

rem == STEP 2: SSL PRE-TRAINING ==
echo.
echo =======================================================
echo  STEP 2: Running Self-Supervised Pre-training
echo =======================================================
echo Running command: python scripts/run_ssl_pretraining.py --epochs %SSL_EPOCHS% --batch-size %SSL_BATCH_SIZE% --save-dir "%SSL_SAVE_DIR%"
python scripts/run_ssl_pretraining.py --epochs %SSL_EPOCHS% --batch-size %SSL_BATCH_SIZE% --save-dir "%SSL_SAVE_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] SSL Pre-training failed. Aborting pipeline.
    pause
    exit /b 1
)
echo [SUCCESS] SSL Pre-training complete. Best encoder saved to %BEST_SSL_MODEL%
echo.

rem == STEP 3: FINAL MODEL FINE-TUNING ==
echo.
echo =======================================================
echo  STEP 3: Running Final Model Fine-Tuning
echo =======================================================
echo Integrating pre-trained encoder with OHEM loss for production model...
echo Running command: python scripts/run_finetuning.py --encoder-weights "%BEST_SSL_MODEL%" --use-ohem --epochs %FINETUNE_EPOCHS% --batch-size %FINETUNE_BATCH_SIZE% --save-dir "%FINETUNE_SAVE_DIR%"
python scripts/run_finetuning.py --encoder-weights "%BEST_SSL_MODEL%" --use-ohem --epochs %FINETUNE_EPOCHS% --batch-size %FINETUNE_BATCH_SIZE% --save-dir "%FINETUNE_SAVE_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Fine-tuning failed. Aborting pipeline.
    pause
    exit /b 1
)
echo [SUCCESS] Fine-tuning complete. Production model ready!
echo.

rem == PIPELINE COMPLETE ==
echo.
echo =======================================================
echo  Pipeline Orchestration Complete!
echo =======================================================
echo [SUCCESS] Data Collection: Complete
echo [SUCCESS] SSL Pre-training: Complete  
echo [SUCCESS] Final Fine-tuning: Complete
echo.
echo Expected Performance Gains:
echo   • SSL Pre-training: +5-15%% mIoU improvement
echo   • OHEM Loss Integration: +2-5%% additional mIoU
echo   • Combined Expected: +7-20%% total improvement
echo.
echo Outputs Created:
echo   • Pre-trained encoder: %BEST_SSL_MODEL%
echo   • Production model: %BEST_FINETUNED_MODEL%
echo   • Training logs: %SSL_SAVE_DIR%/training_log.json and %FINETUNE_SAVE_DIR%/training_log.json
echo.
echo Next Steps:
echo   • Test production model with: python scripts/enhanced_post_processing.py
echo   • Deploy via knowledge distillation: python scripts/knowledge_distillation.py
echo   • Integration ready for road-engineering frontend
echo.
echo Pipeline execution complete! Press any key to exit...
pause