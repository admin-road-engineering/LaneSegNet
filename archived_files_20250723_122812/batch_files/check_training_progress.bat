@echo off
setlocal EnableDelayedExpansion
echo ===================================
echo TRAINING PROGRESS CHECKER
echo ===================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

echo Checking for active Python training processes...
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE | findstr python.exe
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Python training processes found
) else (
    echo WARNING: No Python training processes detected
)
echo.

echo Checking latest model backups...
if exist "model_backups\" (
    echo Latest backup directories:
    dir "model_backups\" /AD /O-D | findstr "combined"
    if %ERRORLEVEL% NEQ 0 (
        echo No combined training backups found yet
        echo Latest backup of any type:
        dir "model_backups\" /AD /O-D | head -5
    )
) else (
    echo No model_backups directory found
)
echo.

echo Checking work_dirs for combined models...
if exist "work_dirs\combined_*.pth" (
    echo Combined models found:
    dir "work_dirs\combined_*.pth" /O-D
    echo.
    echo Latest combined model details:
    python -c "
import torch
from pathlib import Path
models = list(Path('work_dirs').glob('combined_*.pth'))
if models:
    latest = sorted(models, key=lambda x: x.stat().st_mtime)[-1]
    checkpoint = torch.load(latest, map_location='cpu')
    print(f'Model: {latest.name}')
    print(f'Epoch: {checkpoint.get(\"epoch\", \"unknown\")}')
    print(f'mIoU: {checkpoint.get(\"best_miou\", 0)*100:.1f}%%')
    print(f'Approach: {checkpoint.get(\"approach\", \"unknown\")}')
else:
    print('No combined models found')
"
) else (
    echo No combined models found in work_dirs
)
echo.

echo Checking recent file activity...
python -c "
import os
from datetime import datetime, timedelta
from pathlib import Path

now = datetime.now()
recent_files = []

# Check work_dirs
work_dir = Path('work_dirs')
if work_dir.exists():
    for file in work_dir.iterdir():
        if file.is_file():
            mod_time = datetime.fromtimestamp(file.stat().st_mtime)
            if mod_time > now - timedelta(minutes=30):
                recent_files.append((str(file), mod_time))

# Check model_backups
backup_dir = Path('model_backups')
if backup_dir.exists():
    for folder in backup_dir.iterdir():
        if folder.is_dir() and 'combined' in folder.name:
            mod_time = datetime.fromtimestamp(folder.stat().st_mtime)
            if mod_time > now - timedelta(minutes=30):
                recent_files.append((str(folder), mod_time))

if recent_files:
    print('Recent activity (last 30 minutes):')
    for file, time in sorted(recent_files, key=lambda x: x[1], reverse=True):
        print(f'  {file}: {time.strftime(\"%H:%M:%S\")}')
else:
    print('No recent training activity detected')
"
echo.

echo GPU Status:
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.1f}GB') if torch.cuda.is_available() else None"
echo.

echo ===================================
echo PROGRESS CHECK COMPLETE
echo ===================================
echo.
pause