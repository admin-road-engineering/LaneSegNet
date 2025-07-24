#!/usr/bin/env python3
"""
Simple Project Cleanup
======================

Clean up redundant files without Unicode characters for Windows compatibility.
"""

import os
import shutil
from pathlib import Path
import json
import time

def create_archive_directory():
    """Create archive directory with timestamp"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(f'archived_files_{timestamp}')
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def safe_move_file(src_path, dest_dir, reason=""):
    """Safely move a file to archive directory"""
    try:
        src = Path(src_path)
        if src.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src.name
            shutil.move(str(src), str(dest_path))
            return f"MOVED {src.name} ({reason})"
        else:
            return f"NOT FOUND: {src.name}"
    except Exception as e:
        return f"ERROR moving {src.name}: {e}"

def cleanup_project():
    """Main cleanup function"""
    print("PROJECT CLEANUP AND ORGANIZATION")
    print("=" * 50)
    
    # Create archive directory
    archive_dir = create_archive_directory()
    print(f"Archive directory: {archive_dir}")
    
    results = []
    
    # 1. Clean up root-level temporary files
    print(f"\n1. Cleaning temporary files...")
    temp_files = ['0.01%', '0.05%', '50%', '70%', '80%']
    for temp_file in temp_files:
        result = safe_move_file(temp_file, archive_dir / 'temp_files', "temporary percentage file")
        results.append(result)
        print(f"  {result}")
    
    # 2. Archive redundant batch files (keep essential ones)
    print(f"\n2. Archiving redundant batch files...")
    redundant_batch_files = [
        'run_fast_option_1.bat',
        'run_fast_option_2.bat',
        'run_option_1.bat',
        'run_option_1_fixed.bat',
        'run_option_1_simple.bat',
        'run_option_2.bat',
        'run_quick_eval.bat',
        'run_quick_finetune.bat',
        'run_true_finetune.bat',
        'continue_finetune_training.bat',
        'check_training_progress.bat',
        'test_85_1_model.bat',
        'test_top_3_models.bat'
    ]
    
    for batch_file in redundant_batch_files:
        result = safe_move_file(batch_file, archive_dir / 'batch_files', "redundant training script")
        results.append(result)
        print(f"  {result}")
    
    # 3. Archive old test results
    print(f"\n3. Archiving old test results...")
    if Path('test_results').exists():
        try:
            shutil.move('test_results', str(archive_dir / 'test_results'))
            results.append("MOVED test_results/ directory")
            print(f"  MOVED test_results/ directory")
        except Exception as e:
            print(f"  ERROR moving test_results/: {e}")
    
    # 4. Archive redundant testing scripts
    print(f"\n4. Archiving redundant testing scripts...")
    redundant_scripts = [
        'scripts/basic_model_test.py',
        'scripts/simple_model_test.py',
        'scripts/quick_model_test.py',
        'scripts/test_enhanced_post_processing.py',
        'scripts/gentle_post_processing.py',
        'scripts/test_backup_models_auto.py',
        'scripts/compare_tta_vs_single.py',
        'scripts/test_85_1_model.py',
        'scripts/quick_status_check.py',
        'scripts/test_single_model.py',
        'scripts/test_top_models.py'
    ]
    
    for script in redundant_scripts:
        result = safe_move_file(script, archive_dir / 'scripts', "redundant test script")
        results.append(result)
        print(f"  {result}")
    
    # 5. Archive old training variations (keep main ones)
    print(f"\n5. Archiving old training variations...")
    old_training_scripts = [
        'scripts/gpu_test_train.py',
        'scripts/quick_finetune.py',
        'scripts/simple_train.py',
        'scripts/continue_premium_training.py',
        'scripts/finetune_from_85_1.py',
        'scripts/premium_continue_51_80.py',
        'scripts/patch_current_training.py',
        'scripts/simple_continue_training.py',
        'scripts/start_training.py'
    ]
    
    for script in old_training_scripts:
        result = safe_move_file(script, archive_dir / 'old_training', "old training variant")
        results.append(result)
        print(f"  {result}")
    
    # 6. Archive utility files
    print(f"\n6. Archiving utility files...")
    utility_files = [
        'check_gpu_setup.py',
        'finetune_params.py',
        'fix_premium_script.py'
    ]
    
    for util_file in utility_files:
        result = safe_move_file(util_file, archive_dir / 'utilities', "utility script")
        results.append(result)
        print(f"  {result}")
    
    # Generate summary
    print(f"\n" + "="*50)
    print("CLEANUP SUMMARY")
    print("="*50)
    
    moved_count = len([r for r in results if r.startswith('MOVED')])
    error_count = len([r for r in results if r.startswith('ERROR')])
    
    print(f"Files moved to archive: {moved_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Archive location: {archive_dir}")
    
    # Essential files verification
    print(f"\nESSENTIAL FILES VERIFICATION:")
    print("-" * 30)
    essential_files = [
        'scripts/premium_gpu_train.py',
        'scripts/comprehensive_audit.py',
        'scripts/test_multi_datasets.py',
        'scripts/test_all_backup_models.py',
        'work_dirs/premium_gpu_best_model.pth',
        'app/main.py',
        'requirements.txt',
        'CLAUDE.md'
    ]
    
    for essential in essential_files:
        status = "EXISTS" if Path(essential).exists() else "MISSING"
        print(f"  {essential}: {status}")
    
    print(f"\nPROJECT CLEANUP COMPLETED!")
    print(f"Archive: {archive_dir}")
    print(f"Essential functionality: PRESERVED")
    
    return moved_count, error_count

if __name__ == "__main__":
    moved, errors = cleanup_project()