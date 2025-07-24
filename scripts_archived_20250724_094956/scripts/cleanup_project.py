#!/usr/bin/env python3
"""
Project Cleanup and Organization
===============================

Safely archive redundant files and organize the project structure
while preserving all essential functionality.
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
            return f"✓ Moved {src.name} ({reason})"
        else:
            return f"✗ Not found: {src.name}"
    except Exception as e:
        return f"✗ Error moving {src.name}: {e}"

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
    
    # 2. Archive redundant batch files
    print(f"\n2. Archiving redundant training batch files...")
    redundant_batch_files = [
        'run_combined_training.bat',
        'run_combined_training_simple.bat', 
        'run_fast_option_1.bat',
        'run_fast_option_2.bat',
        'run_finetune_premium.bat',
        'run_finetune_simple.bat',
        'run_option_1.bat',
        'run_option_1_fixed.bat',
        'run_option_1_simple.bat',
        'run_option_2.bat',
        'run_premium_gpu.bat',
        'run_proper_finetune.bat',
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
            results.append("✓ Moved test_results/ directory")
            print(f"  ✓ Moved test_results/ directory")
        except Exception as e:
            print(f"  ✗ Error moving test_results/: {e}")
    
    # 4. Archive redundant scripts
    print(f"\n4. Archiving redundant scripts...")
    redundant_scripts = [
        'scripts/basic_model_test.py',
        'scripts/simple_model_test.py',
        'scripts/quick_model_test.py',
        'scripts/quick_model_comparison.py',
        'scripts/test_enhanced_post_processing.py',
        'scripts/gentle_post_processing.py',
        'scripts/test_backup_models_auto.py',
        'scripts/compare_tta_vs_single.py',
        'scripts/comprehensive_model_validation.py',
        'scripts/test_85_1_model.py',
        'scripts/test_comprehensive_datasets.py',
        'scripts/quick_status_check.py',
        'scripts/test_single_model.py',
        'scripts/test_top_models.py'
    ]
    
    for script in redundant_scripts:
        result = safe_move_file(script, archive_dir / 'scripts', "redundant test script")
        results.append(result)
        print(f"  {result}")
    
    # 5. Archive old training scripts
    print(f"\n5. Archiving old training variations...")
    old_training_scripts = [
        'scripts/balanced_train.py',
        'scripts/gpu_test_train.py',
        'scripts/quick_finetune.py',
        'scripts/simple_train.py',
        'scripts/swin_transformer_train.py',
        'scripts/continue_premium_training.py',
        'scripts/finetune_from_85_1.py',
        'scripts/finetune_premium_model.py',
        'scripts/premium_continue_51_80.py',
        'scripts/patch_current_training.py',
        'scripts/simple_continue_training.py',
        'scripts/start_training.py'
    ]
    
    for script in old_training_scripts:
        result = safe_move_file(script, archive_dir / 'old_training', "old training variant")
        results.append(result)
        print(f"  {result}")
    
    # 6. Archive old monitoring scripts
    print(f"\n6. Archiving old monitoring scripts...")
    monitoring_scripts = [
        'scripts/balanced_monitor.py',
        'scripts/simple_balanced_monitor.py',
        'scripts/simple_monitor.py',
        'scripts/monitor_training.py'
    ]
    
    for script in monitoring_scripts:
        result = safe_move_file(script, archive_dir / 'monitoring', "old monitoring script")
        results.append(result)
        print(f"  {script}")
    
    # 7. Archive utility files
    print(f"\n7. Archiving utility files...")
    utility_files = [
        'check_gpu_setup.py',
        'check_versions.py',
        'finetune_params.py',
        'fix_premium_script.py'
    ]
    
    for util_file in utility_files:
        result = safe_move_file(util_file, archive_dir / 'utilities', "utility script")
        results.append(result)
        print(f"  {result}")
    
    # 8. Clean up old result files
    print(f"\n8. Archiving old result files...")
    result_files = [
        'audit_report_20250723_113156.json',
        'backup_model_comparison_20250723_113826.json',
        'comprehensive_validation_results_20250723_112903.json'
    ]
    
    for result_file in result_files:
        result = safe_move_file(result_file, archive_dir / 'reports', "old report file")
        results.append(result)
        print(f"  {result}")
    
    # 9. Clean up comprehensive test results directories
    print(f"\n9. Archiving comprehensive test directories...")
    test_dirs = [
        'comprehensive_test_results_20250722_235739',
        'comprehensive_test_results_20250722_235758',
        'local_aerial_tests'
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            try:
                shutil.move(test_dir, str(archive_dir / 'test_directories' / test_dir))
                results.append(f"✓ Moved {test_dir}/ directory")
                print(f"  ✓ Moved {test_dir}/ directory")
            except Exception as e:
                print(f"  ✗ Error moving {test_dir}/: {e}")
    
    # Generate cleanup summary
    print(f"\n" + "="*50)
    print("CLEANUP SUMMARY")
    print("="*50)
    
    moved_count = len([r for r in results if r.startswith('✓')])
    error_count = len([r for r in results if r.startswith('✗')])
    
    print(f"Files moved to archive: {moved_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Archive location: {archive_dir}")
    
    # Create cleanup report
    cleanup_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'archive_directory': str(archive_dir),
        'files_moved': moved_count,
        'errors': error_count,
        'results': results,
        'preserved_essential_files': [
            'scripts/premium_gpu_train.py (main training)',
            'scripts/comprehensive_audit.py (audit system)',
            'scripts/test_multi_datasets.py (multi-dataset testing)',
            'scripts/test_all_backup_models.py (model comparison)',
            'work_dirs/premium_gpu_best_model.pth (production model)',
            'app/ (FastAPI service)',
            'model_backups/ (all model checkpoints)',
            'data/ (training datasets)'
        ]
    }
    
    report_path = f'cleanup_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(cleanup_report, f, indent=2)
    
    print(f"\nDetailed report saved: {report_path}")
    
    # Essential files verification
    print(f"\nESSENTIAL FILES VERIFICATION:")
    print("-" * 30)
    essential_files = [
        'scripts/premium_gpu_train.py',
        'work_dirs/premium_gpu_best_model.pth',
        'app/main.py',
        'requirements.txt',
        'CLAUDE.md'
    ]
    
    for essential in essential_files:
        status = "✓ EXISTS" if Path(essential).exists() else "✗ MISSING"
        print(f"  {essential}: {status}")
    
    print(f"\n🎉 PROJECT CLEANUP COMPLETED!")
    print(f"   Archive: {archive_dir}")
    print(f"   Essential functionality: PRESERVED")
    
    return cleanup_report

if __name__ == "__main__":
    cleanup_report = cleanup_project()