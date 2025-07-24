#!/usr/bin/env python3
"""
Quick Project Cleanup - Archive obsolete files safely
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Project Cleanup")
    print("=" * 40)
    
    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archived_files_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    # Files to archive (obsolete/completed)
    to_archive = [
        "ADVANCED_TECHNIQUES_PLAN.md",
        "OVERFITTING_FIX_PLAN.md", 
        "PIPELINE_RUN_REPORT_TEMPLATE.md",
        "REVISED_ADVANCED_TECHNIQUES_PLAN.md",
        "audit_report_20250723_113156.json",
        "backup_model_comparison_20250723_113826.json",
        "comprehensive_validation_results_20250723_112903.json",
        "full_dataset_report.json",
        "debug_iou_issue.py"
    ]
    
    # Archive files
    archived_count = 0
    for file in to_archive:
        if Path(file).exists():
            shutil.move(file, archive_dir / file)
            print(f"Archived: {file}")
            archived_count += 1
    
    # Archive test result directories
    test_dirs = [
        "comprehensive_test_results_20250722_235739",
        "comprehensive_test_results_20250722_235758",
        "local_aerial_tests"
    ]
    
    for directory in test_dirs:
        if Path(directory).exists():
            shutil.move(directory, archive_dir / directory)
            print(f"Archived directory: {directory}")
            archived_count += 1
    
    print(f"\nCleanup complete: {archived_count} items archived")
    print(f"Archive created: {archive_dir}")
    print("\nCore functionality preserved:")
    print("- SSL pipeline: run_ssl_pipeline.bat")
    print("- Fine-tuning: run_continue_finetuning.bat") 
    print("- API service: app/")
    print("- Working data: data/ael_mmseg/")
    print("- Tests: tests/")

if __name__ == "__main__":
    main()