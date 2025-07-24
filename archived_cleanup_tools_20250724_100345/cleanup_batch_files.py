#!/usr/bin/env python3
"""
Batch File Cleanup - Archive obsolete batch files while preserving core functionality
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Batch File Cleanup")
    print("=" * 40)
    
    # KEEP these batch files (core functionality)
    keep_files = {
        "run_continue_finetuning.bat",  # CURRENT: Main fine-tuning pipeline with fixes
        "run_ssl_pipeline.bat",         # CURRENT: SSL pre-training pipeline  
        "start_server.bat"              # CURRENT: API server startup
    }
    
    # ARCHIVE these batch files (obsolete/superseded)
    archive_files = {
        "run_backup_model_testing.bat",      # Superseded - testing complete
        "run_combined_training.bat",         # Superseded by SSL pipeline
        "run_combined_training_simple.bat",  # Superseded by SSL pipeline
        "run_comprehensive_audit.bat",       # One-time audit complete
        "run_comprehensive_validation.bat",  # One-time validation complete
        "run_data_collection.bat",          # Collection complete (1,100 images)
        "run_finetune_premium.bat",         # Superseded by run_continue_finetuning.bat
        "run_finetune_simple.bat",          # Superseded by current pipeline
        "run_full_dataset_training.bat",    # Superseded (dataset issues resolved)
        "run_full_pipeline.bat",            # Superseded by modular approach
        "run_optimized_premium_training.bat", # Superseded by SSL pipeline
        "run_premium_gpu.bat",              # Superseded by current pipeline
        "run_proper_finetune.bat",          # Superseded by run_continue_finetuning.bat
        "test_models_with_progress.bat"     # Testing utilities - can be archived
    }
    
    # Create archive directory for batch files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archived_batch_files_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    print(f"Archive directory: {archive_dir}")
    print()
    
    # Archive obsolete batch files
    archived_count = 0
    for batch_file in archive_files:
        if Path(batch_file).exists():
            shutil.move(batch_file, archive_dir / batch_file)
            print(f"ARCHIVED: {batch_file}")
            archived_count += 1
        else:
            print(f"NOT FOUND: {batch_file}")
    
    print(f"\nBatch files archived: {archived_count}")
    print("\nKEPT (Core functionality):")
    for kept_file in keep_files:
        if Path(kept_file).exists():
            print(f"  ✓ {kept_file}")
        else:
            print(f"  ✗ {kept_file} (MISSING - CRITICAL)")
    
    # Create manifest
    with open(archive_dir / "BATCH_ARCHIVE_MANIFEST.txt", 'w') as f:
        f.write(f"Batch File Archive - {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ARCHIVED FILES:\n")
        for file in sorted(archive_files):
            f.write(f"- {file}\n")
        f.write(f"\nKEPT FILES (Core functionality):\n")
        for file in sorted(keep_files):
            f.write(f"- {file}\n")
        f.write(f"\nTotal archived: {archived_count} files\n")
    
    print(f"\nManifest created: {archive_dir}/BATCH_ARCHIVE_MANIFEST.txt")
    print("\n" + "=" * 50)
    print("BATCH FILE CLEANUP COMPLETE")
    print("=" * 50)
    print("Root directory now contains only essential batch files:")
    print("- run_continue_finetuning.bat (Main training pipeline)")
    print("- run_ssl_pipeline.bat (SSL pre-training)")  
    print("- start_server.bat (API server)")

if __name__ == "__main__":
    main()