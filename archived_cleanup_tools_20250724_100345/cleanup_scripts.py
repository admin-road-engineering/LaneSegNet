#!/usr/bin/env python3
"""
Scripts Directory Cleanup - Archive obsolete training scripts while preserving core functionality
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Scripts Directory Cleanup")
    print("=" * 45)
    
    # KEEP these scripts (core functionality)
    keep_scripts = {
        # Current production pipeline
        "run_finetuning.py",              # CURRENT: Main fine-tuning with SSL + OHEM
        "ssl_pretraining.py",             # CURRENT: SSL pre-training (MAE)
        "run_ssl_pretraining.py",         # CURRENT: SSL pre-training runner
        
        # Core production components
        "ohem_loss.py",                   # CURRENT: Advanced loss function
        "enhanced_post_processing.py",    # CURRENT: Production post-processing
        "knowledge_distillation.py",     # CURRENT: Model compression
        "test_time_augmentation.py",     # CURRENT: Production accuracy enhancement
        
        # Testing and monitoring
        "run_tests.py",                   # CURRENT: Testing framework
        "run_tests.bat",                  # CURRENT: Testing batch script
        "monitor_training.py",            # CURRENT: Training monitoring
        
        # Data collection (completed but may be needed)
        "collect_unlabeled_data.py",      # CURRENT: Data collection infrastructure
        "consolidate_unlabeled_data.py",  # CURRENT: Data consolidation
        
        # Utilities that may be needed
        "create_ael_masks.py",            # UTILITY: Mask generation
        "comprehensive_model_validation.py"  # UTILITY: Model validation
    }
    
    # ARCHIVE these scripts (obsolete/superseded)
    archive_scripts = {
        # Obsolete training approaches
        "advanced_augment.py",            # Superseded by current pipeline
        "analyze_miou_discrepancy.py",    # One-time analysis complete
        "balanced_eval.py",               # Superseded by current evaluation
        "balanced_monitor.py",            # Superseded by monitor_training.py
        "balanced_train.py",              # Superseded by current pipeline
        "bayesian_tuner.py",              # Superseded by current approach
        "check_collection_status.py",     # Collection complete
        "cleanup_project.py",             # Old cleanup script
        "combined_dataset_train.py",      # Superseded by SSL pipeline
        "comprehensive_audit.py",         # One-time audit complete
        "create_finetune_data.py",        # Superseded by current data pipeline
        "download_osm_tiles.py",          # Collection complete
        "download_skyscapes.py",          # Collection complete
        "evaluate_baseline.py",           # Evaluation complete
        "evaluate_final.py",              # Evaluation complete
        "fast_gpu_train.py",              # Superseded by current pipeline
        "finetune_premium_model.py",      # Superseded by run_finetuning.py
        "fix_dataset_split.py",           # Data issue resolved
        "full_dataset_premium_training.py", # Superseded by current pipeline
        "generate_carla_aerial.py",       # Data generation complete
        "investigate_test_annotations.py", # Investigation complete
        "optimized_premium_training.py",  # Superseded by SSL pipeline
        "premium_gpu_train.py",           # Superseded by current pipeline
        "prepare_combined_datasets.py",   # Data preparation complete
        "prepare_full_dataset.py",        # Data preparation complete
        "prepare_training_data.py",       # Data preparation complete
        "project_cleanup_current.py",     # Cleanup script (can archive)
        "proper_finetune.py",             # Superseded by run_finetuning.py
        "quick_annotation_check.py",      # Check complete
        "quick_eval.py",                  # Evaluation complete
        "quick_holdout_test.py",          # Testing complete
        "quick_model_comparison.py",      # Comparison complete
        "quick_osm_test.py",              # Testing complete
        "quick_training_setup.py",        # Setup complete
        "regenerate_test_masks.py",       # Mask generation complete
        "simple_balanced_monitor.py",     # Superseded by monitor_training.py
        "simple_cleanup.py",              # Old cleanup script
        "simple_monitor.py",              # Superseded by monitor_training.py
        "swin_transformer_train.py",      # Superseded by current pipeline
        "test_all_backup_models.py",      # Testing complete
        "test_comprehensive_datasets.py", # Testing complete
        "test_current_model_holdout.py",  # Testing complete
        "test_multi_datasets.py",         # Testing complete
        "test_on_training_data.py",       # Testing complete
        "transform_cityscapes_aerial.py"  # Data transformation complete
    }
    
    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"scripts_archived_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    print(f"Archive directory: {archive_dir}")
    print()
    
    # Move to scripts directory
    scripts_path = Path("scripts")
    if not scripts_path.exists():
        print("ERROR: scripts/ directory not found")
        return
    
    os.chdir(scripts_path)
    
    archived_count = 0
    
    # Archive obsolete scripts
    for script in archive_scripts:
        if Path(script).exists():
            destination = Path("..") / archive_dir / "scripts" / script
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(script), str(destination))
            print(f"ARCHIVED: {script}")
            archived_count += 1
        else:
            print(f"NOT FOUND: {script}")
    
    # Check for data directory and archive it if exists
    if Path("data").exists():
        destination = Path("..") / archive_dir / "scripts" / "data"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move("data", str(destination))
        print(f"ARCHIVED: data/ directory")
        archived_count += 1
    
    # Go back to root
    os.chdir("..")
    
    print(f"\nScripts archived: {archived_count}")
    
    # Check what's left in scripts/
    remaining_files = []
    if scripts_path.exists():
        for item in scripts_path.iterdir():
            if item.is_file():
                remaining_files.append(item.name)
    
    print(f"\nREMAINING in scripts/:")
    if remaining_files:
        for file in sorted(remaining_files):
            if file in keep_scripts:
                print(f"  KEPT: {file}")
            else:
                print(f"  UNEXPECTED: {file}")
    else:
        print("  (Directory is empty)")
    
    print(f"\nEXPECTED core scripts: {len(keep_scripts)}")
    print(f"ACTUAL remaining scripts: {len(remaining_files)}")
    
    # Create manifest
    manifest_path = archive_dir / "SCRIPTS_ARCHIVE_MANIFEST.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Scripts Archive - {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ARCHIVED SCRIPTS:\n")
        for script in sorted(archive_scripts):
            f.write(f"- {script}\n")
        f.write(f"\nKEPT SCRIPTS (Core functionality):\n")
        for script in sorted(keep_scripts):
            f.write(f"- {script}\n")
        f.write(f"\nTotal archived: {archived_count} items\n")
        f.write(f"\nPURPOSE:\n")
        f.write("Archive obsolete training scripts and completed utilities\n")
        f.write("while preserving current SSL + fine-tuning pipeline.\n")
    
    print(f"\nManifest created: {manifest_path}")
    
    print("\n" + "=" * 50)
    print("SCRIPTS CLEANUP COMPLETE")
    print("=" * 50)
    print("scripts/ directory now contains only current production scripts:")
    print("- SSL pre-training pipeline")
    print("- Fine-tuning with OHEM loss")
    print("- Production post-processing")
    print("- Testing infrastructure")
    print("- Essential utilities")

if __name__ == "__main__":
    main()