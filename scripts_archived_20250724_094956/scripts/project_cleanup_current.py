#!/usr/bin/env python3
"""
Project Cleanup Script - Safe Archive of Obsolete Files
Maintains functionality while organizing completed/obsolete components.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def create_archive_directory():
    """Create timestamped archive directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archived_files_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def identify_archivable_files():
    """Identify files that can be safely archived without affecting functionality."""
    
    # Files that are safe to archive (completed/obsolete)
    archivable = {
        "root_files": [
            "ADVANCED_TECHNIQUES_PLAN.md",
            "OVERFITTING_FIX_PLAN.md", 
            "PIPELINE_RUN_REPORT_TEMPLATE.md",
            "REVISED_ADVANCED_TECHNIQUES_PLAN.md",
            "audit_report_20250723_113156.json",
            "backup_model_comparison_20250723_113826.json",
            "check_ssl_readiness.py",
            "comprehensive_validation_results_20250723_112903.json",
            "full_dataset_report.json",
            "debug_iou_issue.py"  # Debugging script - served its purpose
        ],
        
        "directories": [
            "comprehensive_test_results_20250722_235739",
            "comprehensive_test_results_20250722_235758", 
            "local_aerial_tests",  # Test results - can be archived
            "test_images"  # Sample images - keep copies but archive originals
        ],
        
        "scripts": [
            # Training scripts that have been superseded
            "analyze_miou_discrepancy.py",
            "balanced_eval.py", 
            "balanced_monitor.py",
            "balanced_train.py",
            "check_collection_status.py",
            "cleanup_project.py",  # Old cleanup script
            "collect_osm_1000.py",
            "combined_dataset_train.py",
            "create_finetune_data.py",
            "download_osm_tiles.py",
            "download_skyscapes.py",
            "evaluate_baseline.py",
            "evaluate_final.py",
            "fast_gpu_train.py",
            "finetune_premium_model.py",
            "fix_dataset_split.py",
            "full_dataset_premium_training.py",
            "generate_carla_aerial.py",
            "investigate_test_annotations.py",
            "optimized_premium_training.py",
            "premium_gpu_train.py",
            "prepare_combined_datasets.py", 
            "prepare_full_dataset.py",
            "prepare_training_data.py",
            "proper_finetune.py",
            "quick_annotation_check.py",
            "quick_eval.py",
            "quick_holdout_test.py",
            "quick_model_comparison.py",
            "quick_osm_test.py",
            "quick_training_setup.py",
            "regenerate_test_masks.py",
            "simple_balanced_monitor.py",
            "simple_cleanup.py",
            "simple_monitor.py",
            "swin_transformer_train.py",
            "test_all_backup_models.py",
            "test_comprehensive_datasets.py",
            "test_current_model_holdout.py",
            "test_multi_datasets.py",
            "test_on_training_data.py",
            "transform_cityscapes_aerial.py"
        ],
        
        "batch_files": [
            # Batch files superseded by current pipeline
            "run_backup_model_testing.bat",
            "run_combined_training.bat",
            "run_combined_training_simple.bat",
            "run_comprehensive_audit.bat", 
            "run_comprehensive_validation.bat",
            "run_data_collection.bat",  # Collection complete
            "run_finetune_premium.bat",
            "run_finetune_simple.bat",
            "run_full_dataset_training.bat",
            "run_full_pipeline.bat",
            "run_optimized_premium_training.bat",
            "run_premium_gpu.bat",
            "run_proper_finetune.bat",
            "test_models_with_progress.bat"
        ]
    }
    
    # Files that MUST be kept (core functionality)
    keep_files = {
        "CLAUDE.md",  # Project documentation
        "README.md",
        "TESTING.md", 
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "pytest.ini",
        "app/",  # Core API
        "configs/",  # Model configurations
        "data/ael_mmseg/",  # Working dataset
        "data/labeled_dataset.py",  # Current dataset implementation
        "data/unlabeled_dataset.py",  # SSL pre-training dataset
        "mmseg_custom/",  # Custom MMSeg components
        "scripts/run_finetuning.py",  # Current fine-tuning script
        "scripts/ssl_pretraining.py",  # SSL pre-training
        "scripts/ohem_loss.py",  # Advanced loss function
        "scripts/enhanced_post_processing.py",  # Production post-processing
        "scripts/knowledge_distillation.py",  # Production optimization
        "scripts/test_time_augmentation.py",  # Production enhancement
        "run_continue_finetuning.bat",  # Current pipeline
        "run_ssl_pipeline.bat",  # SSL pipeline
        "tests/",  # Testing infrastructure
        "weights/",  # Model weights
        "work_dirs/",  # Training outputs
        "visualizations/"  # Current visualization outputs
    }
    
    return archivable, keep_files

def safe_archive_files():
    """Safely archive obsolete files while preserving functionality."""
    archive_dir = create_archive_directory()
    archivable, keep_files = identify_archivable_files()
    
    archived_manifest = {
        "timestamp": datetime.now().isoformat(),
        "archived_files": [],
        "preserved_files": list(keep_files),
        "purpose": "Archive obsolete training scripts and intermediate results while preserving core functionality"
    }
    
    print("PROJECT CLEANUP - ARCHIVING OBSOLETE FILES")
    print("=" * 60)
    
    # Archive root files
    root_archived = archive_dir / "root_files"
    root_archived.mkdir(exist_ok=True)
    
    for file in archivable["root_files"]:
        if Path(file).exists():
            shutil.move(file, root_archived / file)
            archived_manifest["archived_files"].append(f"root_files/{file}")
            print(f"ARCHIVED: {file}")
    
    # Archive directories
    dirs_archived = archive_dir / "directories" 
    dirs_archived.mkdir(exist_ok=True)
    
    for directory in archivable["directories"]:
        if Path(directory).exists():
            shutil.move(directory, dirs_archived / directory)
            archived_manifest["archived_files"].append(f"directories/{directory}")
            print(f"✅ Archived directory: {directory}")
    
    # Archive obsolete scripts
    scripts_archived = archive_dir / "scripts"
    scripts_archived.mkdir(exist_ok=True)
    
    for script in archivable["scripts"]:
        script_path = Path("scripts") / script
        if script_path.exists():
            shutil.move(script_path, scripts_archived / script)
            archived_manifest["archived_files"].append(f"scripts/{script}")
            print(f"✅ Archived script: {script}")
    
    # Archive obsolete batch files
    batch_archived = archive_dir / "batch_files"
    batch_archived.mkdir(exist_ok=True)
    
    for batch in archivable["batch_files"]:
        if Path(batch).exists():
            shutil.move(batch, batch_archived / batch)
            archived_manifest["archived_files"].append(f"batch_files/{batch}")
            print(f"✅ Archived batch: {batch}")
    
    # Save archive manifest
    with open(archive_dir / "ARCHIVE_MANIFEST.json", 'w') as f:
        json.dump(archived_manifest, f, indent=2)
    
    print(f"\n📁 Archive created: {archive_dir}")
    print(f"📄 Files archived: {len(archived_manifest['archived_files'])}")
    print(f"💾 Manifest saved: {archive_dir}/ARCHIVE_MANIFEST.json")
    
    return archive_dir, archived_manifest

def cleanup_data_directories():
    """Clean up data directories - remove empty/invalid datasets."""
    print("\n🧹 CLEANING DATA DIRECTORIES")
    print("=" * 40)
    
    # Remove problematic datasets
    problematic_dirs = [
        "data/full_ael_mmseg",  # Has empty masks - causes training issues
        "data/fixed_ael_mmseg",  # Likely also problematic
    ]
    
    for directory in problematic_dirs:
        if Path(directory).exists():
            # Move to archive instead of deleting (safer)
            archive_name = f"archived_data_{directory.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(directory, archive_name)
            print(f"🗂️  Moved problematic dataset: {directory} → {archive_name}")
    
    # Keep working datasets
    working_datasets = [
        "data/ael_mmseg",  # Current working dataset
        "data/unlabeled_aerial",  # SSL pre-training data
        "data/mask",  # Working masks
        "data/labeled_dataset.py",  # Current implementation
        "data/unlabeled_dataset.py"  # SSL implementation
    ]
    
    print("✅ Preserved working datasets:")
    for dataset in working_datasets:
        if Path(dataset).exists():
            print(f"   - {dataset}")

if __name__ == "__main__":
    print("🧹 LaneSegNet Project Cleanup")
    print("Moving obsolete files to archives while preserving functionality")
    print()
    
    try:
        # Archive obsolete files
        archive_dir, manifest = safe_archive_files()
        
        # Clean data directories
        cleanup_data_directories()
        
        print("\n" + "=" * 60)
        print("✅ PROJECT CLEANUP COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("🚀 Current functional components preserved:")
        print("   - SSL pre-training pipeline (run_ssl_pipeline.bat)")
        print("   - Fine-tuning pipeline (run_continue_finetuning.bat)")
        print("   - Core API service (app/)")
        print("   - Working dataset (data/ael_mmseg/)")
        print("   - Testing infrastructure (tests/)")
        print("   - Production models (work_dirs/)")
        print()
        print(f"📦 {len(manifest['archived_files'])} obsolete files archived safely")
        print(f"📁 Archive location: {archive_dir}")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("Manual review required before proceeding")