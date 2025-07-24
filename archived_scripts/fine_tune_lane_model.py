#!/usr/bin/env python3
"""
Fine-tuning script for 12-class lane marking detection.
Phase 2 enhancement to achieve 80-85% mIoU target.
"""

import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_fine_tuning_environment():
    """
    Setup environment for fine-tuning with our AEL dataset.
    """
    logger.info("Phase 2 Fine-tuning Setup: Preparing 12-class lane marking model")
    
    # Check for required files
    config_file = "configs/mmseg/swin_base_lane_markings_12class.py"
    dataset_dir = "data/ael_mmseg"
    weights_dir = "weights"
    
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return False
    
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        logger.info(f"Created weights directory: {weights_dir}")
    
    return True

def validate_dataset_structure():
    """
    Validate that our AEL dataset has the proper structure for training.
    """
    logger.info("Validating AEL dataset structure for 12-class training...")
    
    required_dirs = [
        "data/ael_mmseg/img_dir/train",
        "data/ael_mmseg/ann_dir/train", 
        "data/ael_mmseg/img_dir/val",
        "data/ael_mmseg/ann_dir/val"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Required directory missing: {dir_path}")
            return False
        
        # Count files
        files = list(Path(dir_path).glob("*"))
        logger.info(f"{dir_path}: {len(files)} files")
    
    logger.info("Dataset structure validation: PASSED")
    return True

def create_training_command():
    """
    Create the MMSegmentation training command for fine-tuning.
    """
    config_file = "configs/mmseg/swin_base_lane_markings_12class.py"
    work_dir = "work_dirs/swin_12class_lane_fine_tune"
    
    # Use ADE20K pretrained weights as starting point
    pretrained_weights = "weights/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth"
    
    training_command = f"""
python -m mmseg.tools.train \\
    {config_file} \\
    --work-dir {work_dir} \\
    --load-from {pretrained_weights} \\
    --seed 42 \\
    --deterministic \\
    --gpu-ids 0
    """
    
    return training_command.strip()

def run_fine_tuning(dry_run=True):
    """
    Execute the fine-tuning process.
    
    Args:
        dry_run: If True, only show what would be executed
    """
    logger.info("Phase 2 Fine-tuning: Starting 12-class lane marking model training")
    
    if not setup_fine_tuning_environment():
        logger.error("Environment setup failed")
        return False
    
    if not validate_dataset_structure():
        logger.error("Dataset validation failed")
        return False
    
    training_command = create_training_command()
    
    if dry_run:
        logger.info("DRY RUN MODE - Would execute:")
        logger.info(training_command)
        logger.info("\nTo actually run training, use: python fine_tune_lane_model.py --execute")
        return True
    
    logger.info("Executing fine-tuning command...")
    logger.info(training_command)
    
    # Execute training
    import subprocess
    try:
        result = subprocess.run(
            training_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("Fine-tuning completed successfully!")
            logger.info("Training output:")
            logger.info(result.stdout)
            
            # Move best checkpoint to weights directory
            work_dir = "work_dirs/swin_12class_lane_fine_tune"
            best_checkpoint = os.path.join(work_dir, "best_mIoU_iter_*.pth")
            
            import glob
            checkpoints = glob.glob(best_checkpoint)
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                target_path = "weights/fine_tuned_12class_lane_markings.pth"
                
                import shutil
                shutil.copy2(latest_checkpoint, target_path)
                logger.info(f"Best checkpoint copied to: {target_path}")
                
                # Update symbolic link for automatic loading
                best_link = "weights/best.pth"
                if os.path.exists(best_link):
                    os.remove(best_link)
                os.symlink(os.path.abspath(target_path), best_link)
                logger.info(f"Updated best model link: {best_link}")
            
            return True
        else:
            logger.error("Fine-tuning failed!")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Fine-tuning timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"Fine-tuning execution failed: {e}")
        return False

def create_evaluation_script():
    """
    Create evaluation script to test fine-tuned model performance.
    """
    eval_script = """
#!/usr/bin/env python3
import json
import time
import requests
import numpy as np

def test_enhanced_model():
    '''Test the fine-tuned 12-class model performance'''
    
    # Test coordinates for Brisbane (same as before)
    test_coordinates = {
        "north": -27.4698,
        "south": -27.4705,
        "east": 153.0258,
        "west": 153.0251
    }
    
    print("Testing Phase 2 Enhanced 12-Class Lane Marking Model")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8010/analyze_road_infrastructure",
            json=test_coordinates,
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"[PASS] Response Time: {response_time:.2f}s")
            print(f"[PASS] Total Elements: {len(results.get('infrastructure_elements', []))}")
            
            # Analyze class diversity
            classes_detected = set()
            for element in results.get('infrastructure_elements', []):
                classes_detected.add(element.get('class', 'unknown'))
            
            print(f"[PASS] Classes Detected: {len(classes_detected)}")
            print(f"   Classes: {', '.join(sorted(classes_detected))}")
            
            # Check for enhancement status
            if results.get('enhancement_status'):
                print(f"[PASS] Enhancement: {results['enhancement_status']}")
            
            # Performance assessment
            if len(classes_detected) >= 3:
                print("[TARGET] IMPROVEMENT: Detecting multiple lane types")
            else:
                print("[LIMITED] Still detecting limited class diversity")
            
            if response_time < 2.0:
                print("[TARGET] PERFORMANCE: Response time target met")
            else:
                print("[SLOW] PERFORMANCE: Response time exceeded target")
                
            return {
                'response_time': response_time,
                'total_elements': len(results.get('infrastructure_elements', [])),
                'classes_detected': len(classes_detected),
                'class_names': list(classes_detected),
                'enhancement_active': bool(results.get('enhancement_status'))
            }
        else:
            print(f"[ERROR] HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return None

if __name__ == "__main__":
    results = test_enhanced_model()
    if results:
        print("\\n" + "=" * 60)
        print("PHASE 2 PERFORMANCE SUMMARY")
        print("=" * 60)
        for key, value in results.items():
            print(f"{key}: {value}")
"""
    
    with open("test_phase2_performance.py", "w", encoding="utf-8") as f:
        f.write(eval_script.strip())
    
    logger.info("Created evaluation script: test_phase2_performance.py")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LaneSegNet for 12-class detection")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually run training (default is dry-run)")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only create evaluation script")
    
    args = parser.parse_args()
    
    if args.eval_only:
        create_evaluation_script()
        return
    
    # Create evaluation script regardless
    create_evaluation_script()
    
    # Run fine-tuning
    success = run_fine_tuning(dry_run=not args.execute)
    
    if success and args.execute:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2 FINE-TUNING COMPLETE")
        logger.info("="*60)
        logger.info("Next steps:")
        logger.info("1. Restart Docker container to load fine-tuned model")
        logger.info("2. Run: python test_phase2_performance.py")
        logger.info("3. Compare results with baseline performance")
    elif success:
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING PREPARATION COMPLETE")
        logger.info("="*60)
        logger.info("To execute training: python fine_tune_lane_model.py --execute")
        logger.info("To test current model: python test_phase2_performance.py")

if __name__ == "__main__":
    main()