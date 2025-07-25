#!/usr/bin/env python3
"""
Phase 0: Configuration Audit Script
Validates consistency across all configuration files to prevent silent failures.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_python_config(config_path: Path) -> Dict[str, Any]:
    """Load a Python configuration file and extract relevant variables."""
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract relevant configuration variables
        config_vars = {}
        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):
                attr_value = getattr(config_module, attr_name)
                if isinstance(attr_value, (int, float, str, list, tuple, dict)):
                    config_vars[attr_name] = attr_value
        
        return config_vars
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}

def audit_global_config():
    """Audit the central global configuration."""
    logger.info("🔍 Auditing global configuration...")
    
    global_config_path = Path("configs/global_config.py")
    if not global_config_path.exists():
        logger.error("❌ Global config not found: configs/global_config.py")
        return False
    
    config = load_python_config(global_config_path)
    
    # Validate required fields
    required_fields = ['NUM_CLASSES', 'IMG_SIZE', 'CLASS_NAMES']
    for field in required_fields:
        if field not in config:
            logger.error(f"❌ Missing required field in global config: {field}")
            return False
        else:
            logger.info(f"✅ {field}: {config[field]}")
    
    # Validate consistency
    if len(config.get('CLASS_NAMES', [])) != config.get('NUM_CLASSES', 0):
        logger.error(f"❌ CLASS_NAMES length ({len(config['CLASS_NAMES'])}) != NUM_CLASSES ({config['NUM_CLASSES']})")
        return False
    
    logger.info("✅ Global configuration validation passed")
    return config

def audit_model_configs():
    """Audit MMSegmentation model configuration files."""
    logger.info("🔍 Auditing model configuration files...")
    
    configs_dir = Path("configs")
    model_configs = list(configs_dir.glob("*.py"))
    
    if not model_configs:
        logger.warning("⚠️ No model configuration files found in configs/")
        return True
    
    config_summary = {}
    
    for config_path in model_configs:
        if config_path.name == "global_config.py":
            continue  # Skip global config (already audited)
            
        logger.info(f"Checking: {config_path.name}")
        config = load_python_config(config_path)
        
        # Extract key parameters
        key_params = {
            'num_classes': config.get('num_classes'),
            'img_size': config.get('img_size'),
            'model_type': config.get('model', {}).get('type') if isinstance(config.get('model'), dict) else None,
            'backbone_type': config.get('model', {}).get('backbone', {}).get('type') if isinstance(config.get('model'), dict) else None,
        }
        
        config_summary[config_path.name] = key_params
        
        for param, value in key_params.items():
            if value is not None:
                logger.info(f"  {param}: {value}")
    
    return config_summary

def audit_training_configs():
    """Audit training-related configuration consistency."""
    logger.info("🔍 Auditing training configuration consistency...")
    
    # Check run_finetuning.py for configuration usage
    finetuning_script = Path("scripts/run_finetuning.py")
    if finetuning_script.exists():
        with open(finetuning_script, 'r') as f:
            content = f.read()
            
        # Check for global config import
        if "from configs.global_config import NUM_CLASSES" in content:
            logger.info("✅ run_finetuning.py uses central NUM_CLASSES")
        else:
            logger.warning("⚠️ run_finetuning.py may not use central NUM_CLASSES")
            
        # Check for hardcoded values
        hardcoded_patterns = [
            ("range(1, 4)", "Hardcoded class range"),
            ("num_classes=3", "Hardcoded num_classes"),
            ("classes=3", "Hardcoded classes"),
        ]
        
        for pattern, description in hardcoded_patterns:
            if pattern in content:
                logger.warning(f"⚠️ Found potential hardcoded value: {description}")
            else:
                logger.info(f"✅ No hardcoded {description.lower()} found")
    
    # Check knowledge_distillation.py
    kd_script = Path("scripts/knowledge_distillation.py")
    if kd_script.exists():
        with open(kd_script, 'r') as f:
            content = f.read()
            
        if "from configs.global_config import NUM_CLASSES" in content:
            logger.info("✅ knowledge_distillation.py uses central NUM_CLASSES")
        else:
            logger.warning("⚠️ knowledge_distillation.py may not use central NUM_CLASSES")
    
    return True

def audit_data_configurations():
    """Audit data loading and preprocessing configurations."""
    logger.info("🔍 Auditing data loading configurations...")
    
    # Check if data directories exist
    data_paths_to_check = [
        ("data/ael_mmseg/img_dir/train", "Training images"),
        ("data/ael_mmseg/ann_dir/train", "Training annotations"),
        ("data/ael_mmseg/img_dir/val", "Validation images"),
        ("data/ael_mmseg/ann_dir/val", "Validation annotations"),
    ]
    
    data_status = {}
    for path_str, description in data_paths_to_check:
        path = Path(path_str)
        if path.exists():
            file_count = len(list(path.glob("*")))
            logger.info(f"✅ {description}: {file_count} files")
            data_status[description] = file_count
        else:
            logger.error(f"❌ {description}: Path not found - {path}")
            data_status[description] = 0
    
    # Check for critical data issues
    if data_status.get("Training images", 0) == 0:
        logger.error("❌ CRITICAL: No training images found")
        return False
    
    if data_status.get("Validation images", 0) == 0:
        logger.error("❌ CRITICAL: No validation images found")
        return False
    
    # Check for data balance
    train_imgs = data_status.get("Training images", 0)
    train_anns = data_status.get("Training annotations", 0)
    val_imgs = data_status.get("Validation images", 0)
    val_anns = data_status.get("Validation annotations", 0)
    
    if train_imgs != train_anns:
        logger.error(f"❌ Training data mismatch: {train_imgs} images vs {train_anns} annotations")
        return False
    
    if val_imgs != val_anns:
        logger.error(f"❌ Validation data mismatch: {val_imgs} images vs {val_anns} annotations")
        return False
    
    logger.info("✅ Data configuration audit passed")
    return data_status

def audit_preprocessing_consistency():
    """Audit preprocessing pipeline consistency."""
    logger.info("🔍 Auditing preprocessing consistency...")
    
    # Check labeled_dataset.py for preprocessing parameters
    dataset_script = Path("data/labeled_dataset.py")
    if dataset_script.exists():
        with open(dataset_script, 'r') as f:
            content = f.read()
        
        # Look for key preprocessing parameters
        preprocessing_checks = [
            ("Normalize", "Normalization transforms"),
            ("Resize", "Resizing transforms"),
            ("ToTensor", "Tensor conversion"),
            ("RGB", "Color space handling"),
            ("BGR", "OpenCV color space"),
        ]
        
        for pattern, description in preprocessing_checks:
            if pattern in content:
                logger.info(f"✅ Found {description}")
            else:
                logger.warning(f"⚠️ May be missing {description}")
    else:
        logger.error("❌ labeled_dataset.py not found - cannot audit preprocessing")
        return False
    
    return True

def generate_audit_report(global_config, model_configs, data_status):
    """Generate comprehensive audit report."""
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "global_config": global_config,
        "model_configs": model_configs,
        "data_status": data_status,
        "critical_issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Analyze for critical issues
    if global_config.get('NUM_CLASSES', 0) <= 0:
        report["critical_issues"].append("Invalid NUM_CLASSES in global config")
    
    if data_status.get("Training images", 0) == 0:
        report["critical_issues"].append("No training data found")
    
    if data_status.get("Validation images", 0) == 0:
        report["critical_issues"].append("No validation data found")
    
    # Generate recommendations
    if not report["critical_issues"]:
        report["recommendations"].append("Configuration audit passed - proceed to Phase 1")
    else:
        report["recommendations"].append("Fix critical issues before proceeding")
    
    # Save report
    report_path = Path("work_dirs/config_audit_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Audit report saved: {report_path}")
    return report

def main():
    """Run comprehensive configuration audit."""
    logger.info("🚀 Starting Phase 0: Configuration Audit")
    logger.info("=" * 60)
    
    try:
        # Run all audit phases
        global_config = audit_global_config()
        if not global_config:
            logger.error("💥 Global configuration audit failed - stopping")
            return False
        
        model_configs = audit_model_configs()
        training_config_ok = audit_training_configs()
        data_status = audit_data_configurations()
        preprocessing_ok = audit_preprocessing_consistency()
        
        if not data_status:
            logger.error("💥 Data configuration audit failed - stopping")
            return False
        
        # Generate comprehensive report
        report = generate_audit_report(global_config, model_configs, data_status)
        
        # Final assessment
        logger.info("=" * 60)
        logger.info("📊 CONFIGURATION AUDIT SUMMARY")
        logger.info("=" * 60)
        
        if report["critical_issues"]:
            logger.error("❌ CRITICAL ISSUES FOUND:")
            for issue in report["critical_issues"]:
                logger.error(f"  • {issue}")
            logger.error("🚫 DO NOT PROCEED TO PHASE 1 - FIX ISSUES FIRST")
            return False
        else:
            logger.info("✅ NO CRITICAL ISSUES FOUND")
            logger.info("🎯 READY TO PROCEED TO PHASE 1: OVERFITTING TEST")
            
        if report["warnings"]:
            logger.warning("⚠️ Warnings found:")
            for warning in report["warnings"]:
                logger.warning(f"  • {warning}")
        
        logger.info("\n📋 Next Steps:")
        logger.info("  1. Review audit report: work_dirs/config_audit_report.json")
        logger.info("  2. Proceed to Phase 1: scripts/overfit_tiny_dataset.py")
        logger.info("  3. Expected time: 4 hours (critical decision point)")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Configuration audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)