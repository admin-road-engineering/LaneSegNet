#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization for Phase 3.2.5
Limited 5-10 trials for efficient tuning without compromising stability
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import from our premium training script
import sys
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, EnhancedDiceFocalLoss, PremiumDataset, calculate_detailed_metrics

def objective(trial):
    """
    Bayesian optimization objective function.
    Limited parameter space to maintain stability.
    """
    print(f"\n=== Optuna Trial {trial.number + 1} ===")
    
    # Limited hyperparameter search space (expert panel guidance)
    lr = trial.suggest_float('learning_rate', 5e-4, 2e-3, log=True)
    dice_weight = trial.suggest_float('dice_weight', 0.5, 0.7)
    focal_weight = 1.0 - dice_weight  # Ensure they sum to 1
    
    # Keep class weights fixed for stability [0.1, 5.0, 5.0] - 3 classes only
    class_weights = [0.1, 5.0, 5.0]
    
    print(f"Trial params: LR={lr:.2e}, Dice={dice_weight:.2f}, Focal={focal_weight:.2f}")
    
    try:
        # Quick training on subset for efficiency
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Small model for fast trials (3 classes: background, white_solid, white_dashed)
        model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)
        
        # Enhanced loss with trial parameters
        criterion = EnhancedDiceFocalLoss(
            alpha=1.0,
            gamma=2.0,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            class_weights=class_weights,
            label_smoothing=0.05
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # Small subset for fast evaluation (10% of data)
        full_train_dataset = PremiumDataset("data/ael_mmseg/img_dir/train", "data/ael_mmseg/ann_dir/train", mode='train')
        full_val_dataset = PremiumDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
        
        # Sample subset
        train_size = len(full_train_dataset) // 10  # 10% for speed
        val_size = len(full_val_dataset) // 5      # 20% for validation
        
        train_indices = np.random.choice(len(full_train_dataset), train_size, replace=False)
        val_indices = np.random.choice(len(full_val_dataset), val_size, replace=False)
        
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_val_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=2)
        
        print(f"Training on {len(train_subset)} samples, validating on {len(val_subset)} samples")
        
        # Short training loop (5 epochs for speed)
        model.train()
        for epoch in range(5):
            epoch_losses = []
            
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss, dice, focal = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            print(f"  Epoch {epoch+1}/5: Loss {np.mean(epoch_losses):.4f}")
        
        # Evaluation
        model.eval()
        all_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                pred = torch.argmax(outputs, dim=1)
                
                for i in range(images.size(0)):
                    ious, _, _, _ = calculate_detailed_metrics(pred[i], masks[i])
                    all_ious.append(ious)
        
        # Calculate metrics
        mean_ious = np.mean(all_ious, axis=0)
        overall_miou = np.mean(mean_ious)
        lane_classes_miou = np.mean(mean_ious[1:])  # Focus on lanes
        
        # Balanced score (same as main training)
        balanced_score = (overall_miou * 0.4) + (lane_classes_miou * 0.6)
        
        print(f"  Results: mIoU={overall_miou:.1%}, Lane mIoU={lane_classes_miou:.1%}, Balanced={balanced_score:.1%}")
        
        # Cleanup
        del model, criterion, optimizer
        torch.cuda.empty_cache()
        
        return balanced_score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

def run_bayesian_optimization(n_trials=8):
    """
    Run limited Bayesian optimization for hyperparameter tuning.
    Expert panel recommendation: 5-10 trials for efficiency.
    """
    print("=== Bayesian Hyperparameter Optimization - Phase 3.2.5 ===")
    print(f"Running {n_trials} trials for optimal LR and Dice/Focal weights")
    print("Class weights fixed at [0.1, 5.0, 5.0] for stability (3 classes)")
    print()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"premium_lane_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour max
    
    # Results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Best balanced score: {study.best_value:.1%}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        if 'learning_rate' in key:
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value:.3f}")
    
    # Save results
    results_dir = Path("work_dirs")
    results_dir.mkdir(exist_ok=True)
    
    optimization_results = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / "bayesian_optimization_results.json", 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\nResults saved to work_dirs/bayesian_optimization_results.json")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Use best parameters in premium training")
    print("2. Expected gain: 1-3% mIoU over default parameters")
    print("3. If best score < 50%, increase n_trials or check data quality")
    
    return study.best_params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=8, help='Number of optimization trials (default: 8)')
    parser.add_argument('--quick', action='store_true', help='Quick 5-trial run for testing')
    
    args = parser.parse_args()
    
    n_trials = 5 if args.quick else args.trials
    
    print(f"Starting optimization with {n_trials} trials...")
    best_params = run_bayesian_optimization(n_trials)
    
    print("\nOptimization complete! Use these parameters in premium training:")
    print(f"  python scripts/premium_gpu_train.py --lr {best_params.get('learning_rate', 1e-3):.2e} --dice-weight {best_params.get('dice_weight', 0.6):.3f}")