#!/usr/bin/env python3
"""
Emergency patch to add versioned saving to running training
Can be imported into existing training script
"""

import torch
import json
from datetime import datetime
from pathlib import Path

class VersionedModelSaver:
    """Add to existing training loop for automatic versioned backups"""
    
    def __init__(self, save_dir="work_dirs"):
        self.save_dir = Path(save_dir)
        self.history_file = self.save_dir / "training_history.json"
        self.load_history()
        
    def load_history(self):
        """Load existing training history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}
    
    def save_versioned_checkpoint(self, model, epoch, miou, per_class_ious, is_best=False):
        """Save model with automatic versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create versioned filename
        versioned_name = f"premium_epoch{epoch:03d}_{miou*100:.1f}pct_{timestamp}.pth"
        versioned_path = self.save_dir / versioned_name
        
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'miou': miou,
            'per_class_ious': per_class_ious,
            'timestamp': timestamp,
            'is_best': is_best
        }, versioned_path)
        
        # Update history
        self.history[f"epoch_{epoch}"] = {
            'miou': miou,
            'per_class_ious': per_class_ious,
            'checkpoint': versioned_name,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        # Save history
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Versioned checkpoint saved: {versioned_name}")
        
        if is_best:
            # Also update the "best" pointer (but keep all versions)
            best_path = self.save_dir / "premium_gpu_best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'miou': miou,
                'per_class_ious': per_class_ious,
                'timestamp': timestamp,
                'source_checkpoint': versioned_name
            }, best_path)
            print(f"✅ Updated best model pointer (source: {versioned_name})")
        
        return versioned_path
    
    def get_best_models(self, top_k=5):
        """Get top K best models from history"""
        sorted_epochs = sorted(
            self.history.items(),
            key=lambda x: x[1]['miou'],
            reverse=True
        )
        return sorted_epochs[:top_k]

# Usage example - can be integrated into existing training loop:
"""
# Add to your training script:
model_saver = VersionedModelSaver()

# In your training loop, replace:
# torch.save(model.state_dict(), "work_dirs/premium_gpu_best_model.pth")

# With:
model_saver.save_versioned_checkpoint(
    model=model,
    epoch=epoch,
    miou=current_miou, 
    per_class_ious=mean_ious.tolist(),
    is_best=(current_miou > best_miou)
)
"""