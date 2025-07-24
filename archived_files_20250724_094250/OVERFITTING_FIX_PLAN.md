# Comprehensive Plan to Fix Model Overfitting and Improve Generalization

## Current Situation Analysis
- **Training mIoU**: 85.1% (reported)
- **Validation mIoU**: 79.4% (likely contaminated)
- **True Test mIoU**: 41.8% (proper holdout)
- **Generalization Gap**: 37.6% (severe overfitting)
- **Key Issue**: Model memorized training data without learning generalizable patterns

## Phase 1: Dataset Integrity Fix (Days 1-2)

### Step 1.1: Create Clean Dataset Split
1. **Verify dataset isolation**:
   - Use hash-based splitting (not random) to ensure reproducibility
   - Create checksums of each split to detect contamination
   - Implement strict file tracking to prevent leakage

2. **Implementation steps**:
   ```
   a. Load all 7,817 sample IDs
   b. Sort IDs deterministically 
   c. Use modulo hash splitting:
      - Hash(ID) % 100 < 70 → train
      - Hash(ID) % 100 < 85 → val
      - Hash(ID) % 100 >= 85 → test
   d. Verify zero overlap between splits
   e. Save split manifests for tracking
   ```

3. **Validation checks**:
   - Ensure balanced class distribution across splits
   - Verify geographic diversity (all 7 cities represented)
   - Check annotation quality consistency

### Step 1.2: Fix Mask Generation Pipeline
1. **Standardize mask format**:
   - Map original multi-class masks to 3-class system consistently
   - Implement proper class mapping logic
   - Add validation to ensure no data loss

2. **Quality assurance**:
   - Spot-check 100 random masks from each split
   - Verify pixel counts match between original and converted
   - Create visualization tools for manual inspection

## Phase 2: Model Architecture Redesign (Days 3-4)

### Step 2.1: Reduce Model Complexity
1. **Current issue**: 8.9M parameter model on 5,471 training samples
2. **New architecture**:
   ```
   - Reduce to 2-3M parameters
   - Simpler encoder: ResNet18 or MobileNetV2
   - Lighter decoder: Reduce channel dimensions by 50%
   - Remove unnecessary attention mechanisms
   ```

### Step 2.2: Add Strong Regularization
1. **Dropout strategy**:
   - Increase dropout: 0.3 → 0.5
   - Add spatial dropout in encoder
   - Apply dropout to skip connections

2. **Weight regularization**:
   - L2 weight decay: 1e-3
   - Gradient clipping: max_norm=1.0
   - Batch normalization with momentum=0.9

3. **Data augmentation pipeline**:
   ```
   - Geometric: Random flip, rotation (±15°), scale (0.8-1.2)
   - Color: Brightness, contrast, saturation jitter
   - Noise: Gaussian noise, blur
   - Lane-specific: Random lane occlusion, perspective shifts
   ```

## Phase 3: Training Methodology Overhaul (Days 5-7)

### Step 3.1: Implement Proper Validation
1. **Early stopping strategy**:
   - Monitor validation loss (not mIoU)
   - Patience: 10 epochs
   - Restore best weights on stop

2. **Learning rate scheduling**:
   - Start: 1e-3
   - ReduceLROnPlateau: factor=0.5, patience=5
   - Min LR: 1e-6

3. **Batch size optimization**:
   - Increase batch size: 8 → 32
   - Use gradient accumulation if needed
   - Ensure batch statistics are stable

### Step 3.2: Loss Function Improvements
1. **Balanced loss combination**:
   ```
   Total Loss = 0.3 * DiceLoss + 0.3 * FocalLoss + 0.4 * BoundaryLoss
   ```

2. **Class weighting refinement**:
   - Calculate true class frequencies from clean splits
   - Use inverse frequency weighting
   - Add boundary-aware loss for edge quality

### Step 3.3: Training Protocol
1. **Progressive training**:
   - Stage 1: Train on easy samples (clear lanes)
   - Stage 2: Add medium difficulty (partial occlusion)
   - Stage 3: Full dataset (all difficulties)

2. **Curriculum learning**:
   - Start with 512x512 resolution
   - Progress to 1024x1024
   - Final training at 1280x1280

## Phase 4: Evaluation Framework (Days 8-9)

### Step 4.1: Robust Testing Protocol
1. **Cross-validation**:
   - Implement 5-fold cross-validation
   - Report mean and std of metrics
   - Ensure each fold maintains class balance

2. **Multiple metrics**:
   - Primary: mIoU on holdout test
   - Secondary: Dice, F1, Precision, Recall
   - Lane-specific: Lane continuity, connectivity

3. **Generalization tests**:
   - Test on each city separately
   - Evaluate on different lighting conditions
   - Check performance on edge cases

### Step 4.2: Overfitting Detection
1. **Training monitoring**:
   - Plot train vs validation curves every epoch
   - Track gradient norms
   - Monitor weight distribution changes

2. **Early warning system**:
   - Alert if val_loss increases for 3 epochs
   - Flag if train/val gap > 10%
   - Stop if test performance degrades

## Phase 5: Implementation Checklist

### Pre-implementation Requirements:
- [ ] Backup current model and results
- [ ] Set up experiment tracking (MLflow/WandB)
- [ ] Prepare compute resources (GPU time)
- [ ] Create evaluation scripts

### Implementation Order:
1. **Day 1-2**: Fix dataset splits and verify integrity
2. **Day 3-4**: Implement new model architecture
3. **Day 5-6**: Set up training pipeline with regularization
4. **Day 7**: Run initial training experiments
5. **Day 8-9**: Evaluate and iterate

### Success Criteria:
- [ ] Test mIoU > 65% (realistic target)
- [ ] Train-test gap < 15%
- [ ] Consistent performance across cities
- [ ] Model size < 50MB

## Phase 6: Monitoring and Iteration

### Key Metrics to Track:
1. **Per-epoch**:
   - Train/Val/Test mIoU
   - Loss components
   - Learning rate
   - Gradient norms

2. **Per-experiment**:
   - Best validation mIoU
   - Epochs to convergence
   - Final test performance
   - Overfitting ratio

### Iteration Strategy:
1. **If overfitting persists**:
   - Further reduce model size
   - Increase augmentation strength
   - Add noise to training

2. **If underfitting**:
   - Gradually increase capacity
   - Reduce regularization
   - Check for data quality issues

3. **If unstable training**:
   - Reduce learning rate
   - Increase batch size
   - Add gradient clipping

## Expected Outcomes

### Realistic Performance Targets:
- **Training mIoU**: 70-75%
- **Validation mIoU**: 65-70%
- **Test mIoU**: 60-65%
- **Generalization gap**: < 10%

### Model Characteristics:
- **Size**: 20-30MB (vs current 38MB)
- **Inference time**: < 100ms
- **Robustness**: Consistent across conditions
- **Interpretability**: Clear failure modes

## Risk Mitigation

### Potential Issues and Solutions:
1. **Dataset too small**:
   - Consider pseudo-labeling
   - Explore transfer learning
   - Add synthetic data generation

2. **Class imbalance severe**:
   - Implement OHEM (Online Hard Example Mining)
   - Use focal loss variants
   - Balance sampling strategy

3. **Computation constraints**:
   - Use mixed precision training
   - Implement checkpoint resuming
   - Optimize data loading pipeline

## Next Steps After Plan Approval

1. **Create tracking document** for daily progress
2. **Set up automated testing** for each component
3. **Establish baseline metrics** before changes
4. **Schedule regular review checkpoints**
5. **Prepare rollback strategy** if needed

---

**Note**: This plan prioritizes fixing the fundamental overfitting issue over achieving maximum performance. A properly generalizing 65% mIoU model is far more valuable than an overfitted 85% model.