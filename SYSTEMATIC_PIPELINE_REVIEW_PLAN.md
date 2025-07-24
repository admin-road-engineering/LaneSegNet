# SYSTEMATIC PIPELINE REVIEW PLAN

## 🎯 OBJECTIVE
Identify root cause of persistent 2.8% IoU despite implemented fixes through comprehensive end-to-end validation.

## 📋 PHASE 0: CONFIGURATION AUDIT *(IMMEDIATE - 2 hours)*

### 0.1 Configuration Consistency Check
```python
# scripts/audit_configurations.py
- Review all .py config files in configs/ directory
- Check image resolution and normalization statistics consistency
- Verify class definitions and numerical order [0,1,2]
- Validate learning rates, weight decay, optimizer parameters
- Compare training vs inference configuration alignment
- Check RGB vs BGR channel order consistency
```

## 📋 PHASE 1: BASELINE VALIDATION *(CRITICAL FIRST - 4 hours)*

### 1.1 Tiny Dataset Overfitting Test
```python
# scripts/overfit_tiny_dataset.py
- Create clean subset of 10-20 samples
- Train model to 100% IoU on tiny set
- CRITICAL DECISION POINT:
  - SUCCESS → Problem is in full dataset/validation (go to Phase 2)
  - FAILURE → Problem is in core training loop (go to Phase 3)
```

### 1.2 Synthetic Data Validation
```python
# scripts/test_synthetic_data.py
- Generate perfect synthetic masks with known IoU
- Test model with trivial learning task
- Validate against random baseline performance
```

## 📋 PHASE 2: DATA INTEGRITY DEEP DIVE *(If Phase 1 succeeds)*

### 1.1 Ground Truth Validation
```python
# scripts/validate_ground_truth.py
- Sample 50 random images from training set
- Visualize image + mask overlays
- Verify class labels match expected [0,1,2] format
- Check for label corruption or mismatch
- Validate mask pixel value distribution
```

### 2.2 Data Loading Pipeline Audit
```python
# scripts/audit_data_loading.py  
- Trace single sample through LabeledDataset
- Verify transformations don't corrupt labels
- Check tensor shapes and dtypes at each step
- Validate class indices after augmentation
- Compare raw vs processed data
- **CRITICAL**: Compare preprocessing vs pre-trained model requirements
- Verify normalization stats match pre-trained backbone expectations
```

### 1.3 Class Distribution Analysis
```python
# scripts/analyze_class_distribution.py
- Calculate true class frequencies across entire dataset
- Identify class imbalance severity
- Check for empty or near-empty classes
- Validate against dataset assumptions
```

## 📋 PHASE 3: MODEL ARCHITECTURE VALIDATION *(If Phase 1 fails)*

### 2.1 Forward Pass Verification
```python
# scripts/validate_model_forward.py
- Single sample end-to-end trace
- Verify output shape: (B, 3, 512, 512)
- Check output value ranges and distributions
- Validate softmax/argmax behavior
- Test with known synthetic inputs
```

### 2.2 Weight Loading Verification
```python
# scripts/verify_weight_loading.py
- Compare loaded vs expected weight shapes
- Verify encoder weights properly transferred
- Check decoder initialization
- Validate NUM_CLASSES consistency across layers
```

### 2.3 Architecture Consistency Check
```python
# scripts/check_architecture.py
- Verify final_conv.out_channels == NUM_CLASSES
- Check encoder-decoder connection
- Validate positional embedding interpolation
- Test gradient flow through entire model
```

## 📋 PHASE 4: LOSS FUNCTION & TRAINING ANALYSIS *(If Phase 1 fails)*

### 3.1 Loss Function Behavior
```python
# scripts/analyze_loss_behavior.py
- Monitor per-class loss components
- Check gradient magnitudes
- Verify OHEM selection criteria
- Compare loss vs IoU correlation
```

### 3.2 Learning Rate & Optimization
```python
# scripts/debug_optimization.py
- Track parameter updates per layer
- Monitor gradient norms
- Check for vanishing/exploding gradients
- Validate differential learning rates
```

### 3.3 Training Dynamics
```python
# scripts/analyze_training_dynamics.py
- Plot loss curves with detailed breakdown
- Monitor prediction diversity over epochs
- Check for mode collapse or overfitting
- Validate early stopping triggers
```

## 📋 PHASE 5: METRIC CALCULATION VALIDATION *(Medium Priority)*

### 4.1 IoU Calculation Deep Dive
```python
# scripts/debug_iou_calculation.py
- Manual IoU calculation on synthetic data
- Compare against ground truth IoU implementations
- Test edge cases (empty predictions, no ground truth)
- Validate nanmean handling
```

### 4.2 Prediction Analysis
```python
# scripts/analyze_predictions.py
- Visualize model predictions vs ground truth
- Check prediction confidence distributions
- Identify systematic prediction patterns
- Validate post-processing effects
```

## 📋 PHASE 6: END-TO-END PIPELINE TRACING *(Medium Priority)*

### 5.1 Single Sample Debug
```python
# scripts/trace_single_sample.py
- Follow one sample through entire pipeline
- Log intermediate outputs at each stage
- Verify data consistency throughout
- Check for silent failures or corruptions
```

### 5.2 Batch Processing Validation
```python
# scripts/validate_batch_processing.py
- Compare single vs batch processing results
- Check for batch size dependencies
- Validate GPU vs CPU consistency
- Test memory usage patterns
```

## 📋 PHASE 7: ADVANCED DIAGNOSTICS *(If needed)*

### 7.1 Advanced Sanity Checks
```python
# scripts/advanced_sanity_checks.py
- Compare against known good baseline model implementations
- Cross-validate with different IoU calculation libraries
- Test with different random seeds for reproducibility
```

## 🔍 OPTIMIZED IMPLEMENTATION SEQUENCE

### **IMMEDIATE (First 6 hours)** - Critical Path
1. **Hour 1-2**: Phase 0 - Configuration audit
2. **Hour 3-6**: Phase 1 - Baseline validation (overfitting test)
3. **DECISION POINT**: Branch based on Phase 1 results

### **If Phase 1 SUCCEEDS** (Problem in data/validation)
1. **Day 1**: Phase 2 - Data integrity deep dive
2. **Day 2**: Phase 5 - Metric calculation validation  
3. **Day 3**: Fix implementation and validation

### **If Phase 1 FAILS** (Problem in core training)
1. **Day 1**: Phase 3 - Model architecture validation
2. **Day 2**: Phase 4 - Loss function analysis
3. **Day 3**: Fix implementation and validation

## 🎯 SUCCESS CRITERIA

**Phase 0 Success**: All configurations consistent, no parameter mismatches
**Phase 1 Success**: Model achieves >90% IoU on 10-sample subset within 50 epochs
- If SUCCESS → Data/validation issue confirmed
- If FAILURE → Core training loop issue confirmed
**Phase 2/3 Success**: Root cause identified with specific evidence
**Overall Success**: IoU improves to >30% on full dataset after fix

## 📊 DELIVERABLES

1. **Diagnostic Scripts**: Fast, targeted validation scripts
2. **Root Cause Analysis**: Specific issue identification with evidence  
3. **Fix Implementation**: Targeted solutions for identified problems
4. **Permanent Test Suite**: Convert critical validations to pytest tests
5. **CI Integration**: Add key checks to .github/workflows/ci.yml

## 🚨 CRITICAL ASSUMPTIONS TO VALIDATE

1. **Data Assumption**: Ground truth masks contain correct class labels [0,1,2]
2. **Model Assumption**: Architecture properly handles 3-class segmentation
3. **Training Assumption**: Loss function appropriate for class imbalance
4. **Metric Assumption**: IoU calculation matches standard implementations

## 📋 IMMEDIATE NEXT STEPS

1. **Stop current training** to prevent resource waste
2. **Start with Phase 0**: Configuration audit (2 hours)
3. **Execute Phase 1**: Overfitting test (4 hours) - **CRITICAL DECISION POINT**
4. **Branch based on results**: Data path vs Core training path
5. **Implement permanent tests** for future prevention

## ⚠️ RISK MITIGATION

- **Parallel investigation**: Multiple team members can work different phases
- **Incremental validation**: Each script validates previous findings
- **Rollback capability**: Maintain known working configurations
- **Documentation**: All findings logged for future reference

---

## 🎯 KEY OPTIMIZATIONS BASED ON ENGINEERING FEEDBACK

1. **Overfitting Test First**: Cuts problem space in half within 6 hours
2. **Configuration Audit**: Prevents silent failures from parameter mismatches  
3. **Preprocessing Scrutiny**: Critical for pre-trained model compatibility
4. **Permanent Test Suite**: Prevents regression of discovered issues
5. **Decision Tree Approach**: Eliminates unnecessary investigation paths

---

**Status**: Plan optimized and ready for immediate implementation
**Priority**: CRITICAL - Address persistent performance ceiling  
**Timeline**: 6 hours for diagnosis, 3 days for fix implementation
**Resource Requirement**: 1 engineer, GPU access for overfitting test