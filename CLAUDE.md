# Project Status: ViT-Base Optimization

## Summary
The project has successfully moved from a critical performance bottleneck (1.3% IoU) to a promising new baseline. The root cause was identified as a failure of the ViT architecture to train from scratch on the specific dataset. The introduction of pre-trained weights has resolved this fundamental issue, establishing a new, stable baseline of **15.1% IoU**.

The current strategy is to rigorously optimize this `ViT-Base` model before exploring more complex architectures.

## Progress Breakdown

1.  **Initial State:** Model performance was critically stalled at ~1.3% IoU, failing to learn even on a small, overfittable dataset.
2.  **Root Cause Analysis:** A systematic review determined the issue was not data integrity but a fundamental problem with the training methodology. The Vision Transformer (ViT) architecture, when initialized randomly, could not converge on this highly specialized and imbalanced dataset.
3.  **The Breakthrough (Pre-trained Weights):** The hypothesis was tested by loading a `ViT-Base` model with weights pre-trained on ImageNet. This was immediately successful, breaking the performance ceiling and achieving **15.1% IoU** on the validation set.
4.  **Strategic Pivot (Methodical Optimization):** After the breakthrough, the initial plan to immediately scale up to larger models (e.g., ViT-Large) was revised. The current, more rigorous approach is to first maximize the potential of the existing `ViT-Base` architecture.

## Current Phase: Foundational Optimization

The project is now executing a two-pronged strategy to establish a fully-optimized baseline for the `ViT-Base` model.

### Phase 4A: Hyperparameter Optimization (Quantitative)
The goal is to find the optimal training configuration for the current architecture. This involves systematically testing:
-   **Learning Rate Schedules:** Cosine annealing, warm restarts, linear decay.
-   **Optimizer Settings:** AdamW parameters, weight decay, gradient clipping.
-   **Training Duration:** Extending epochs with robust early stopping.
-   **Data Augmentations:** Evaluating the impact of stronger augmentations.

### Phase 4B: Qualitative Error Analysis
Parallel to the quantitative sweep, a deep dive into the model's failure modes will be conducted to inform future architectural choices. This involves:
-   Visualizing model predictions vs. ground truth.
-   Classifying common errors (e.g., thin lanes, shadows, intersections).
-   Analyzing performance on a per-class basis.

## Next Immediate Steps
To enable the hyperparameter sweep, the core training script (`scripts/run_finetuning.py`) is currently being refactored. The hardcoded optimizer and scheduler settings are being exposed as command-line arguments, making the script fully configurable for automated, parallel experimentation.
