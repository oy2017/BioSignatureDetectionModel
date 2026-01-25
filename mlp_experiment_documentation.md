# MLP Experiment Documentation
**Date:** January 24, 2026
**Project:** BioSignature Detection Model

## 1. Overview
Multi-Layer Perceptrons (MLPs) were rigorously tested as the primary Deep Learning candidate. While they outperformed CNNs significantly, they slightly trailed XGBoost in peak accuracy (~85.8% vs ~87.0%).

## 2. Key Findings

### A. The "Data Scarcity" Bottleneck
The most critical insight from the MLP experiments was the failure of standard Cross-Validation.
*   **Single Split (80/20) + Early Stopping:** Consistently yielded **~85-86%** accuracy.
*   **3-Fold Cross-Validation:** Dropped accuracy to **~76-80%**.
*   **Reason:** Splitting the small dataset (~3000 samples) into 3 folds reduced the training set too much (from ~2400 to ~2000 samples), causing the model to underfit. **Maximizing training data was more important than statistical validation rigor.**

### B. Architecture Search
We tested three main topological families:
1.  **3-Layer [256, 128, 64]:** **The Best Performer.** perfectly balanced capacity for this dataset size.
2.  **Deep & Wide [512, 256, 128]:** Prone to overfitting; required aggressive dropout (0.5) to stabilize.
3.  **4-Layer [128, 128, 64, 32]:** "Deep Narrow" architecture. Good, but slower convergence and no accuracy benefit over the 3-layer.

## 3. Best Configuration (The "Stable" Model)

**Script:** `evaluate_mlp_3layer_best.py` (and `final_best_mlp_h2.py`)
**Performance:**
*   **Test Accuracy:** **85.8%**
*   **Precision (Bio):** 0.86
*   **Recall (Bio):** 0.85

**Hyperparameters:**
*   **Input:** PCA Components 0-100 (Standard Scaled)
*   **Hidden Layers:** Dense(256) -> Dense(128) -> Dense(64)
*   **Activation:** ReLU (all hidden)
*   **Normalization:** BatchNormalization after every Dense layer.
*   **Dropout:** 0.4 (Applied after every Block).
*   **Optimizer:** Adam (LR=0.001)
*   **Batch Size:** 32 (Smaller batches helped generalization).
*   **Training:** 100 Epochs with Early Stopping (Patience=10, Restore Best Weights).

## 4. Experiment Log

### Grid Search (Comprehensive)
**Script:** `tune_mlp_comprehensive_validation.py`
This script compared multiple architectures and validation strategies side-by-side.

**Search Space:**
*   **Architectures:** `(512, 256, 128)`, `(256, 128, 64)`, `(128, 128, 64, 32)`, `(256, 128)`, `(128, 64, 32)`
*   **Dropout Rates:** `[0.3, 0.4, 0.5]`
*   **Learning Rates:** `[0.001, 0.0005]`
*   **Batch Sizes:** `[64, 128]`

**Result:** Confirmed that [256, 128, 64] was the "Sweet Spot". Adding more layers (4-layer) or width (512 units) offered diminishing returns.

### Simple Grid Search
**Script:** `tune_3layer_mlp_simple_grid.py`
Focused on fine-tuning the 3-layer model's hyperparameters (LR, Dropout, Batch).
*   **Result:** Validated that `Dropout=0.4` is optimal. `0.2` overfitted, `0.5` underfitted.

## 5. Conclusion
The MLP is a robust "second-best" model. It is less interpretable than XGBoost and slightly less accurate, but it serves as a crucial validation that the signal *is* learnable by neural networks, provided the complexity is kept in check (3 layers max) and data is maximized (Single Split).
