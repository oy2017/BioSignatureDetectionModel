# Random Forest Experiment Documentation
**Date:** January 24, 2026
**Project:** BioSignature Detection Model

## 1. Overview
Random Forest (RF) served as the primary ensemble baseline. Extensive testing confirmed it is a stable, reliable performer (~84% accuracy) but consistently trails XGBoost (~87%) and optimized MLPs (~85%).

## 2. Experiment Setup

### Data & Preprocessing
*   **Input:** PCA Components (Standard Scaled).
*   **Primary Feature Set:** Components 2-102 (Skipping PC0/PC1).
*   **Validation:** Stratified Cross-Validation (during tuning) and held-out Test Set (final evaluation).

### Scripts
*   `tune_rf_gridsearch.py`: Comprehensive Grid Search.
*   `evaluate_random_forest.py`: Flexible evaluation script for testing PCA slices.
*   `evaluate_tuned_rf.py`: Implementation of the best-found configuration (Uncommitted).

## 3. Detailed Experiment Log

### A. Grid Search (Optimization)
**Script:** `tune_rf_gridsearch.py`
**Scope:** 108 Parameter Combinations

**Search Space:**
*   **n_estimators:** `[100, 200, 300]`
*   **max_depth:** `[None, 10, 20, 30]`
*   **min_samples_split:** `[2, 5, 10]`
*   **min_samples_leaf:** `[1, 2, 4]`

**Best Configuration:**
*   `n_estimators`: **300**
*   `max_depth`: **None** (Unconstrained)
*   `min_samples_split`: **2**
*   `min_samples_leaf`: **1**
**Result:** **83.78% Accuracy**

### B. Tree Depth Sensitivity
Experiments isolating `max_depth` (with PCA 2-102) revealed that the model requires deep trees to capture the signal.

| Configuration | Accuracy | Observation |
| :--- | :--- | :--- |
| **Depth = None (Full)** | **~84%** | Best performance. Complex boundaries required. |
| Depth = 20 | ~83% | Slight degradation. |
| Depth = 10 | ~82% | Noticeable underfitting. |
| Depth = 5 | ~80% | Significant underfitting (loss of ~4% accuracy). |

### C. Feature Count Sensitivity (PCA Slices)
Experiments varying the number of PCA components (with best model settings) showed that the biosignature signal is distributed.

| PCA Slice | Components | Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| **Index 2-102** | **100** | **~84%** | Optimal balance of signal vs. noise. |
| Index 2-32 | 30 | ~78% | **Data Starvation.** 30 components are insufficient. |
| Index 2-12 | 10 | ~72% | Critical signal loss. |

### D. Signal Isolation (Start Index)
Early tests confirmed the necessity of skipping the first two components.

| Start Index | Accuracy | Observation |
| :--- | :--- | :--- |
| Index 0 | ~78% | Dominant variance (PC0/PC1) acts as noise. |
| **Index 2** | **~84%** | Clearest signal. |
| Index 10 | ~81% | Signal loss begins. |

### E. Error Analysis & Visualization (Uncommitted Work)
Recent work in `evaluate_tuned_rf.py` and `scatter_error_random_forest.png` provided deeper insights into *why* the model fails.

*   **Script:** `evaluate_tuned_rf.py` implements the clean, best-case model (`Est=300, Depth=None`) for rapid testing.
*   **Visual Findings:** The error scatter plot reveals that mistakes are not random. They cluster densely around the **chemical thresholds** (Log(CH4) ~ -6.0 and Log(O3) ~ -7.0).
*   **Implication:** The Random Forest is successfully learning the general rules but struggles with precision at the exact "boundary conditions" where a planet transitions from non-bio to bio. This suggests that "hard negatives" near the boundary are the main source of error.

## 4. Conclusion
*   **Stability:** Random Forest is extremely stable across runs but has a "performance ceiling" of ~84% on this dataset.
*   **Requirements:** It demands **deep trees** and a **wide feature set** (100+ components). Shallow trees or aggressive dimensionality reduction (e.g., only 30 components) cause immediate performance drops.
