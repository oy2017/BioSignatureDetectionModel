# Random Forest Experiment Documentation
**Date:** January 24, 2026 (Updated Feb 13, 2026)
**Project:** BioSignature Detection Model

## 1. Overview
Random Forest (RF) served as the primary ensemble baseline. After extensive data cleaning (removing values > 1.0) and re-tuning, Random Forest has emerged as a top-tier model, achieving **88.89% Accuracy**, significantly improving upon its previous ~84% benchmark.

## 2. Experiment Setup

### Data & Preprocessing
*   **Input:** PCA Components (Standard Scaled).
*   **Primary Feature Set:** Components 2-102 (Skipping PC0/PC1).
*   **Cleaning:** Rows with transit depths > 1.0 or NaNs were strictly filtered out.
*   **Validation:** Stratified Cross-Validation (during tuning) and held-out Test Set (final evaluation).

### Scripts
*   `tune_rf_gridsearch.py`: Comprehensive Grid Search.
*   `evaluate_random_forest.py`: Flexible evaluation script for testing PCA slices.
*   `evaluate_tuned_rf.py`: Implementation of the best-found configuration.

## 3. Detailed Experiment Log

### A. Grid Search (Optimization) - **Clean Data Run**
**Date:** Feb 13, 2026
**Script:** `tune_rf_gridsearch.py`
**Scope:** 108 Parameter Combinations

**Best Configuration:**
*   `n_estimators`: **300**
*   `max_depth`: **None** (Unconstrained)
*   `min_samples_split`: **5**
*   `min_samples_leaf`: **2**
*   **Result:** **88.89% Accuracy**

### B. Tree Depth Sensitivity
Experiments isolating `max_depth` (with PCA 2-102) confirmed that the model requires deep trees to capture the signal.

| Configuration | Accuracy | Observation |
| :--- | :--- | :--- |
| **Depth = None (Full)** | **88.89%** | Best performance. Complex boundaries required. |
| Depth = 20 | ~83% | Slight degradation. |
| Depth = 10 | ~82% | Noticeable underfitting. |
| Depth = 5 | ~80% | Significant underfitting (loss of ~4% accuracy). |

### C. Feature Count Sensitivity (PCA Slices)
Experiments varying the number of PCA components (with best model settings) showed that the biosignature signal is distributed.

| PCA Slice | Components | Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| **Index 2-102** | **100** | **~88%** | Optimal balance of signal vs. noise. |
| Index 2-32 | 30 | ~78% | **Data Starvation.** 30 components are insufficient. |
| Index 2-12 | 10 | ~72% | Critical signal loss. |

### D. Signal Isolation (Start Index)
Early tests confirmed the necessity of skipping the first two components.

| Start Index | Accuracy | Observation |
| :--- | :--- | :--- |
| Index 0 | ~78% | Dominant variance (PC0/PC1) acts as noise. |
| **Index 2** | **~88%** | Clearest signal. |
| Index 10 | ~81% | Signal loss begins. |

## 4. Conclusion
*   **Performance:** Random Forest is now a leading model (88.89%), potentially outperforming the CNN (85%) and matching optimized MLPs.
*   **Robustness:** The model benefited significantly from the removal of invalid physical data (transit depths > 1.0).
*   **Requirements:** It demands **deep trees** and a **wide feature set** (100+ components).