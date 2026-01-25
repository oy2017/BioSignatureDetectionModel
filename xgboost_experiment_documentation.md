# XGBoost Experiment Documentation
**Date:** January 24, 2026
**Project:** BioSignature Detection Model

## 1. Overview
XGBoost has consistently proven to be the **top-performing model** for biosignature detection in H2-dominated atmospheres, achieving accuracies of ~86-87%. This document summarizes the key experiments, configurations, and results.

## 2. Experiment Setup

### Data
*   **Input:** Multi-Res spectra (binned)
*   **Preprocessing:** Standard Scaling -> PCA
*   **Feature Engineering:** **Critical Step:** Discarding the first 2 Principal Components (PC0, PC1) and using the subsequent 100 components (indices 2-102).
*   **Rationale:** PC0 and PC1 likely capture dominant systematic variance (e.g., star temperature, planet radius) that obscures the subtle chemical biosignatures.

### Scripts
*   `tune_xgboost_gridsearch.py`: Comprehensive Grid Search with 3-Fold Stratified Cross-Validation.
*   `evaluate_xgboost.py`: Standalone script to train and evaluate a specific configuration on the held-out test set.
*   `analyze_error_physics.py`: Uses a stable XGBoost model to correlate errors with physical parameters.

## 3. Key Experiments & Results

### A. Grid Search (Best Configuration)
**Script:** `tune_xgboost_gridsearch.py`
**Search Space:**
*   `learning_rate`: [0.01, 0.05, 0.1, 0.2]
*   `max_depth`: [3, 5, 7, 10]
*   `n_estimators`: [100, 200, 300]
*   `subsample`: [0.8, 1.0]

**Best Found Parameters:**
*   `n_estimators`: **300**
*   `max_depth`: **5**
*   `learning_rate`: **0.2**
*   `subsample`: **1.0**

**Performance:**
*   **Test Accuracy:** **86.96%**
*   **Precision (Bio):** 0.88
*   **Recall (Bio):** 0.86
*   **F1-Score:** 0.87

### B. Stable Baseline (Used for Physics Analysis)
**Context:** A slightly more conservative model used in `analyze_error_physics.py` and `measure_model_variance.py` to ensure stability across multiple seeds.

**Parameters:**
*   `n_estimators`: **200**
*   `max_depth`: **5**
*   `learning_rate`: **0.1**

**Performance:**
*   **Test Accuracy:** ~85.5%
*   **Consistency:** Validated across 5 independent test sets with low variance (+/- 1.4%).

### C. Learning Rate Impact (Sensitivity Check)
Experiments varying only the Learning Rate (with fixed `Est=200, Depth=5`) showed:

| Learning Rate | Accuracy | Observation |
| :--- | :--- | :--- |
| **0.2** | **~87%** | Best performer, converges aggressively. |
| **0.1** | ~86% | Very stable, good balance. |
| **0.05** | ~84% | Slower convergence, slightly underfit. |
| **0.01** | ~80% | Significantly underfit. |

## 4. Physics & Chemical Error Analysis (Uncommitted Work)
Advanced analysis using `analyze_error_physics.py` and the resulting plots in `final_results/physics_analysis/` reveals specific failure modes.

### A. Chemical Thresholds
The `scatter_error_xgboost.png` (and the physics-specific version `scatter_error_chemistry.png`) shows that errors are not randomly distributed across chemical concentrations.
*   **Observation:** Errors cluster tightly around the **Log(CH4) = -6** and **Log(O3) = -7** detection boundaries.
*   **Insight:** The model has learned the "physics" of the biosignature but lacks the signal-to-noise ratio in the PCA components to resolve "borderline" cases with 100% precision.

### B. Physical Parameter Correlations
*   **Star Temperature:** Higher error rates are observed for planets orbiting very hot or very cool stars compared to Sun-like stars. This suggests that the "background" spectral noise from the star is not perfectly neutralized by the current PCA strategy.
*   **Planet Radius:** Smaller planets (lower signal-to-noise in transit spectra) show a slightly higher error rate, as expected.

## 5. Conclusion
The optimal strategy for XGBoost on this dataset is:
1.  **Features:** PCA Components 2-102.
2.  **Hyperparameters:** `n_estimators=300`, `max_depth=5`, `learning_rate=0.2`.
3.  **Result:** ~87% Accuracy.
4.  **Final Verdict:** XGBoost is the most precise model for resolving the chemical boundaries of biosignatures, though it remains sensitive to the host star's temperature.
