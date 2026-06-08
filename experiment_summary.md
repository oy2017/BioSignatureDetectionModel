# Project Technical Summary: Biosignature Detection via Machine Learning

## 1. Overview of Methodology
This project developed a machine learning pipeline to detect atmospheric biosignatures (methane and ozone disequilibrium) in high-dimensional exoplanet transmission spectra simulated at an Ariel-like resolving power of 200. The final pipeline utilizes Principal Component Analysis (PCA) for signal isolation and evaluates four primary architectures: XGBoost, Random Forest, Multi-Layer Perceptron (MLP), and 1D-Convolutional Neural Network (CNN).

## 2. Evolution of the Data Pipeline

### Phase 1: Exploratory Development
Initial experiments used separate scripts for each model. A "Skip PC0-PC1" strategy was identified as critical for isolating chemical features from physical systematics. 
*   **Result:** Early MLP and CNN models achieved ~83-87% accuracy.
*   **Artifacts:** `evaluate_deep_learning.py`, `tune_mlp_gridsearch.py`.

### Phase 2: Unified Master Pipeline & Bug Fix
A rigorous audit of the Phase 1 scripts revealed a **StandardScaler Scaling Bug**. The old scripts applied a second `StandardScaler` to PCA subsets *after* slicing, which artificially inflated the amplitude of subtle chemical wiggles.
*   **The Fix:** A unified Master Evaluation script (`run_master_5set_evaluation.py`) was implemented. It fit PCA once and extracted slices without redundant scaling, ensuring the models see physically realistic feature variances.
*   **Discovery of Baseline Dependence:** Removing the scaling bug revealed that while XGBoost is **baseline-independent** (accuracy remained high without PC0/1), Neural Networks are **baseline-dependent**. Without PC0 and PC1 (planetary radius and temperature) to orient the gradient descent, MLP/CNN accuracy collapsed to ~55%.

## 3. Final Model Configurations (GridSearch Winners)

Through an exhaustive master grid search and 5-set validation, the following optimal configurations were locked for the final paper:

| Model | PCA Subset | Hyperparameters |
| :--- | :--- | :--- |
| **XGBoost** | PC 2-101 | Est=300, Depth=5, LR=0.1, Subsample=0.8 |
| **Random Forest** | PC 2-101 | Est=300, Depth=None, Leaf=2, Split=2 |
| **MLP** | PC 0-101 | 256-128-64 neurons, Dropout=0.3, LR=0.0005, Batch=128 |
| **1D-CNN** | PC 0-101 | 64/128 filters, Kernel=5, Dropout=0.3, LR=0.001, Batch=64 |

## 4. Definitive Results (Table II)
Evaluated across 5 independent test sets (N~550 each) using the rigorous pipeline:

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 88.59% (± 1.66%) | 87.93% (± 1.53%) | 89.55% (± 1.80%) | **88.73% (± 1.63%)** |
| **Random Forest** | 86.69% (± 0.70%) | 87.03% (± 0.68%) | 86.33% (± 1.55%) | **86.67% (± 0.74%)** |
| **MLP** | 88.17% (± 0.96%) | 87.78% (± 1.54%) | 88.82% (± 1.82%) | **88.28% (± 0.94%)** |
| **CNN** | 81.86% (± 0.86%) | 87.04% (± 1.96%) | 75.07% (± 1.42%) | **80.59% (± 0.90%)** |

## 5. Physical Insights for the Paper
*   **Pearson Correlation:** Confirmed PC0 is a proxy for Planetary Radius ($r=0.961$) and PC1 captures Temperature. Both have near-zero correlation with chemical targets.
*   **Model Robustness:** XGBoost is recommended for the Ariel mission because it can detect biosignatures strictly from relative chemical morphology, making it immune to the absolute continuum errors common in real telescope data. MLP, while equally accurate, is more fragile due to its dependence on the absolute baseline.
