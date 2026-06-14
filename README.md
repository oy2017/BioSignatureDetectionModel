# A PCA-Assisted Machine Learning Framework for Biosignature Detection in High-Dimensional Exoplanet Transmission Spectra

This repository contains the complete codebase, data generation scripts, and statistical verification tools used to develop an automated triage system for the ESA Ariel mission. The pipeline evaluates the feasibility of identifying biosignatures ($CH_4$ and $O_3$) using high-dimensional spectral data at the theoretical benchmark of Ariel's Tier 3.

## 🌟 Key Findings
*   **Operational Parity:** Gradient-Boosted Trees (XGBoost) and Deep Neural Networks (MLP) achieved statistical parity in raw predictive power (~88.5% accuracy) at $R \approx 100$ equivalent resolution.
*   **Baseline Independence:** Through PCA ablation, we proved that neural networks are physically fragile—relying heavily on the broad physical continuum (Radius/Temperature)—whereas XGBoost successfully isolates the chemical absorption structure.
*   **Probability Calibration:** XGBoost demonstrated exceptional "out-of-the-box" calibration (Brier Score = 0.0719) compared to the overconfident neural networks, establishing it as the superior operational tool for prioritizing costly Tier 4 phase-curve observations.

## 🛠️ Reproducibility Guide

To reproduce the exact findings, tables, and figures reported in the manuscript, execute the following pipeline in order:

### Phase 1: Data Generation
Generate the $R \approx 100$ (200-bin) synthetic spectra using the MultiREx/TauREx 3 radiative transfer models.
```bash
# Generate the 2,700-planet training multiverse
python generate_multiverse_data.py H2

# Generate the 5 disjoint testing sets (~550 planets each)
python generate_5_sets.py
```

### Phase 2: Hyperparameter Optimization
Execute the exhaustive cross-validated grid search to establish the absolute mathematical optima (Table 2 in the manuscript) for XGBoost, Random Forest, MLP, and 1D-CNN architectures.
```bash
python run_master_gridsearch.py
```

### Phase 3: Final Model Evaluation
Train the optimized models utilizing the unified 102-component PCA feature space (with `StandardScaler` whitening) and evaluate them across the 5 independent test sets.
```bash
python run_master_5set_evaluation.py
```
*(Outputs: The final metrics used in **Table II**, and the quadrant-labeled Confusion Matrices).*

### Phase 4: Statistical Verification
Calculate local classification agreement and aggregate population stability to prove the model hierarchy.
```bash
# Calculate p-values via pairwise McNemar's Test
python run_pairwise_mcnemar.py

# Calculate 95% Confidence Intervals via 10,000 Bootstrap iterations
python verify_bootstrapping.py
```

### Phase 5: Error Analysis and Physics Visualizations
Generate the publication-ready figures mapping model performance against physical parameters and operational reliability.
```bash
# Figure 6: Reliability Diagrams and Brier Scores
python plot_calibration_curves.py

# Precision-Recall Curve for Threshold Moving (Intra-Tier 3 Triage)
python plot_precision_recall_curve.py

# Figure 4: Physical Corner Plot (Errors vs. Radius, Mass, Temperature)
python plot_all_model_corner_errors_scatter.py

# Figure 5: Chemical Corner Plot (Errors vs. CH4, O3)
python plot_chemical_scatter_comparison.py

# Mathematical confirmation of error clustering in low scale-height regimes
python verify_error_clustering.py
```

## 📊 Final Results Summary (Table II)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 88.59% (± 1.66%) | 87.93% (± 1.53%) | 89.55% (± 1.80%) | **88.73% (± 1.63%)** |
| **MLP** | 88.17% (± 0.96%) | 87.78% (± 1.54%) | 88.82% (± 1.82%) | **88.28% (± 0.94%)** |
| **Random Forest** | 86.69% (± 0.70%) | 87.03% (± 0.68%) | 86.33% (± 1.55%) | **86.67% (± 0.74%)** |
| **1D-CNN** | 81.86% (± 0.86%) | 87.04% (± 1.54%) | 75.07% (± 1.42%) | **80.59% (± 0.90%)** |

## 📖 Documentation
For a technical history of the project, including the discovery of the PCA Scaling Artifact and the deep learning tuning process, see the internal `.md` journals located in the repository root.