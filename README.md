# A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra

This repository contains the complete codebase, data generation scripts, and statistical verification tools behind a benchmark of supervised classifiers for biosignature-candidate screening in synthetic exoplanet transmission spectra. The pipeline evaluates whether a model can recover a predefined $CH_4$ / $O_3$ abundance-threshold labelling from high-dimensional spectra simulated at Ariel Tier 3 benchmark resolution.

> **Status:** the associated manuscript is under revision following peer review. Sections describing the interpretation of the PCA feature space are being revised; the numerical results below are current.

## 🌟 Key Findings

*   **Model hierarchy:** Gradient-boosted trees (XGBoost) achieved the strongest overall performance at 88.17% accuracy, ahead of Random Forest (86.51%), the MLP (79.34%), and the 1D-CNN (75.53%), all evaluated on a shared 102-component PCA feature space across five independent test sets.
*   **Probability calibration:** XGBoost produced the best-calibrated probability estimates (Brier score 0.0861, near-diagonal reliability curve) against Random Forest (0.1304), MLP (0.1466), and 1D-CNN (0.2625), making its confidence estimates the most usable for ranking candidates.
*   **Variance rank is decoupled from discriminative rank:** the first two principal components carry 98.41% of the spectral variance but classify at chance when used alone (52.1% accuracy, single-feature AUC 0.506 and 0.530). Removing them costs no performance. Discriminative information is distributed thinly across the low-variance tail, and 14 of the 20 most informative components fall outside the 20 highest-variance ones.
*   **Whitening is an optimisation remedy, not a feature selector:** XGBoost is provably unaffected by the post-PCA `StandardScaler` (identical metrics with and without it, at every component range tested). The MLP's dependence on whitening disappears entirely if the two uninformative high-variance components are simply dropped instead (79.49% unwhitened on PCs 2–101, versus 78.76% whitened on PCs 0–101). The 1D-CNN still requires it.

## ⚙️ Installation & Setup

**Python 3.10 is required** — TensorFlow 2.21.0 publishes wheels for cp310–cp313 only. All package versions in `requirements.txt` are pinned to those reported in the manuscript; results are sensitive to them.

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/oy2017/BioSignatureDetectionModel.git
cd BioSignatureDetectionModel

# Create and activate a virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all pinned packages
pip install -r requirements.txt
```

### 2. Verify MultiREx Installation

```bash
python -c "import multirex; print(f'MultiREx version: {multirex.__version__}')"
```

## 🛠️ Reproducibility Guide

Execute the pipeline in order to reproduce the manuscript's tables and figures.

### Phase 1: Data Generation

Generate the 550-bin synthetic spectra spanning 0.5–7.8 μm using the MultiREx / TauREx 3 radiative transfer models. At 550 bins across this range the effective resolving power is $R \approx 200$, matching the upper end of Ariel's Tier 3 benchmark.

```bash
# Generate the 3,000-planet training multiverse (2,696 survive cleaning)
python generate_multiverse_data.py H2

# Generate the 5 disjoint testing sets (~540 planets each)
python generate_5_sets.py
```

### Phase 2: Hyperparameter Optimization

Exhaustive cross-validated grid search establishing the optima reported in Table 2 for XGBoost, Random Forest, MLP, and 1D-CNN.

```bash
python run_master_gridsearch.py
```

### Phase 3: Final Model Evaluation

Train the optimized models on the unified 102-component PCA feature space and evaluate across the five independent test sets.

```bash
python run_master_5set_evaluation.py
```

*(Outputs: the metrics in **Table 3**, plus quadrant-labeled confusion matrices.)*

### Phase 4: Statistical Verification

```bash
# Pairwise McNemar's tests
python run_pairwise_mcnemar.py

# 95% confidence intervals via 10,000 bootstrap iterations
python verify_bootstrapping.py
```

### Phase 5: Error Analysis and Physics Visualizations

```bash
# Reliability diagrams and Brier scores
python plot_calibration_curves.py

# Precision-recall curve for threshold moving
python plot_precision_recall_curve.py

# Physical corner plot (errors vs. radius, mass, temperature)
python plot_all_model_corner_errors_scatter.py

# Chemical corner plot (errors vs. CH4, O3)
python plot_chemical_scatter_comparison.py

# Error clustering in low scale-height regimes
python verify_error_clustering.py
```

### Phase 6: Feature-Space Analysis

Characterises where label-discriminative information actually resides in the PCA feature space, and whether the whitening step is necessary.

```bash
# Retrain on restricted PCA index ranges (selective component removal)
python ablate_pc_ranges.py

# Per-component AUC and mutual information vs. explained variance
python analyze_pc_discriminative_power.py

# Cross {component range} x {whitening on/off} for MLP, CNN, and an
# XGBoost scale-invariance control
python test_whitening_necessity.py
```

## 📊 Final Results Summary (Table 3)

Mean ± standard deviation across the five independent test sets.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 88.17% (± 1.99%) | 88.02% (± 1.30%) | 88.50% (± 3.45%) | **88.24% (± 2.22%)** |
| **Random Forest** | 86.51% (± 1.96%) | 87.17% (± 2.28%) | 85.85% (± 2.74%) | **86.48% (± 2.06%)** |
| **MLP** | 79.34% (± 1.47%) | 72.30% (± 1.52%) | 95.57% (± 1.46%) | **82.32% (± 1.29%)** |
| **1D-CNN** | 75.53% (± 1.05%) | 68.72% (± 0.79%) | 94.33% (± 1.61%) | **79.51% (± 0.76%)** |
| MLP (unwhitened) | 74.81% (± 2.61%) | 83.83% (± 3.11%) | 61.80% (± 3.42%) | 71.14% (± 3.35%) |
| 1D-CNN (unwhitened) | 64.51% (± 0.79%) | 58.97% (± 0.66%) | 96.91% (± 0.84%) | 73.32% (± 0.48%) |
| 1D-CNN (raw spectra) | 50.56% (± 1.58%) | 50.42% (± 1.01%) | 85.01% (± 2.00%) | 63.29% (± 1.25%) |

**Reproducibility note:** XGBoost uses `subsample=0.8`, so the sampled rows depend on training-row ordering even at a fixed `random_state`. Accuracy varies by roughly 0.4 percentage points across shuffles; prefer a mean over several restarts to a single run.

## 📖 Documentation

For a technical history of the project, including the deep learning tuning process, see the `.md` journals in the repository root. Outputs from the feature-space analyses in Phase 6 are written to `final_results/`.
