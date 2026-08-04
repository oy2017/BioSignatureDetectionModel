# A Calibrated, Robustness-Characterized Machine-Learning Benchmark for Threshold-Based Biosignature Labeling of Synthetic Exoplanet Transmission Spectra

This repository contains the complete codebase, data, and analysis scripts behind a benchmark of supervised classifiers for threshold-based biosignature *labelling* of synthetic exoplanet transmission spectra. The pipeline evaluates whether a model can recover a predefined $CH_4$ / $O_3$ abundance-threshold labelling from high-dimensional spectra simulated at Ariel Tier 3 benchmark resolution ($R \approx 200$), and — the focus of the revision — characterises how that classifier degrades under seven axes of domain shift.

> **Status:** the associated manuscript is under revision following peer review (Journal of High School Science). The numerical results below are current. The response to reviewers and the revised manuscript live in `revision/`.

## 🌟 Key Findings

*   **Model hierarchy:** Gradient-boosted trees (XGBoost) achieved the strongest overall performance at 88.17% accuracy, ahead of Random Forest (86.51%), the MLP (79.34%), and the 1D-CNN (75.53%), all evaluated on a shared 102-component PCA feature space across five independent test sets.
*   **Probability calibration:** XGBoost produced the best-calibrated probability estimates (Brier score 0.0861, near-diagonal reliability curve, ECE 0.032) against Random Forest (0.1304), MLP (0.1466), and 1D-CNN (0.2625), making its confidence estimates the most usable for ranking candidates.
*   **Variance rank is decoupled from discriminative rank:** the first two principal components carry 98.41% of the spectral variance but classify at chance when used alone. Removing them costs no performance. Discriminative information is distributed thinly across the low-variance tail.
*   **Whitening is an optimisation remedy, not a feature selector:** XGBoost is unaffected by the post-PCA `StandardScaler`; the MLP's dependence on it disappears if the two uninformative high-variance components are dropped instead; only the 1D-CNN still requires it.
*   **Graceful, traceable domain shift:** across seven perturbation axes the frozen pipeline degrades smoothly and reduces to three measured mechanisms — feature-amplitude suppression, chromatic distortion, and a decision-threshold bias — rather than failing opaquely (Section 4.6).

## ⚙️ Installation & Setup

**Python 3.10 is required** — TensorFlow 2.21.0 publishes wheels for cp310–cp313 only. All package versions in `requirements.txt` are pinned to those reported in the manuscript; results are sensitive to them (in particular, XGBoost uses `subsample=0.8`, so accuracy shifts ~0.4 pts with training-row ordering even at a fixed `random_state` — prefer a mean over restarts).

```bash
# Clone the repository
git clone https://github.com/oy2017/BioSignatureDetectionModel.git
cd BioSignatureDetectionModel

# Create and activate a virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all pinned packages (includes MultiREx + TauREx 3, scikit-learn,
# xgboost, and tensorflow/keras)
pip install -r requirements.txt

# Verify MultiREx
python -c "import multirex; print(f'MultiREx version: {multirex.__version__}')"
```

## 📦 Data Availability

**All datasets are committed to the repository**, so every *evaluation and analysis* command below runs directly without regenerating data:

| File(s) | Contents |
| :-- | :-- |
| `multirex_spectra_H2_train.parquet` | 2,696 training spectra (from 3,000, post-cleaning) |
| `multirex_spectra_H2_test_set_{1..5}.parquet` | five independent test sets (~540 each; the primary evaluation data) |
| `multirex_spectra_H2_test.parquet` | single pooled test set (used by a few older figure/stat scripts) |
| `multirex_spectra_H2_{cloudy,hazy}_*.parquet` | §4.6 aerosol test sets (fresh draws) |
| `multirex_spectra_H2_paired_{cloudy,hazy}_*.parquet` | §4.6 aerosol test sets (paired re-renders of the committed planets) |
| `multirex_spectra_H2_exotransmit_set_{1..5}.parquet` | §4.6 independent-RT-code (Exo-Transmit) re-renders |
| `multirex_spectra_H2_opacityswap_set_{1..5}[_o3].parquet` | §4.6 alternative-opacity (ExoMol/HITRAN) re-renders |
| `final_results/ariel_nsr_curves.npz` | §4.6 ExoRad2/Ariel noise-to-signal curves |

The `generate_*` commands below **regenerate** these from scratch and are only needed if you want to rebuild the data. Several of them require external tools installed at machine-specific paths (Exo-Transmit, the ExoMolOP/HITRAN opacity tables, a PHOENIX spectral grid, ExoRad2, and cloud/haze MultiREx forks); those prerequisites are flagged inline. Reproducing the paper's numbers does **not** require them — run the `evaluate_*` scripts on the committed data.

---

## 🛠️ Reproducibility Guide

Commands run from the repo root. Unless noted, scripts take no arguments (`fill_gas` is hardcoded to `H2`) and write their outputs under `final_results/`.

### Phase 0 — (Optional) Data generation

```bash
# Training set + a single pooled test set
python generate_multiverse_data.py H2 --purpose train
python generate_multiverse_data.py H2 --purpose test   # creates multirex_spectra_H2_test.parquet

# The five independent test sets (calls the generator internally)
python generate_5_sets.py
```

### Phase 1 — Model tuning & headline performance (Table 3, Figure 2)

```bash
# Exhaustive cross-validated grid search for all four models (one-time; prints
# the optimal hyperparameters that are hardcoded into the evaluation scripts)
python run_master_gridsearch.py

# Train the tuned models on the 102-component PCA space, evaluate over 5 sets.
# -> Table 3 metrics (stdout) + final_results/CM_{XGBoost,Random_Forest,MLP,CNN}.png
#    CM_XGBoost.png is Figure 2.
python run_master_5set_evaluation.py
```

### Phase 2 — Statistical significance

```bash
python run_pairwise_mcnemar.py     # pairwise McNemar tests (reads multirex_spectra_H2_test.parquet)
python verify_bootstrapping.py     # 10k-bootstrap CI on the XGBoost-vs-RF F1 gap
```

### Phase 3 — Calibration & triage reliability (Figure 8, Section 4.4)

```bash
# Reliability diagrams + ECE/Brier. -> final_results/calibration_curves.png (Figure 8)
#                                       final_results/H2_calibration.txt
python plot_calibration_curves.py

# Precision-recall curve for threshold moving (Section 4.5)
python plot_precision_recall_curve.py
```

### Phase 4 — PCA feature-space analysis (Section 4.2; Figures 1, 3, 4, 5)

```bash
# Figure 1  — PCA reconstruction + residuals
python generate_pca_two_panel_plot.py            # -> final_results/H2_pca_two_panel_reconstruction_final.png

# Figure 3  — per-component variance / AUC / mutual information
python analyze_pc_discriminative_power.py        # -> pc_discriminative_power.png + H2_pc_discriminative_power.txt

# Figure 4  — CH4/O3 imprint projection onto the components
python analyze_chem_projection.py                # -> chem_projection.png + H2_chem_projection.txt

# Figure 5  — PCA loading vectors of the first six components
python analyze_pc_physical_drivers.py            # -> pc_loadings.png + H2_pc_drivers.txt

# Supporting ablations (the four analyses requested by the reviewer)
python ablate_pc_ranges.py                       # -> H2_pc_range_ablation.txt   (selective component removal)
python test_whitening_necessity.py               # -> H2_whitening_necessity.txt (whitening on/off; add --quick for a fast pass)
python supervised_dr_comparison.py               # -> H2_supervised_dr.txt       (PCA vs PLS/LDA at matched dim)
python compare_feature_weighting.py              # -> H2_feature_weighting.txt   (alternative per-component weightings)
python analyze_o3_band_occlusion.py              # -> H2_o3_band_occlusion.txt
```

### Phase 5 — Error analysis (Section 4.3; Figures 6, 7)

```bash
# Figure 6 — physical-parameter error corner plot
python fix_xgboost_corner.py                     # -> final_results/plots/corner_plot_errors_scatter_xgboost_fixed.png

# Figure 7 — CH4/O3 error distribution across the four models
python plot_chemical_scatter_comparison.py       # -> model_comparison_chemical_scatter.png (reads multirex_spectra_H2_test.parquet)

# Error rate across physical parameter space (quintiles)
python analyze_error_quintiles.py                # -> H2_error_quintiles.txt
python analyze_error_quintiles_noise.py          # -> H2_error_quintiles_noise.txt (the same binning under injected noise, §4.6)

# Labelling-convention analyses (reviewer R1-4)
python analyze_label_margin.py                   # -> H2_label_margin.txt        (accuracy vs distance to threshold)
python analyze_threshold_sensitivity.py          # -> H2_threshold_sensitivity.txt (moving both cutoffs ±0.5 dex)

# Neural-network run-to-run scatter (reviewer R1-6/R1-8)
python measure_mlp_reproducibility.py            # -> H2_mlp_reproducibility.txt
```

### Phase 6 — Robustness under domain shift (Section 4.6; Table 5, Figure 9)

All `evaluate_*` scripts run on the **committed** perturbed test sets. Each `generate_*` step (which rebuilds that data and needs the external tool noted) is optional.

```bash
# Figure 9 + injected-systematics sweep (axis 5: resolution/SNR, offset, gain,
# 1/λ contamination, white & correlated noise). -> domain_shift_accuracy.png + H2_domain_shift_sweep.txt
python domain_shift_sweep.py

# Axis 7 — out-of-envelope (radius) extrapolation split. -> H2_extrapolation_split.txt
python domain_shift_sweep.py --mode extrapolation

# --- Axis 1: independent radiative-transfer code (Exo-Transmit) ---
# (optional regen) python generate_exotransmit_testset.py   # needs Exo-Transmit at ~/exotransmit_src
python evaluate_exotransmit.py                   # -> H2_exotransmit.txt
python compare_whitening_exotransmit.py          # -> H2_whitening_exotransmit.txt (whitened vs unwhitened MLP under code change)

# --- Axis 2: alternative molecular opacity (ExoMol line lists; + HITRAN O3) ---
# (optional regen) python convert_hitran_o3_to_taurex.py            # HITRAN O3 -> TauREx HDF5 (~/exomolop_o3)
# (optional regen) python generate_opacity_swap_testset.py --validate
# (optional regen) python generate_opacity_swap_testset.py --generate            # 3-molecule arm  (needs ~/exomolop)
# (optional regen) python generate_opacity_swap_testset.py --generate --with-o3  # 4-molecule arm
python evaluate_opacity_swap.py                  # -> H2_opacity_swap.txt      (H2O/CH4/CO2 swapped)
python evaluate_opacity_swap.py --o3             # -> H2_opacity_swap_o3.txt   (ozone swapped too)

# --- Axis 3: clouds and hazes ---
# (optional regen) python generate_cloudy_testset.py    # needs the grey-deck MultiREx fork
python evaluate_cloudy.py                        # -> H2_cloudy_evaluation.txt  (writes the CSV that evaluate_hazy reads)
# (optional regen) python generate_hazy_testset.py      # needs the Lee-Mie MultiREx fork
python evaluate_hazy.py                          # -> H2_hazy_evaluation.txt    (run AFTER evaluate_cloudy.py)
# Paired aerosol re-renders (no draw confound), with McNemar flip breakdown:
# (optional regen) python generate_aerosol_paired.py
python evaluate_aerosol_paired.py                # -> H2_aerosol_paired.txt

# --- Axis 4: physical stellar contamination (Transit Light Source Effect) ---
# Requires a PHOENIX/BT-Settl FITS grid; edit PHX_DIR at the top of the script to point at it.
python evaluate_spots_phoenix.py                 # -> H2_spots_phoenix.txt

# --- Axis 6: realistic instrument noise (ExoRad2 / Ariel) ---
# Uses the committed final_results/ariel_nsr_curves.npz. To regenerate it:
#   (from ariel_noise_model/) run ExoRad2 on ariel_payload.xml, then: python build_ariel_nsr.py
python evaluate_ariel_noise.py                   # -> H2_ariel_noise.txt

# Whitening-robustness error bars over 5 MLP restarts (reviewer R1-8)
python domain_shift_mlp_restarts.py              # -> H2_whitening_restarts.txt
```

**Dependency notes for Phase 6:**
- `evaluate_hazy.py` reads `final_results/H2_cloudy_evaluation.csv`, so **run `evaluate_cloudy.py` first**.
- Every `evaluate_*` axis is otherwise independent and can run in any order on the committed data.
- The `generate_*` steps and the stellar/opacity axes rely on **machine-specific external paths** (`~/exotransmit_src`, `~/exomolop`, `~/exomolop_o3`, a PHOENIX FITS grid, an ExoRad2 environment). These are needed only to rebuild the perturbed data, not to reproduce the reported numbers.

---

## 🗺️ Figure & output provenance

The clean, correctly-numbered figure set is in `revision/figures/figure 1.png … figure 9.png`. Each is copied from a source file its script produces (see `revision/figure_manifest.md`):

| Figure | Generating command | Source file |
| :-: | :-- | :-- |
| 1 | `generate_pca_two_panel_plot.py` | `final_results/H2_pca_two_panel_reconstruction_final.png` |
| 2 | `run_master_5set_evaluation.py` | `final_results/CM_XGBoost.png` |
| 3 | `analyze_pc_discriminative_power.py` | `final_results/pc_discriminative_power.png` |
| 4 | `analyze_chem_projection.py` | `final_results/chem_projection.png` |
| 5 | `analyze_pc_physical_drivers.py` | `final_results/pc_loadings.png` |
| 6 | `fix_xgboost_corner.py` | `final_results/plots/corner_plot_errors_scatter_xgboost_fixed.png` |
| 7 | `plot_chemical_scatter_comparison.py` | `final_results/model_comparison_chemical_scatter.png` |
| 8 | `plot_calibration_curves.py` | `final_results/calibration_curves.png` |
| 9 | `domain_shift_sweep.py` | `final_results/domain_shift_accuracy.png` |

The `final_results/*.txt` files hold the numbers quoted in the text; each is written by the correspondingly named script in the phases above (e.g. `H2_domain_shift_sweep.txt`, `H2_exotransmit.txt`, `H2_opacity_swap.txt`, `H2_cloudy_evaluation.txt`, `H2_hazy_evaluation.txt`, `H2_spots_phoenix.txt`, `H2_ariel_noise.txt`, `H2_calibration.txt`, `H2_label_margin.txt`, `H2_threshold_sensitivity.txt`).

---

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

---

## 📁 Repository layout & notes

- `revision/` — the revised manuscript (`revised_manuscript.md`), the response to reviewers, and the figure manifest.
- `final_results/` — text/CSV outputs and the nine figure-source PNGs.
- `ariel_noise_model/` — the reconstructed Ariel payload (`ariel_payload.xml`) and the ExoRad2 NSR builder for the instrument-noise axis.
- Additional diagnostic scripts (`analyze_pca_variance.py`, `analyze_pca_correlation.py`, `verify_pca_physics.py`, `calculate_feature_variance.py`, `plot_all_model_corner_errors*.py`, `generate_separate_*_plots.py`, `plot_final_results.py`, etc.) predate the canonical figure scripts and are **not** needed to reproduce the paper; they are kept for provenance.

All simulated radiative-transfer physics can be reproduced with the public MultiREx and TauREx 3 libraries pinned in `requirements.txt`.
