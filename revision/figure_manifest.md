# Figure manifest — assembly note for `revised_manuscript.md`

Production notes for assembling the submission. **Not part of the manuscript.**

The clean, correctly-numbered figure set is in **`revision/figures/figure 1.png … figure 10.png`** (numbers match the manuscript). Each was copied from the source file its script produces, and each was verified by opening the image. Every generating script is committed to the repo, so the full set is reproducible: regenerate the source file, then re-copy it to `revision/figures/figure N.png`.

| Fig | Content | Source file (script output) | Generating script (tracked) | Verified |
| :-: | :-- | :-- | :-- | :-: |
| 1 | PCA reconstruction + residuals | `final_results/H2_pca_two_panel_reconstruction_final.png` | `generate_pca_two_panel_plot.py` | ✓ |
| 2 | Confusion matrix (XGBoost) | `final_results/CM_XGBoost.png` | `run_master_5set_evaluation.py` | ✓ |
| 3 | Per-component variance / AUC / MI | `final_results/pc_discriminative_power.png` | `analyze_pc_discriminative_power.py` | ✓ |
| 4 | CH₄/O₃ imprint projection onto PCs | `final_results/chem_projection.png` | `analyze_chem_projection.py` | ✓ |
| 5 | PCA loading vectors (first six PCs) | `final_results/pc_loadings.png` | `analyze_pc_physical_drivers.py` | ✓ |
| 6 | Corner plot of errors over the independent sampled directions | `final_results/plots/figure6_independent_axes.png` | `plot_figure6_independent_axes.py` | ✓ |
| 7 | CH₄/O₃ error distribution (4 models) | `final_results/model_comparison_chemical_scatter.png` | `plot_chemical_scatter_comparison.py` | ✓ |
| 8 | Calibration / reliability curves | `final_results/calibration_curves.png` | `plot_calibration_curves.py` | ✓ |
| 9 | Precision–recall curve + operating points (clear vs 10⁴ Pa deck) | `final_results/plots/pr_curve_xgboost.png` | `plot_precision_recall_curve.py` | ✓ |
| 10 | Domain-shift accuracy vs strength (6 families) | `final_results/domain_shift_accuracy.png` | `domain_shift_sweep.py` | ✓ |

## Notes

- The standalone scree plot (`H2_pca_scree_plot_final.png`, from `generate_final_scree_plot.py`) is not used — its explained-variance data is the top panel of Figure 3.
- Figure 6 has been re-drawn twice. It was first regenerated after `fix_xgboost_corner.py` was aligned with the headline pipeline: all 102 components (it previously used PCs 2–101), 200 estimators and subsample 0.8 (previously 300 and 1.0), the five test sets pooled rather than one, and the same transit-depth cleaning as the rest of the pipeline. The version before that predated the current data files by six days and showed the physical parameters as uncorrelated clouds, which the delivered dataset does not reproduce. That regeneration exposed the sampling coupling now stated in Section 3.1, and three of its six panels were degenerate diagonals carrying no information. `plot_figure6_independent_axes.py` therefore replaces it with the directions that are actually independent — planet radius (standing for the coupled group), planet mass, and the measured per-spectrum feature amplitude on a log axis. It reports 88.58%, matching Section 4.2 and `H2_error_quintiles.txt`.
- Figure 9 is new in this revision. `plot_precision_recall_curve.py` was rewritten from a version that predated the five-set framework: it read a single legacy test file and used pre-tuning hyperparameters, and its output was never produced. It now uses the pooled five test sets, the tuned configuration of Table 2, and the same transit-depth cleaning, reproducing 88.58% at the default threshold. The domain-shifted overlay re-scores the same planets from `multirex_spectra_H2_paired_cloudy_1e4Pa.parquet`.
- Figure 8 is pooled over all five test sets (2,697 planets) and binned by equal counts, and its MLP/CNN are trained under a fixed TensorFlow seed, so `plot_calibration_curves.py` regenerates it directly (the XGBoost and Random Forest curves exactly; the neural-network curves to within the ~0.3% between-process scatter documented in `H2_mlp_reproducibility.txt`). Its statistics are written to `final_results/H2_calibration.txt` and the ECE values quoted in the caption come from that file. An earlier version of this figure used a single test set (~540 planets) with sklearn's default equal-width bins; because XGBoost's probabilities are strongly bimodal, that left 8–26 planets in the mid-range bins and produced large binomial scatter that read as miscalibration. The pooled equal-count version replaces it.
- The Brier scores quoted in Section 4.4 (XGBoost 0.0861, RF 0.1304, MLP 0.1466, 1D-CNN 0.2625) are the five-set means carried over from the original submission. XGBoost, Random Forest and MLP reproduce in `H2_calibration.txt` (0.0832 / 0.1284 / 0.1505) within the averaging convention and restart scatter. **The 1D-CNN's 0.2625 does not reproduce** — three seeded restarts give 0.1551 ± 0.0036 — and no committed script computes it (`run_master_5set_evaluation.py`, which produced Table 3, does not calculate Brier scores at all). Confirm its provenance before resubmission.
