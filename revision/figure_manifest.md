# Figure manifest — assembly note for `revised_manuscript.md`

Production notes for assembling the submission. **Not part of the manuscript.**

The clean, correctly-numbered figure set is in **`revision/figures/figure 1.png … figure 9.png`** (numbers match the manuscript). Each was copied from the source file its script produces, and each was verified by opening the image. Every generating script is committed to the repo, so the full set is reproducible: regenerate the source file, then re-copy it to `revision/figures/figure N.png`.

| Fig | Content | Source file (script output) | Generating script (tracked) | Verified |
| :-: | :-- | :-- | :-- | :-: |
| 1 | PCA reconstruction + residuals | `final_results/H2_pca_two_panel_reconstruction_final.png` | `generate_pca_two_panel_plot.py` | ✓ |
| 2 | Confusion matrix (XGBoost) | `final_results/CM_XGBoost.png` | `run_master_5set_evaluation.py` | ✓ |
| 3 | Per-component variance / AUC / MI | `final_results/pc_discriminative_power.png` | `analyze_pc_discriminative_power.py` | ✓ |
| 4 | CH₄/O₃ imprint projection onto PCs | `final_results/chem_projection.png` | `analyze_chem_projection.py` | ✓ |
| 5 | PCA loading vectors (first six PCs) | `final_results/pc_loadings.png` | `analyze_pc_physical_drivers.py` | ✓ |
| 6 | Physical-parameter corner plot (errors) | `final_results/plots/corner_plot_errors_scatter_xgboost_fixed.png` | `fix_xgboost_corner.py` | ✓ |
| 7 | CH₄/O₃ error distribution (4 models) | `final_results/model_comparison_chemical_scatter.png` | `plot_chemical_scatter_comparison.py` | ✓ |
| 8 | Calibration / reliability curves | `final_results/calibration_curves.png` | `plot_calibration_curves.py` | ✓ |
| 9 | Domain-shift accuracy vs strength (6 families) | `final_results/domain_shift_accuracy.png` | `domain_shift_sweep.py` | ✓ |

## Notes

- The standalone scree plot (`H2_pca_scree_plot_final.png`, from `generate_final_scree_plot.py`) is not used — its explained-variance data is the top panel of Figure 3.
- Figure 8 is rendered on a single test set (`multirex_spectra_H2_test.parquet`); its MLP and CNN are trained without a fixed TensorFlow seed, so the figure is one stochastic realisation and its own Brier values (XGBoost 0.090, RF 0.130, MLP 0.144, CNN 0.249) differ slightly from the reported five-set means quoted in the caption. The figure illustrates calibration shape; the caption quotes the canonical values.
