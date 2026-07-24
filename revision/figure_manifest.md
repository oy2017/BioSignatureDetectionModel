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

- The buggy `plots/figure 5.png` and the stale mis-numbered `final_results/figure 1–6.png` copies have been **deleted** (they were untracked). Use only `revision/figures/`.
- **Dropped:** the standalone scree plot (`H2_pca_scree_plot_final.png`, from `generate_final_scree_plot.py`) — its explained-variance data is the top panel of Figure 3, so it is redundant.
- **Figure 8 pending regen:** the title in `plot_calibration_curves.py` has been corrected (no longer says "Biosignature Detection"), but the current `calibration_curves.png` / `figure 8.png` still shows the old title. Re-run `plot_calibration_curves.py` and re-copy to `revision/figures/figure 8.png`. The MLP/CNN curves are stochastic (SEED = 42), so do this in the final coordinated figure pass to keep the curves consistent with the reported Brier scores.

## Other pending items before submission
- **Acknowledgments** — needs Owen/mentor input on assistance to disclose (editorial, technical, analytical, writing, and any AI tools).
- **References** — carry over refs 1–42, apply the R2-3 fixes (refs 1/2, 9, 39) and add the new citations (Rackham 2018, Allard 2012, Mugnai 2020, Kempton 2017, Freedman 2008/2014, Lupu 2014, Chubb 2021, Polyansky 2018, Yurchenko 2017/2020, Jolliffe 1982, Grinsztajn 2022, Shwartz-Ziv & Armon 2022).
- **Tense/person pass** (R2-6) — past tense, third person throughout, at final assembly.
