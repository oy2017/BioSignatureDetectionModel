# Which distribution shifts can you train away?

Anonymized code and results for the ML4PS 2026 submission *"Which distribution shifts
can you train away? A fidelity budget with a repair rule for simulator-trained spectral
classifiers."*

Every number in the paper is in a committed file here, so nothing has to be re-run to
check a claim. Regenerating the spectra takes about ten minutes; the full pipeline takes
about three hours on 22 cores.

## Claim → command → committed file

| Paper claim | Command | File |
| :-- | :-- | :-- |
| §3 Representation dominates: 90.3 / 90.1 / 89.1 % normalized; 83.9 % PCA; 72.6 % raw; McNemar p = 0.36 | `python pipeline.py --config ariel` | `results/ariel_indomain.txt`, `ariel_summary.json` |
| §3 Linear probe 85.4 %, two-band index 74.8 % | `python realism.py --config ariel` | `results/ariel_realism.txt` |
| §3 Normalization gain tracks depth spread (+31.0 / +24.8 / +19.3 by radius band) | `python prevalence.py --config ariel` | `results/ariel_prevalence.txt`, `ariel_radius_bands.csv` |
| §3 Per-channel offsets cost 10.6 points; global offset costs 0 | `python realism.py --config ariel` | `results/ariel_realism.txt` |
| §4 Table 1, the fidelity budget (all rows) | `./run_shifts.sh` then `python evaluate_shifts.py --config ariel` | `results/ariel_shifts.txt`, `ariel_shifts.csv` |
| §4 Correlated noise beats white at every kernel width; spot contrast sensitivity | `python sensitivity.py --config ariel` | `results/ariel_sensitivity.txt` |
| §5 Repair rule: 79–89 % recovered for spots and haze, 34–46 % for correlated noise | `python augment.py --render` then `--fit` | `results/ariel_augment.txt`, `ariel_augment.csv` |
| §6 Transfer: ρ = 0.973 / 0.962 / 0.881 across four pipelines | `python evaluate_shifts.py --config ariel --model {norm_mlp,norm_rf,pca_xgb}` | `results/ariel_{norm_mlp,norm_rf,pca_xgb}_shifts.csv` |
| §6 Ranking inversion under strong haze (61.5 % vs 77.8 %) | same files | `results/ariel_*_shifts.csv` |
| §7 Prevalence: 0.96 threshold holds 50 % precision at 1 % base rate | `python prevalence.py --config ariel` | `results/ariel_prevalence.txt` |
| Figure | `python plots/fig4_fidelity_sweeps.py` | `results/figures/fig4_fidelity_sweeps.png` |
| Resolution ladder (paper mentions the configuration only) | `python pipeline.py --config {r100,r200} --reuse-params` | `results/{r100,r200}_indomain.txt` |

## Reproducing

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OMP_NUM_THREADS=8          # XGBoost with 20 threads was ~100x slower under load

python ariel_bins.py                                              # bin edges, 3 configurations
python generate_grid.py --n-train 20000 --n-test 2000 --n-sets 5 --jobs 12
python bin_spectra.py
python pipeline.py --config ariel                                 # ~55 min: tuning + evaluation
./run_shifts.sh                                                   # all re-renders
python evaluate_shifts.py --config ariel
python augment.py --render --jobs 6 && python augment.py --fit
python realism.py --config ariel && python prevalence.py --config ariel
python sensitivity.py --config ariel
python plots/fig4_fidelity_sweeps.py
```

To check a number without regenerating anything, open the file named in the table.
The frozen pipelines in `v2/models/` can score any spectrum directly.

## What is here, and what is not

Committed: all code, every result file behind every number, the four figures, and the
frozen pipelines for the four compared configurations.

Not committed: the spectra themselves (2.7 GB, rebuilt by `generate_grid.py` in about
ten minutes) and the BT-Settl/PHOENIX stellar atlas (a public download from the STScI
reference atlases, named in `shift_tlse.py`). Random-forest models are omitted because
each is ~190 MB and regenerates from the tuned hyperparameters in
`results/ariel_tuning.json`.

Regenerating the *shifted* spectra additionally needs Exo-Transmit, the ExoMolOP tables,
a converted HITRAN ozone table, and a Mie-capable MultiREx fork; each script names its
own requirement in its docstring. None is needed to verify a number.

## Notes for reviewers

- The shift protocol is paired per planet: every shifted spectrum is a re-render of a
  specific test planet from its recorded parameters, and carries that planet's own noise
  realization, with sigma taken from its *clean* spectrum so a suppressed atmosphere is
  not also given less noise.
- `evaluate_shifts.py --model <key>` re-scores any frozen pipeline against the identical
  shifted data; this is how §6's transfer result is produced.
- The grid is a stress-test box with independently drawn parameters, not an
  astrophysical population. `results/independence.txt` reports the delivered
  correlations (max |r| = 0.07 between bulk parameters).
- Labels are a deterministic function of the injected abundances. The task is recovery
  of a labeling convention; the chemistry is a vehicle.
