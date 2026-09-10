# Did your model learn the physics or the simulator?

Anonymized code and results for the ML4PS 2026 submission *"Did your model learn the
physics or the simulator? Intervening on a generative process, one ingredient at a
time."*

Every number in the paper is in a committed file here, so no claim requires a re-run to
check. Regenerating the spectra takes about ten minutes; the full pipeline takes about
three hours on 22 cores.

## Claim to command to file

| Paper claim | Command | File |
| :-- | :-- | :-- |
| §2 18,156 training and 9,072 test spectra; parameters drawn independently, max \|r\| = 0.07 | `python generate_grid.py` | `results/independence.txt`, `results/generation.txt` |
| §3 Table 1, the fidelity budget (all rows), against a clean baseline of 90.33 % | `./run_shifts.sh` then `python evaluate_shifts.py --config ariel` | `results/ariel_shifts.txt`, `ariel_shifts.csv` |
| §3 Radius extrapolation costs 6.2 points against a matched control | same | `results/ariel_shifts.txt` (final block) |
| §3 Correlated noise beats white at every kernel width, 14.7–19.7 vs 9.8; stellar cost 22.6–28.1 across spot contrast | `python sensitivity.py --config ariel` | `results/ariel_sensitivity.txt` |
| §3 and §5 Per-channel offsets cost 10.6 points at 2× noise; a global offset costs 0.0 | `python realism.py --config ariel` | `results/ariel_realism.txt` |
| §4 Repair table; 79–89 % recovered on six deterministic cases, 29–35 % on four stochastic ones, for at most 1.4 points of clean accuracy | `python augment.py --render` then `--fit`; `python augment_white.py` | `results/ariel_augment.txt`, `ariel_augment.csv`, `ariel_augment_white.txt` |
| §5 Transfer: ρ = 0.944 sharing a representation, 0.846 sharing a model family, 0.846 sharing neither; Mann-Whitney p = 0.009 and 0.024; amplitude proxy ρ = 0.391 over 48 cases; magnitudes spread a median 8× across pipelines | `python transfer.py --config ariel` | `results/ariel_transfer.txt`, `ariel_transfer.csv` |
| §5 Ranking inversion: 90.33 → 61.49 % normalized against 83.91 → 77.80 % PCA under haze 3×10⁷ m⁻³, raw 72.56 → 59.60 %; break-even at 28 % prevalence | `python evaluate_shifts.py --config ariel --model <key>` for each of the six cells, then `transfer.py` | `results/ariel_shifts.csv`, `ariel_{norm_rf,pca_xgb,pca_rf,raw_xgb,raw_rf}_shifts.csv` |
| §5 In-domain: 72.6 % raw, 83.9 % PCA, 90.3 ± 0.4 % normalized; MLP 73.7 → 90.1 ± 0.3 %; three families within 1.3 points, top two tied at McNemar p = 0.36; logistic floor 85.4 ± 0.4 % | `python pipeline.py --config ariel` | `results/ariel_indomain.txt`, `ariel_summary.json` |
| §5 Normalization gain tracks the spread of absolute transit depth (+31.0 at 4.35 dex, +24.8 at 2.91, +19.3 at 2.31) | `python prevalence.py --config ariel` | `results/ariel_radius_bands.csv`, `ariel_prevalence.txt` |
| Figure (printed as `budget.png`) | `python plots/fig4_fidelity_sweeps.py` | `results/figures/fig4_fidelity_sweeps.png` |

To check a number without regenerating anything, open the file named in the row.

## Beyond the paper

These ran but did not fit four pages, and are included because they bear on the claims.

| Question | Command | File |
| :-- | :-- | :-- |
| Does a cheap invertibility measure predict what augmentation buys? It does not: the ridge inverse-map R² *anti*-correlates with recovered fraction (0.74–0.77 on deterministic axes at 80–89 % recovered, 0.73–0.91 on stochastic axes at 18–29 %). | `python invertibility.py --config ariel` | `results/ariel_invertibility.txt` |
| Does the result survive at other spectral resolutions? Accuracy is nearly flat from 102 to 550 bins. | `python pipeline.py --config {r100,r200} --reuse-params` | `results/{r100,r200}_indomain.txt` |
| What precision survives a realistic base rate, and on realistic targets only? | `python prevalence.py --config ariel`, `python realism.py --config ariel` | `results/ariel_prevalence.txt`, `ariel_realism.txt` |
| What does a 50 ppm absolute noise floor do to every conclusion? | `python pipeline.py --config ariel_abs50` | `results/ariel_abs50_indomain.txt` |

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
for m in norm_rf pca_xgb pca_rf raw_xgb raw_rf; do                # the other five grid cells
  python evaluate_shifts.py --config ariel --model $m --tag $m
done
python transfer.py --config ariel
python augment.py --render --jobs 6 && python augment.py --fit && python augment_white.py
python realism.py --config ariel && python prevalence.py --config ariel
python sensitivity.py --config ariel && python invertibility.py --config ariel
python plots/fig4_fidelity_sweeps.py
```

The frozen pipelines in `v2/models/` score any spectrum directly, so a reviewer can
re-run the shift evaluation without retraining anything.

## What is here, and what is not

Committed: all code, every result file behind every number, the figure, and the frozen
pipelines for the compared configurations.

Not committed: the spectra themselves (2.7 GB, rebuilt by `generate_grid.py` in about ten
minutes) and the BT-Settl/PHOENIX stellar atlas (a public download from the STScI
reference atlases, named in `shift_tlse.py`). Random-forest models are omitted because
each is ~190 MB and regenerates from the tuned hyperparameters in
`results/ariel_tuning.json`.

Regenerating the *shifted* spectra additionally needs Exo-Transmit, the ExoMolOP tables, a
converted HITRAN ozone table, and a Mie-capable MultiREx fork; each script names its own
requirement in its docstring. None is needed to verify a number.

## Notes for reviewers

- The shift protocol is paired per planet: every shifted spectrum is a re-render of a
  specific test planet from its recorded parameters, carrying that planet's own noise
  realization, with sigma taken from its *clean* spectrum so a suppressed atmosphere is
  not also given less noise.
- Radius extrapolation is the one axis that is not a re-render. It retrains on a
  restricted grid and is scored against a control that trains on a same-size random draw
  from the whole grid and tests on the same large planets, so only the training
  distribution differs. `transfer.py` excludes it from the correlations for that reason.
- `evaluate_shifts.py --model <key>` re-scores any frozen pipeline against the identical
  shifted data. This is how §5's balanced grid of two model families crossed with three
  representations is produced.
- The grid is a stress-test box with independently drawn parameters, not an astrophysical
  population. `results/independence.txt` reports the delivered correlations.
- Labels are a deterministic function of the injected abundances. The task is recovery of
  a labeling convention; the chemistry is a vehicle.
