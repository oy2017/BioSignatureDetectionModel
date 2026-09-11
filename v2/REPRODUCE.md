# Reproducing every number in the paper

Paper: *Which simulation choices matter for a machine-learning triage classifier of
Ariel-like exoplanet transmission spectra* (submitted to NHSJS). Everything below runs
from this directory.

## Quickest check (no forward model, ~2 minutes)

The result files behind every number in the paper are committed under `results/`. To
confirm a value without regenerating anything, open the file named in the table below.
To re-derive the headline table and figures from the committed models and spectra:

```bash
export OMP_NUM_THREADS=8
python tables.py                       # Table 2, the resolution ladder, Table 3
python plots/fig2_ladder_calibration.py
```

## Full rebuild (~3 hours on 22 cores)

```bash
python ariel_bins.py                                        # bin edges for the three configurations
python generate_grid.py --n-train 20000 --n-test 2000 --n-sets 5 --jobs 12   # ~10 min
python bin_spectra.py
python pipeline.py --config ariel                           # tuning + evaluation, ~55 min
python pipeline.py --config r100 --reuse-params --mlp-restarts 3
python pipeline.py --config r200 --reuse-params --mlp-restarts 3
python pipeline.py --config ariel_abs50 --reuse-params --mlp-restarts 3
python analyze_labels.py --config ariel
python analyze_labels.py --config ariel_abs50
./run_shifts.sh                                             # all re-renders, ~35 min
python evaluate_shifts.py --config ariel
python analyze_calibration.py --config ariel
python physics_baseline.py --config ariel
python augment.py --render --jobs 6 && python augment.py --fit
python tables.py && for f in plots/fig*.py; do python $f; done
```

`OMP_NUM_THREADS=8` matters: XGBoost with 20 threads on a loaded machine was ~100x
slower than with 8 in our runs.

## Claim → command → committed evidence

| Claim in the paper | Command | Result file |
| :-- | :-- | :-- |
| Grid sizes; 91 % of draws survive cleaning; acceptance by mass decile | `generate_grid.py` | `results/generation.txt` |
| Parameters are independent (max abs r = 0.07) | `generate_grid.py` | `results/independence.txt` |
| Binning reproduces MultiREx's own binner | `generate_grid.py --check` | stdout |
| **Table 2**, all nine model × feature rows; McNemar; bootstrap | `pipeline.py --config ariel` | `results/ariel_indomain.txt`, `ariel_summary.json` |
| Normalized XGBoost 90.33 ± 0.42 %, MLP 90.14 %, RF 89.08 % | same | `results/ariel_indomain.txt` |
| **Resolution ladder** 90.3 / 90.8 / 92.1 % | `pipeline.py --config r100 / r200 --reuse-params` | `results/r100_indomain.txt`, `r200_indomain.txt`, `tables.md` |
| Absolute 50 ppm convention: 87.0 % and its error structure | `pipeline.py --config ariel_abs50`; `analyze_labels.py --config ariel_abs50` | `results/ariel_abs50_indomain.txt`, `ariel_abs50_labels.txt` |
| Threshold sensitivity (89.5–91.6 % over ±0.5 dex) | `analyze_labels.py --config ariel` | `results/ariel_labels.txt` §1 |
| Margin analysis (65.5 % within 0.25 dex) | same | `results/ariel_labels.txt` §2 |
| Amplitude quintiles under both noise conventions | same, both configs | `results/ariel_labels.txt` §3, `ariel_abs50_labels.txt` §3 |
| **Table 3**, all 20 re-rendered cases and injected families; extrapolation | `run_shifts.sh` then `evaluate_shifts.py --config ariel` | `results/ariel_shifts.txt`, `ariel_shifts.csv` |
| Exo-Transmit −1.0; ExoMol −1.7; +HITRAN O₃ −15.3 | `shift_exotransmit.py`, `shift_opacity.py`, `evaluate_shifts.py` | `results/ariel_shifts.txt` |
| Clouds and hazes, five levels each | `shift_aerosol.py`, `evaluate_shifts.py` | `results/ariel_shifts.txt` |
| Stellar spots and faculae, seven cases | `shift_tlse.py`, `evaluate_shifts.py` | `results/ariel_shifts.txt` |
| Calibration, reliability curves, ECE per model | `analyze_calibration.py --config ariel` | `results/ariel_calibration.txt`, `ariel_reliability.csv` |
| Operating thresholds, clean vs cloud deck | same | `results/ariel_calibration.txt` |
| **Band-depth baseline**: index 61.5 %, index+XGBoost 66.0 % | `physics_baseline.py --config ariel` | `results/ariel_baseline.txt` |
| **Augmentation recovery** for spots, haze, correlated noise | `augment.py --render` then `--fit` | `results/ariel_augment.txt`, `ariel_augment.csv` |
| **Figures 1–5** | `plots/fig1..fig5*.py` | `results/figures/` |

## What is and is not committed

Committed: all code, `results/` (every text, CSV and figure file behind the paper),
`models/` (the frozen pipelines), and `ariel_bins.json`.

Not committed: `data/` (2.3 GB of spectra) and `phoenix/` (the stellar atlas).
`generate_grid.py` rebuilds `data/` in about ten minutes; the PHOENIX atlas is a public
download from the STScI reference atlases, named in `shift_tlse.py`.

## External tools the re-render scripts need

Only for regenerating shifted spectra, not for checking any number:
Exo-Transmit (`~/exotransmit_src`), the ExoMolOP tables (`~/exomolop`), a converted
HITRAN ozone table (`~/exomolop_o3`), and a Mie-capable MultiREx fork for the aerosol
prescriptions. Each script names its requirement in its docstring.
