# v2: the rebuilt study behind the NHSJS paper

All numbers in `../nhsjs/manuscript.md` come from `results/`. Python: `~/tfenv/bin/python`
(MultiREx 0.3.1 with the Mie-capable fork, TauREx 3, scikit-learn, XGBoost, TensorFlow,
python-docx). Set `OMP_NUM_THREADS=8` when running the training scripts; XGBoost with
20 threads under other load slowed 100× (see the note in `pipeline.py`).

## Pipeline, in order

| Step | Command | Output |
| :-- | :-- | :-- |
| Bin definitions | `python ariel_bins.py` | `ariel_bins.json` (Ariel delivered 102 bins; R = 100 275; R = 200 550) |
| Grid | `python generate_grid.py --n-train 20000 --n-test 2000 --n-sets 5 --jobs 12` | `data/{split}_params.parquet`, `data/{split}_native.npy`, `results/generation.txt`, `results/independence.txt` |
| Binning | `python bin_spectra.py` | `data/{split}_{ariel,r100,r200}.npy` |
| In-domain study | `python pipeline.py --config ariel` (then `--config r100 --reuse-params`, `--config r200 --reuse-params`, `--config ariel_abs50 --reuse-params`) | `results/{cfg}_indomain.txt`, `{cfg}_summary.json`, `{cfg}_probs.parquet`, `models/` |
| Label analysis | `python analyze_labels.py --config ariel` (and `ariel_abs50`) | `results/{cfg}_labels.txt/.parquet` |
| Re-renders (test splits) | `python shift_tlse.py`; `python shift_exotransmit.py --jobs 8`; `python shift_opacity.py --variant both --jobs 6`; `python shift_aerosol.py --kind both --jobs 6` (see `run_shifts.sh`) | `data/{split}_native_{axis}.npy` (20 axes × 5 splits) |
| Shift evaluation | `python evaluate_shifts.py --config ariel` | `results/ariel_shifts.txt/.csv` |
| Calibration and thresholds | `python analyze_calibration.py --config ariel` | `results/ariel_calibration.txt`, `ariel_reliability.csv`, `ariel_pr.parquet` |
| Tables and figures | `python tables.py`; `python plots/fig{1..5}_*.py` | `results/tables.md`, `results/figures/` |

Noise conventions: `common.py` selects peak-to-peak Ariel-shaped noise (SNR 15) for
plain config names and an absolute floor for names like `ariel_abs50` (50 ppm).
Test-set noise realisations are seeded per split, so every script scores the same
noisy spectra and shifted spectra get the same realisation as their clean counterpart.

External inputs: `phoenix/` (STScI BT-Settl atlas, downloaded by the session),
`~/exotransmit_src`, `~/exomolop`, `~/exomolop_o3`, and `../final_results/ariel_nsr_curves.npz`.
