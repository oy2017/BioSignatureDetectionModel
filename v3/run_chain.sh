#!/usr/bin/env bash
# Post-tuning analysis chain for the v3 (C/O-label) study, in dependency order.
# Prerequisites, all produced by earlier steps:
#   results/ariel_best.json + models/ariel_<best>.joblib      (pipeline.py --config ariel)
#   data/test*_native_{cloud,haze,tlse,exotransmit,exomol,quenched}*.npy   (shift_*.py, generate_grid.py --mode quenched)
#   data/train_native_aug_{spots,haze}.npy, data/train_native_quenched.npy   (augment.py renders, generate_grid.py)
# Each step logs to results/<step>.log; a failing step stops the chain so nothing downstream runs on missing inputs.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/tfenv/bin/python
step() { local name=$1; shift; echo "=== $name  $(date '+%H:%M:%S')"; "$PY" "$@" > "results/$name.log" 2>&1 || { echo "FAILED: $name (see results/$name.log)"; exit 1; }; }

step evaluate_shifts    evaluate_shifts.py --config ariel          # the fidelity budget, every axis found on disk
step analyze_labels     analyze_labels.py --config ariel           # cut sensitivity, |log C/O| margin, amplitude
step analyze_calibration analyze_calibration.py --config ariel     # reliability, thresholds under the 1e4 Pa deck
step realism            realism.py --config ariel                  # per-channel offsets, floors, realistic subset
step physics_baseline   physics_baseline.py --config ariel         # carbon-minus-water band index
step prevalence         prevalence.py --config ariel               # operating points at realistic base rates
step augment            augment.py --render --fit --jobs 4 --config ariel   # repair: spots, haze, correlated noise (renders the two training augmentations first)
step augment_white      augment_white.py                           # repair: white noise
step augment_ramp       augment_ramp.py                            # repair: gain ramp (the confound-breaker)
step augment_quenched   augment_quenched.py --config ariel         # repair: Axis 8, against the pre-registration
step frequency          frequency.py                               # the Fourier alternative, measured
step invertibility      invertibility.py --config ariel            # the invertibility alternative, measured
step sensitivity        sensitivity.py --config ariel              # train-draw, spot contrast, correlation length
echo "=== chain complete  $(date '+%H:%M:%S')"
echo "next: run_multimodel.sh (five further pipelines) then transfer.py, then tables.py"
