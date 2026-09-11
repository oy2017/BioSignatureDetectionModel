#!/bin/bash
cd /mnt/c/Users/owenh/BioSignatureDetectionModel/v2; PY=~/tfenv/bin/python; export OMP_NUM_THREADS=8
echo "== start $(date)"
$PY pipeline.py --config ariel_abs50 --reuse-params --mlp-restarts 3 > results/ariel_abs50_pipeline.log 2>&1; echo "== abs50 pipeline done $(date)"
$PY analyze_labels.py --config ariel_abs50 > results/ariel_abs50_labels.log 2>&1; echo "== abs50 labels done $(date)"
$PY evaluate_shifts.py --config ariel --no-extrapolation > results/ariel_shifts2.log 2>&1; echo "== ariel shifts (with absolute family) done $(date)"
$PY analyze_calibration.py --config ariel_abs50 > results/ariel_abs50_calibration.log 2>&1; echo "== abs50 calibration done $(date)"
echo "== all done $(date)"
