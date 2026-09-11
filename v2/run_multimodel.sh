#!/bin/bash
cd /mnt/c/Users/owenh/BioSignatureDetectionModel/v2; PY=~/tfenv/bin/python; export OMP_NUM_THREADS=8
for m in norm_mlp pca_xgb norm_rf; do
  echo "== $m $(date)"
  $PY evaluate_shifts.py --config ariel --model $m --no-extrapolation > results/shift_$m.log 2>&1
done
echo "== multimodel done $(date)"
