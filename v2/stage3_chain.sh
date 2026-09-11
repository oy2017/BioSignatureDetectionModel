#!/bin/bash
# Runs after the Ariel tuning finishes: ladder rungs, label analysis, then (after renders) shifts, calibration, tables, figures.
cd /mnt/c/Users/owenh/BioSignatureDetectionModel/v2
PY=~/tfenv/bin/python
F="grep -v -i warning\|Loading\|tensorflow\|cuda\|oneDNN\|rebuild"
while kill -0 92699 2>/dev/null; do sleep 60; done
echo "== ariel done $(date)"
$PY pipeline.py --config r100 --reuse-params --mlp-restarts 3 > results/r100_pipeline.log 2>&1
echo "== r100 done $(date)"
$PY pipeline.py --config r200 --reuse-params --mlp-restarts 3 > results/r200_pipeline.log 2>&1
echo "== r200 done $(date)"
$PY analyze_labels.py --config ariel > results/ariel_labels.log 2>&1
echo "== labels done $(date)"
while pgrep -f run_shifts.sh >/dev/null; do sleep 60; done
echo "== renders done $(date)"
$PY evaluate_shifts.py --config ariel > results/ariel_shifts.log 2>&1
echo "== shifts done $(date)"
$PY analyze_calibration.py --config ariel > results/ariel_calibration.log 2>&1
$PY tables.py > /dev/null 2>&1
for f in plots/fig2_ladder_calibration.py plots/fig3_margin_amplitude.py plots/fig4_fidelity_sweeps.py plots/fig5_threshold_transfer.py; do $PY $f 2>&1 | tail -1; done
echo "== all done $(date)"
