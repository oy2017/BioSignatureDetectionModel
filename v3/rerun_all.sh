#!/usr/bin/env bash
# Full regeneration of the v3 study with NH3 absorbing (opacNH3.dat added to MultiREx, 2026-09-12).
# Everything downstream of the forward model is rebuilt from scratch; the NH3-less run is kept in
# data_noNH3/, models_noNH3/ and results_noNH3/ for the record. Stages:
#   0  archive, fresh data/ and models/
#   1  primary grid + binning
#   2  in parallel: (A) tuning  (B) MultiREx axis renders, sequential  (C) Exo-Transmit renders
#      then (D) ExoMol renders alone among MultiREx renders
#   3  every analysis script, in dependency order; a failure is logged and the rest continues
set -uo pipefail
cd "$(dirname "$0")"
PY=~/tfenv/bin/python; LOG=results/rerun_all.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
run() { local name=$1; shift; say "start $name"; if "$@" > "results/$name.log" 2>&1; then say "done  $name"; else say "FAILED $name (results/$name.log)"; fi; }
export OMP_NUM_THREADS=1

# ---- stage 0
say "=== stage 0: archive"
[ -d data_noNH3 ] || mv data data_noNH3
[ -d models_noNH3 ] || mv models models_noNH3
mkdir -p data/mcs models
cp data_noNH3/mcs/Ariel_MCS_Known_2026-05-11.csv data/mcs/
rm -f results/ariel_best.json results/*.pid

# ---- stage 1
say "=== stage 1: grid"
run generate_grid $PY generate_grid.py --jobs 12
run bin_spectra   $PY bin_spectra.py

# ---- stage 2
say "=== stage 2: tuning + renders in parallel"
( export OMP_NUM_THREADS=8; run ariel_pipeline $PY pipeline.py --config ariel ) &
PID_A=$!
( run shift_exotransmit $PY shift_exotransmit.py --splits test1 test2 test3 test4 test5 train --jobs 4 --no-wait ) &
PID_C=$!
(
  run shift_aerosol      $PY shift_aerosol.py --jobs 6
  run shift_tlse         $PY shift_tlse.py
  run generate_quenched  $PY generate_grid.py --mode quenched --jobs 6 --splits train test1 test2 test3 test4 test5
  run shift_absorbers    $PY shift_absorbers.py --jobs 6 --mode equilibrium
  run shift_absorbers_q  $PY shift_absorbers.py --jobs 6 --mode quenched
  run shift_absorbers_tr $PY shift_absorbers.py --jobs 6 --mode equilibrium --splits train
  run randomize_train    $PY randomize_train.py --jobs 6
  for k in 1e7 1e8 1e10 1e11; do run kzz_$k $PY generate_grid.py --mode quenched --kzz $k --jobs 6 --splits test1 test2 test3 test4 test5; done
  run mcs_testset        $PY mcs_testset.py --jobs 6
  run alfnoor_render     $PY alfnoor_screen.py --render --jobs 6
) &
PID_B=$!
wait $PID_B
say "=== stage 2D: ExoMol renders (alone among MultiREx renders)"
run shift_opacity $PY shift_opacity.py --variant exomol --splits test1 test2 test3 test4 test5 train --jobs 4
wait $PID_A; wait $PID_C

# ---- stage 3
say "=== stage 3: analysis"
export OMP_NUM_THREADS=8
run compound          $PY compound.py --config ariel
run run_chain         bash run_chain.sh
run run_multimodel    bash run_multimodel.sh
for m in pca_rf raw_xgb raw_rf; do run shift_$m $PY evaluate_shifts.py --config ariel --model $m --no-extrapolation; done
run transfer          $PY transfer.py --config ariel
run tables            $PY tables.py
run oracle            $PY oracle.py --jobs 6
run trust_detect      $PY trust_detect.py --config ariel
run trust_randomized  $PY trust_randomized.py --config ariel
run trust_envelope    $PY trust_envelope.py --config ariel
run host_dependence   $PY host_dependence.py --config ariel
run tier_screen       $PY tier_screen.py
run mcs_eval          $PY mcs_eval.py
run alfnoor_fit       $PY alfnoor_screen.py --fit
run alfnoor_fit_half  $PY alfnoor_screen.py --fit --noise-factor 0.5
run kzz_eval          $PY kzz_eval.py --config ariel
say "=== all done"
