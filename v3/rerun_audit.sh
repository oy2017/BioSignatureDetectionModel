#!/usr/bin/env bash
# Full regeneration of the v3 study after the 2026-09-12 audit. Fixes in force (see MULTIREX_FORK.md
# and git log): Exo-Transmit opacity pressures in Pa (fork change 6); grey deck kept with a Mie haze
# (fork change 7); ExoMol swap keeps NH3; width-aware noise shapes (noise.nsr_binned); training noise
# scaled to the clean spectrum; corrected quench profile; correlated noise tagged held-out; mission
# noise from the payload model (mcs_eval); consortium screen rebuilt as alfnoor_faithful.py.
# The pre-audit state is kept in data_prePfix/, models_prePfix/ and results_prePfix/.
#   0  archive; fresh data/ and models/; carry over inputs that no fix touches
#   1  primary grid + binning
#   2  in parallel: (A) tuning  (B) MultiREx renders, sequential; then (D) ExoMol renders
#   3  every analysis script in dependency order; a failure is logged and the rest continues
#   4  figures
set -uo pipefail
cd "$(dirname "$0")"
PY=~/tfenv/bin/python; LOG=results/rerun_audit.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
run() { local name=$1; shift; say "start $name"; if "$@" > "results/$name.log" 2>&1; then say "done  $name"; else say "FAILED $name (results/$name.log)"; fi; }
export OMP_NUM_THREADS=1

# ---- stage 0
say "=== stage 0: archive"
$PY -c "import forward_model_guard" || { say "forward-model guard failed; aborting"; exit 1; }
[ -d results_prePfix ] || cp -r results results_prePfix
[ -d data_prePfix ] || mv data data_prePfix
[ -d models_prePfix ] || mv models models_prePfix
mkdir -p data/mcs data/alfnoor models
cp data_prePfix/mcs/Ariel_MCS_Known_2026-05-11.csv data/mcs/
cp data_prePfix/alfnoor/pop1_params.parquet data_prePfix/alfnoor/pop3_params.parquet data/alfnoor/   # the consortium population draws
for f in data_prePfix/*_native_exotransmit.npy; do cp "$f" data/; done                                # Exo-Transmit's own C code: untouched by every fix
rm -f results/ariel_best.json results/*.pid

# ---- stage 1
say "=== stage 1: grid"
run generate_grid $PY generate_grid.py --jobs 12
run bin_spectra   $PY bin_spectra.py

# ---- stage 2
say "=== stage 2: tuning + renders in parallel"
( export OMP_NUM_THREADS=8; run ariel_pipeline $PY pipeline.py --config ariel ) &
PID_A=$!
(
  run shift_aerosol      $PY shift_aerosol.py --jobs 8
  run shift_tlse         $PY shift_tlse.py
  run generate_quenched  $PY generate_grid.py --mode quenched --jobs 8 --splits train test1 test2 test3 test4 test5
  run shift_absorbers    $PY shift_absorbers.py --jobs 8 --mode equilibrium
  run shift_absorbers_q  $PY shift_absorbers.py --jobs 8 --mode quenched
  run shift_absorbers_tr $PY shift_absorbers.py --jobs 8 --mode equilibrium --splits train
  run randomize_train    $PY randomize_train.py --jobs 8
  for k in 1e7 1e8 1e10 1e11; do run kzz_$k $PY generate_grid.py --mode quenched --kzz $k --jobs 8 --splits test1 test2 test3 test4 test5; done
  run mcs_testset        $PY mcs_testset.py --jobs 8
  run alfnoor_render     $PY alfnoor_faithful.py --render --jobs 8
) &
PID_B=$!
wait $PID_B
say "=== stage 2D: ExoMol renders"
run shift_opacity $PY shift_opacity.py --variant exomol --splits test1 test2 test3 test4 test5 train --jobs 6
wait $PID_A

# ---- stage 3
say "=== stage 3: analysis"
export OMP_NUM_THREADS=8
run compound          $PY compound.py --config ariel
run run_chain         bash run_chain.sh
run run_multimodel    bash run_multimodel.sh
for m in pca_rf raw_xgb raw_rf; do run shift_$m $PY evaluate_shifts.py --config ariel --model $m --no-extrapolation; done
run transfer          $PY transfer.py --config ariel
run tables            $PY tables.py
run oracle            $PY oracle.py --jobs 8
run trust_detect      $PY trust_detect.py --config ariel
run trust_randomized  $PY trust_randomized.py --config ariel
run trust_envelope    $PY trust_envelope.py --config ariel
run host_dependence   $PY host_dependence.py --config ariel
run conformal         $PY conformal_calibration.py --config ariel
run mechanism_bands   $PY mechanism_bands.py --config ariel
run kzz_eval          $PY kzz_eval.py --config ariel
run tier_screen       $PY tier_screen.py
run oracle_tier1      $PY oracle.py --jobs 8 --config tier1
run trust_detect_tier1     $PY trust_detect.py --config tier1
run trust_randomized_tier1 $PY trust_randomized.py --config tier1
run trust_envelope_tier1   $PY trust_envelope.py --config tier1
run mcs_eval          $PY mcs_eval.py
run alfnoor_fit       $PY alfnoor_faithful.py --fit --natives faithful --layout tier3_r20 --noise radiometric

# ---- stage 4
say "=== stage 4: figures"
for f in fig1_map fig2_absorbers fig3_detect fig4_tiers; do run plot_$f $PY plots/$f.py; done
say "=== all done"
