#!/bin/bash
# Re-render the five test splits under every physics axis. Run from v2/.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
echo "== stellar contamination $(date)"; $PY shift_tlse.py
echo "== independent RT code $(date)";   $PY shift_exotransmit.py --jobs 6 --no-wait
echo "== opacity swap $(date)";          $PY shift_opacity.py --variant both --jobs 6
echo "== clouds and hazes $(date)";      $PY shift_aerosol.py --kind both --jobs 6
echo "== done $(date)"
