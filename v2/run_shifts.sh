#!/bin/bash
# Sequential re-render jobs for the five test splits (Stage 5 data).
cd /mnt/c/Users/owenh/BioSignatureDetectionModel/v2
PY=~/tfenv/bin/python
echo "== tlse $(date)";        $PY shift_tlse.py                              2>&1 | grep -v -i "warning\|Loading"
echo "== exotransmit $(date)"; $PY shift_exotransmit.py --jobs 6 --no-wait    2>&1 | grep -v -i "warning\|Loading"
echo "== opacity $(date)";     $PY shift_opacity.py --variant both --jobs 6   2>&1 | grep -v -i "warning\|Loading"
echo "== aerosol $(date)";     $PY shift_aerosol.py --kind both --jobs 6      2>&1 | grep -v -i "warning\|Loading"
echo "== done $(date)"
