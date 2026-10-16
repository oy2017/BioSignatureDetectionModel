#!/usr/bin/env bash
# Fetch every external file the v3 pipeline needs but the repository does not carry.
#
#   v3/data/opacities/opac*.dat   nine Exo-Transmit opacity tables (Kempton et al. 2017), ~220 MB
#   ~/fastchem_input/             FastChem's solar abundances and equilibrium constants (Stock et al. 2018)
#   v2/phoenix/phoenixm00_*.fits  the STScI PHOENIX solar-metallicity atlas, 142 files, ~1.1 GB
#                                 (v3/phoenix is a symlink to it); used by the star-spot departures
#   v3/data/mcs/Ariel_MCS_Known_2026-05-11.csv   the Ariel Mission Candidate Sample (Edwards & Tinetti 2022)
#
# Files already present with the right checksum are skipped. Usage:
#   bash fetch_external_data.sh            # everything
#   bash fetch_external_data.sh --no-phoenix   # skip the 1.1 GB atlas (only the spot departures need it)
set -euo pipefail
cd "$(dirname "$0")"
PHOENIX=1; [ "${1:-}" = "--no-phoenix" ] && PHOENIX=0

md5of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
fetch() {   # fetch <url> <dest> [md5]
  local url=$1 dest=$2 sum=${3:-}
  if [ -f "$dest" ] && { [ -z "$sum" ] || [ "$(md5of "$dest")" = "$sum" ]; }; then echo "  ok      $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
  if [ -n "$sum" ] && [ "$(md5of "$dest")" != "$sum" ]; then echo "CHECKSUM MISMATCH: $dest" >&2; exit 1; fi
  echo "  fetched $dest"
}

echo "== Exo-Transmit opacity tables -> v3/data/opacities"
EXO=https://raw.githubusercontent.com/elizakempton/Exo_Transmit/master/Opac
while read -r sum name; do fetch "$EXO/$name" "v3/data/opacities/$name" "$sum"; done <<'EOF'
b4b98ef07a7aaf2372f3ae45ad04250d opacH2O.dat
49f7b33f2e21894ba285b04ab86b0385 opacCH4.dat
9b48ce6c202edaaf115740b2bca4ae8c opacCO2.dat
b2c19ecd11d70364795df5ca7836d245 opacO2.dat
f677c8068ee19f379a8f36acd0daaa98 opacO3.dat
4b3949ff95360639ba1ef675ec7397a5 opacCO.dat
c2cb08df194f8738c7ae282419440846 opacHCN.dat
bb0d1d938fe85738c28f3a3b3cb1ac6a opacC2H2.dat
9674d3e042684692b99bc319ab087c2c opacNH3.dat
EOF

echo "== FastChem inputs -> ~/fastchem_input"
FC=https://raw.githubusercontent.com/exoclime/FastChem/master/input
fetch "$FC/element_abundances/asplund_2009.dat" "$HOME/fastchem_input/asplund_2009.dat" adcf179f360d64e51a4d768b21e46a70
fetch "$FC/logK/logK.dat" "$HOME/fastchem_input/logK.dat" b9bc4797fa986dbf90dfbdfb0edd82fc

echo "== Ariel Mission Candidate Sample -> v3/data/mcs"
fetch "https://raw.githubusercontent.com/arielmission-space/Mission_Candidate_Sample/main/target_lists/Ariel_MCS_Known_2026-05-11.csv" \
      "v3/data/mcs/Ariel_MCS_Known_2026-05-11.csv" 61a4e150c7ec87b5677f7fc32bf5e649

if [ "$PHOENIX" = 1 ]; then
  echo "== STScI PHOENIX atlas (solar metallicity) -> v2/phoenix"
  PHX=https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/phoenixm00
  mkdir -p v2/phoenix
  curl -fsSL "$PHX/" | grep -o 'phoenixm00_[0-9]*\.fits' | sort -u > v2/phoenix/.index
  n=$(wc -l < v2/phoenix/.index | tr -d ' '); echo "  $n files in the atlas index"
  # no published checksums; a file is accepted if it starts like a FITS file
  fetch_phx() { local f="v2/phoenix/$1"; if [ -s "$f" ] && [ "$(head -c 6 "$f")" = "SIMPLE" ]; then return 0; fi
                curl -fsSL --retry 3 -o "$f.part" "$PHX/$1" && mv "$f.part" "$f"; }
  export -f fetch_phx; export PHX
  xargs -P 8 -n 1 bash -c 'fetch_phx "$0"' < v2/phoenix/.index
  bad=$(while read -r f; do [ "$(head -c 6 "v2/phoenix/$f")" = "SIMPLE" ] || echo "$f"; done < v2/phoenix/.index)
  [ -z "$bad" ] && echo "  ok      $n PHOENIX files" || { echo "corrupt downloads: $bad" >&2; exit 1; }
else
  echo "== PHOENIX atlas skipped (--no-phoenix)"
fi
echo "done"
