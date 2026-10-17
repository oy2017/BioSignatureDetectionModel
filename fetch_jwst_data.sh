#!/usr/bin/env bash
# Fetch the published JWST transmission-spectrum products used by v3/wasp39_assemble.py and
# v3/jwst_panel.py into v3/data/jwst/ (git-ignored), from their original archives, and verify
# every spectrum file against v3/jwst_manifest.md5. Provenance per file: v3/jwst_validation_brief.md, part 3.
#
# Zenodo drops large transfers every few MB, so archives are fetched with a resume loop that stops
# when the zip verifies. Archives are kept under v3/data/jwst/archives/ so a rerun costs nothing.
#
#   bash fetch_jwst_data.sh            # everything (about 3.5 GB of archives; hours on a throttled link)
#   bash fetch_jwst_data.sh --verify   # only check the manifest
set -uo pipefail
cd "$(dirname "$0")"
ROOT=v3/data/jwst; ARC=$ROOT/archives; mkdir -p "$ARC" "$ROOT/wasp39" "$ROOT/panel"
md5of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }

zenodo() {   # zenodo <record> <file> : resume loop until the zip verifies
  local rec=$1 f=$2 dest="$ARC/${rec}_$2"
  if unzip -tq "$dest" >/dev/null 2>&1; then echo "  ok      $dest"; return 0; fi
  for attempt in $(seq 1 600); do
    curl -fsSL --retry 0 -C - -o "$dest" "https://zenodo.org/records/$rec/files/$f?download=1" 2>/dev/null
    unzip -tq "$dest" >/dev/null 2>&1 && { echo "  fetched $dest ($attempt attempts)"; return 0; }
    sleep 3
  done
  echo "GAVE UP on $dest" >&2; return 1
}
extract() {  # extract <zip> <destdir> <member patterns...>
  local z=$1 d=$2; shift 2; mkdir -p "$d"; unzip -oq "$z" "$@" -d "$d" 2>/dev/null || true
}
get() {      # get <url> <dest>
  [ -s "$2" ] && { echo "  ok      $2"; return 0; }
  mkdir -p "$(dirname "$2")"; curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" && echo "  fetched $2"
}

if [ "${1:-}" != "--verify" ]; then
  echo "== WASP-39 b: JWST ERS products"
  zenodo 7709712 zenodo_upload.zip       && extract "$ARC/7709712_zenodo_upload.zip" "$ROOT/wasp39/7709712/extracted" "zenodo_upload/4_TRANSMISSION_SPECTRA/*"
  zenodo 7101283 NIRCam_Zenodo_files.zip && extract "$ARC/7101283_NIRCam_Zenodo_files.zip" "$ROOT/wasp39/7101283/extracted" "2_TRANSMISSION_SPECTRA/*"
  zenodo 7185300 G395H_paper_data.zip    && extract "$ARC/7185300_G395H_paper_data.zip" "$ROOT/wasp39/7185300/extracted" "G395H_paper_data/3_TRANSMISSION_SPECTRA/*"
  zenodo 7388032 W39_PRISM_data.zip      && extract "$ARC/7388032_W39_PRISM_data.zip" "$ROOT/wasp39/7388032/extracted" "*"
  zenodo 10055845 miri_lrs_zenodo.zip    && extract "$ARC/10055845_miri_lrs_zenodo.zip" "$ROOT/wasp39/10055845/extracted" "*"
  zenodo 10161743 ERS_DataSynthesis_Zenodo.zip && extract "$ARC/10161743_ERS_DataSynthesis_Zenodo.zip" "$ROOT/wasp39/10161743/extracted" "*"

  echo "== panel"
  P=$ROOT/panel
  zenodo 15228454 TOI-270d_transit_spectra.zip && extract "$ARC/15228454_TOI-270d_transit_spectra.zip" "$P/TOI-270d/Felix2025_zenodo15228454" "TOI-270d_transit_spectra/native_*.dat"
  zenodo 16277833 Spectra.zip                  && extract "$ARC/16277833_Spectra.zip" "$P/K2-18b/Jaziri2025_zenodo16277833" "Spectra/*.txt"
  zenodo 14735689 K2-18_reanalysis_data.zip    && extract "$ARC/14735689_K2-18_reanalysis_data.zip" "$P/K2-18b/schmidt" "*exoTEDRF*R100.dat"
  zenodo 15176438 TOI421b.zip                  && extract "$ARC/15176438_TOI421b.zip" "$P/TOI-421b/Davenport2025_zenodo15176438" "TOI421b/NIRISS/exoTEDRF/*" "TOI421b/NIRISS/Fu_spec.txt" "TOI421b/NIRSpec/Deming_spectrum.txt"
  zenodo 15122285 Zenodo_WASP-166b.zip         && extract "$ARC/15122285_Zenodo_WASP-166b.zip" "$P/WASP-166b/extracted" "*Table_Save*" "*README*"
  zenodo 10780449 FinalResults.zip             && extract "$ARC/10780449_FinalResults.zip" "$P/WASP-107b" "FinalResults/FinalSpectra/*"
  zenodo 11459715 HD189733b_JWST_data_model.zip && extract "$ARC/11459715_HD189733b_JWST_data_model.zip" "$P/HD189733b" "zenodo/Data/transit spectrum/*"
  get "https://zenodo.org/records/10891400/files/Sing_2024_Fig2_WASP107b_transit_spec_data.csv?download=1" "$P/WASP-107b/Sing_2024_Fig2_WASP107b_transit_spec_data.csv"
  get "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/940/L35/table1.dat" "$P/HAT-P-18b/Fu2022_vizier_J_ApJ_940_L35/table1.dat"
  get "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/940/L35/ReadMe" "$P/HAT-P-18b/Fu2022_vizier_J_ApJ_940_L35/ReadMe"
  get "https://content.cld.iop.org/journals/1538-3881/170/5/292/revision1/ajae0929f2.tar.gz" "$ARC/Gressier2025_ajae0929f2.tar.gz" \
    && mkdir -p "$P/HAT-P-26b/Gressier2025_IOP_figdata/f2" && tar -xzf "$ARC/Gressier2025_ajae0929f2.tar.gz" -C "$P/HAT-P-26b/Gressier2025_IOP_figdata/f2" 2>/dev/null
  for t in 3 4; do get "https://iopscience.iop.org/2041-8205/970/1/L10/suppdata/apjlad55e9t${t}_mrt.txt?doi=10.3847/2041-8213/ad55e9" "$P/GJ3470b/Beatty2024_apjlad55e9t${t}_mrt.txt"; done
  echo "(the original K2-18 b, TOI-270 d and TOI-1231 b spectra are on OSF: osf.io/36djh, osf.io/8fu36, osf.io/mryd6; add them here when OSF is reachable)"
fi

echo "== verifying against v3/jwst_manifest.md5"
bad=0; n=0
while read -r sum path; do
  [ -z "$sum" ] && continue; n=$((n+1))
  if [ ! -f "$path" ]; then echo "  MISSING  $path"; bad=$((bad+1))
  elif [ "$(md5of "$path")" != "$sum" ]; then echo "  CHANGED  $path"; bad=$((bad+1)); fi
done < v3/jwst_manifest.md5
echo "  $((n-bad)) of $n files verified"; [ "$bad" = 0 ]
