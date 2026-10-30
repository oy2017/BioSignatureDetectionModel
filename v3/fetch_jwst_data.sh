#!/usr/bin/env bash
# Downloads the published JWST transmission spectra used by jwst_panel.py (data/jwst/panel) and wasp39_assemble.py
# (data/jwst/wasp39) into the directory layout those scripts read. Everything comes from public Zenodo, OSF and VizieR
# records; the two IOP figure-data / machine-readable-table sources that have no stable download URL are listed at the
# end with the manual step. Re-running is safe: existing files are kept. Downloads are untrusted data: nothing here is
# executed, only unpacked.
#   bash fetch_jwst_data.sh            # everything (about 6 GB, dominated by the K2-18 b opacity and retrieval archives, which are skipped)
#   bash fetch_jwst_data.sh panel      # the panel only;  ... wasp39  # the WASP-39 b products only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; D="$HERE/data/jwst"; WHAT="${1:-all}"
need() { command -v "$1" >/dev/null || { echo "need $1" >&2; exit 1; }; }; need curl; need unzip; need python3

zenodo_files() {   # record -> "key url" per line (follows the record redirect)
  curl -sL "https://zenodo.org/api/records/$1" | python3 -I -c 'import sys,json
for f in json.load(sys.stdin).get("files",[]): print(f["key"], f["links"]["self"])'
}
zget() {   # zget <record> <dest dir> [only-this-file] : download the record's files (or one) into dest and unzip any zip there
  local rec=$1 dest=$2 only=${3:-}; mkdir -p "$dest"
  zenodo_files "$rec" | while read -r key url; do
    [ -n "$only" ] && [ "$key" != "$only" ] && continue
    case "$key" in Opacities.zip|Combine_retrieval.zip|MIRI_retrieval.zip|corner_plots.zip) continue;; esac   # multi-GB, not spectra
    if [ ! -e "$dest/$key" ]; then echo "zenodo $rec: $key"; curl -sL -o "$dest/$key" "$url"; fi
    case "$key" in *.zip) unzip -qo "$dest/$key" -d "$dest";; esac
  done
}
osf_zip() {   # osf_zip <node id> <dest dir> : the whole OSF storage of a project as a zip, unpacked into dest
  local id=$1 dest=$2; mkdir -p "$dest"
  if [ ! -e "$dest/.osf_$id.zip" ]; then echo "osf $id"; curl -sL -o "$dest/.osf_$id.zip" "https://files.osf.io/v1/resources/$id/providers/osfstorage/?zip="; unzip -qo "$dest/.osf_$id.zip" -d "$dest"; fi
}

if [ "$WHAT" = all ] || [ "$WHAT" = panel ]; then
  P="$D/panel"
  zget 15228454 "$P/TOI-270d/Felix2025_zenodo15228454" TOI-270d_transit_spectra.zip          # Felix et al. 2025
  osf_zip 8fu36 "$P/TOI-270d/Holmberg2024_osf_8fu36"                                          # Holmberg & Madhusudhan 2024
  osf_zip 36djh "$P/K2-18b/Madhusudhan2023_osf_36djh"                                         # Madhusudhan et al. 2023
  osf_zip hpu8g "$P/K2-18b/Hu2025_osf_hpu8g"                                                  # Hu et al. 2025 (spectra/ inside)
  zget 16277833 "$P/K2-18b/Jaziri2025_zenodo16277833" Spectra.zip                             # Jaziri et al. 2025 (MIRI copy)
  zget 14735689 "$P/K2-18b/schmidt/K2-18_zenodo"                                              # Schmidt & MacDonald 2025
  osf_zip mryd6 "$P/TOI-1231b/Sarkar2026_osf_mryd6"                                           # Sarkar et al. 2026
  zget 15176438 "$P/TOI-421b/Davenport2025_zenodo15176438"                                    # Davenport et al. 2025
  mkdir -p "$P/HAT-P-18b/Fu2022_vizier_J_ApJ_940_L35"                                         # Fu et al. 2022
  for f in ReadMe table1.dat; do [ -e "$P/HAT-P-18b/Fu2022_vizier_J_ApJ_940_L35/$f" ] || curl -sL -o "$P/HAT-P-18b/Fu2022_vizier_J_ApJ_940_L35/$f" "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/940/L35/$f"; done
  zget 10780449 "$P/WASP-107b"                                                                # Welbanks et al. 2024 (FinalResults/ inside)
  zget 10891400 "$P/WASP-107b" Sing_2024_Fig2_WASP107b_transit_spec_data.csv                  # Sing et al. 2024
  zget 11459715 "$P/HD189733b/zenodo"                                                         # Fu et al. 2024
  zget 15122285 "$P/WASP-166b/extracted"                                                      # Mayo et al. 2025
  zget 10557924 "$P/HD209458b/Xue2024_zenodo10557924" spectra_final.csv                       # Xue et al. 2024
  zget 8360121  "$P/WASP-17b/Grant2023_zenodo8360121"                                         # Grant et al. 2023 (ExoTiC-MIRI/ inside)
  zget 14193061 "$P/WASP-17b/Louie2025_zenodo14193061"                                        # Louie et al. 2025 (Data_reduction_Ahsoka/ inside)
  zget 17065171 "$P/WASP-96b/Radica2026_zenodo17065171"                                       # Radica et al. 2026 (WASP-96b_Zenodo/ inside)
  zget 18734230 "$P/WASP52b/Pan2026_zenodo18734230" data_behind_figures.csv                   # Pan et al. 2026
  # third batch (held out, 2026-10-29)
  for f in WASP-94_NIRISS_SOSS_order1_Rreference.csv WASP-94_NIRSPEC_G395H_nrs1_Rreference.csv WASP-94_NIRSPEC_G395H_nrs2_Rreference.csv; do zget 22803020 "$P/WASP-94Ab/Zenodo22803020" "$f"; done   # limb reanalysis of Ahrer et al. 2025
  zget 13366851 "$P/WASP127b/Fu2025_zenodo13366851"                                          # Fu et al. 2025 compilation (Spectra/ inside; HIP 67522b from Thao et al. 2024)
  zget 19356399 "$P/HAT-P-12b/Heinke2026_zenodo19356398"                                     # Heinke et al. 2026 (712 MB zip)
  mkdir -p "$P/TOI-732c/Rigby2025_osf_wsjmu"                                                  # Rigby et al. 2025 (TOI-732 c), individual OSF files
  for kv in README.txt:s9rwg TOI-732c_niriss.txt:8t23g TOI-732c_nirspec_g395h_nrs1.txt:nk7t4 TOI-732c_nirspec_g395h_nrs2.txt:a2t9w TOI-732c_miri_JexoPipe.txt:t8b3a TOI-732c_miri_JexoRes.txt:9trv6; do
    f=${kv%%:*}; id=${kv##*:}; [ -e "$P/TOI-732c/Rigby2025_osf_wsjmu/$f" ] || curl -sL -o "$P/TOI-732c/Rigby2025_osf_wsjmu/$f" "https://osf.io/download/$id/"; done
  mkdir -p "$P/GJ9827d/PiauletGhorayeb2024_apjl_ad6f00_mrt" "$P/GJ1214b/Schlawin2024_apjl_ad7fef_fig2data" "$P/WASP-80b/Bell2023_nature_s41586-023-06687-0_sourcedata"
  [ -e "$P/GJ9827d/PiauletGhorayeb2024_apjl_ad6f00_mrt/apjlad6f00t4_mrt.txt" ] || curl -sL -o "$P/GJ9827d/PiauletGhorayeb2024_apjl_ad6f00_mrt/apjlad6f00t4_mrt.txt" "https://content.cld.iop.org/journals/2041-8205/974/1/L10/revision1/apjlad6f00t4_mrt.txt"   # Piaulet-Ghorayeb et al. 2024
  if [ ! -e "$P/GJ1214b/Schlawin2024_apjl_ad7fef_fig2data/fig2_spec_fig2_tshirt.mrt" ]; then                                                                                           # Schlawin et al. 2024, figure 2 data
    curl -sL -o "$P/GJ1214b/Schlawin2024_apjl_ad7fef_fig2data/apjlad7feff2.tar.gz" "https://content.cld.iop.org/journals/2041-8205/974/2/L33/revision1/apjlad7feff2.tar.gz"; tar -xzf "$P/GJ1214b/Schlawin2024_apjl_ad7fef_fig2data/apjlad7feff2.tar.gz" -C "$P/GJ1214b/Schlawin2024_apjl_ad7fef_fig2data"; fi
  [ -e "$P/WASP-80b/Bell2023_nature_s41586-023-06687-0_sourcedata/41586_2023_6687_MOESM5_ESM.xlsx" ] || curl -sL -o "$P/WASP-80b/Bell2023_nature_s41586-023-06687-0_sourcedata/41586_2023_6687_MOESM5_ESM.xlsx" "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06687-0/MediaObjects/41586_2023_6687_MOESM5_ESM.xlsx"   # Bell et al. 2023 source data
  cat <<'MANUAL'
Three sources have no stable direct URL and are fetched by hand from the journal pages:
  HAT-P-26b  Gressier et al. 2025, AJ: "figure data" tar.gz for figures 1-2 from the article page -> data/jwst/panel/HAT-P-26b/Gressier2025_IOP_figdata/{f1,f2}/
  GJ3470b    Beatty et al. 2024, ApJL 970 L10: machine-readable tables 3-4 -> data/jwst/panel/GJ3470b/Beatty2024_apjlad55e9t{3,4}_mrt.txt
  L98-59d    Gressier et al. 2024, ApJL 975 L10: machine-readable table 3 (the link is time-limited) -> data/jwst/panel/L98-59d/Gressier2024_apjl_ad73d1_mrt/apjlad73d1t3_mrt.dat
MANUAL
  # fourth batch (independent test of the frozen rule, 2026-10-30)
  zget 15097695 "$P/TrES-4b/Meech2025_zenodo15097695" transmission_spectra.tar.gz               # Meech et al. 2025 (principal Tiberius reduction is in transmission_spectra/tiberius/)
  [ -d "$P/TrES-4b/Meech2025_zenodo15097695/transmission_spectra" ] || tar -xzf "$P/TrES-4b/Meech2025_zenodo15097695/transmission_spectra.tar.gz" -C "$P/TrES-4b/Meech2025_zenodo15097695"
  for f in data.tar.gz README.txt; do zget 21500595 "$P/HATS-6b/GuzmanCaloca2026_zenodo21500595" "$f"; done                                            # Guzman Caloca et al. 2026 (Eureka1, data/BINNED_coadd.txt)
  [ -d "$P/HATS-6b/GuzmanCaloca2026_zenodo21500595/data" ] || tar -xzf "$P/HATS-6b/GuzmanCaloca2026_zenodo21500595/data.tar.gz" -C "$P/HATS-6b/GuzmanCaloca2026_zenodo21500595"
  zget 18864528 "$P/NGTS-2b/Fairman2026_zenodo18864528" ExoTiC-JEDI_data_products.zip           # Fairman et al. 2026 (NetCDF; R=400 and R=100 transit spectra)
  zget 18682265 "$P/HAT-P-30b/Claringbold2026_zenodo18682265" transmission_spectra.zip           # Claringbold et al. 2026 (transmission_spectra/Eureka_R400.dat)
  for f in transmission_spectrum_toi199_eureka_nativespectral.txt transmission_spectrum_toi199_tswift.txt; do zget 17653116 "$P/TOI-199b/BelloArufe2026_zenodo17653116" "$f"; done   # Bello-Arufe et al. 2026
  zget 16782958 "$P/V1298Taub/Barat2025_zenodo16782958" Readme.md.txt                           # Barat et al. 2025 (the spectrum's key has a space, so zget cannot take it)
  if [ ! -d "$P/V1298Taub/Barat2025_zenodo16782958/final spectrum" ]; then curl -sL -o "$P/V1298Taub/Barat2025_zenodo16782958/final_spectrum.zip" "https://zenodo.org/api/records/16782958/files/final%20spectrum.zip/content"; unzip -qo "$P/V1298Taub/Barat2025_zenodo16782958/final_spectrum.zip" -d "$P/V1298Taub/Barat2025_zenodo16782958"; fi
  if [ ! -e "$P/LP791-18c/Roy2026_zenodo17316562/LP79118c/LP79118c_fullRes_spectrum.spec" ]; then   # Roy et al. 2026: 867 MB zip that also holds pickles; only the three spectra are unpacked, the zip is then removed
    mkdir -p "$P/LP791-18c/Roy2026_zenodo17316562"; curl -sL -o "$P/LP791-18c/Roy2026_zenodo17316562/LP79118c.zip" "https://zenodo.org/api/records/17316562/files/LP79118c.zip/content"
    unzip -qo "$P/LP791-18c/Roy2026_zenodo17316562/LP79118c.zip" 'LP79118c/LP79118c_R25_clean_u1u2_EurekaV1_Feb2025_newLD.spec' 'LP79118c/LP79118c_fullRes_spectrum.spec' 'LP79118c/spec_ExoTEDRF_R50.spec' -d "$P/LP791-18c/Roy2026_zenodo17316562"; rm -f "$P/LP791-18c/Roy2026_zenodo17316562/LP79118c.zip"; fi
  mkdir -p "$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data" "$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data" "$P/TOI-5205b/Canas2026_aj_ae4976_fig2data" "$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data" "$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data" "$P/Kepler-51d/LibbyRoberts2026_aj_ae33c0_mrt"
  if [ ! -e "$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data/dbf2.txt" ]; then curl -sL -o "$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data/apjlaea4aef2.tar.gz" "https://content.cld.iop.org/journals/2041-8205/1009/2/L28/revision1/apjlaea4aef2.tar.gz"; tar -xzf "$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data/apjlaea4aef2.tar.gz" -C "$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data"; fi            # Zhang et al. 2026, figure 2 data
  if [ ! -e "$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data/dbf2.txt" ]; then curl -sL -o "$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data/ajaea90af2.tar.gz" "https://content.cld.iop.org/journals/1538-3881/172/5/275/revision1/ajaea90af2.tar.gz"; tar -xzf "$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data/ajaea90af2.tar.gz" -C "$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data"; fi                    # Ashtari et al. 2026, figure 2 data
  if [ ! -e "$P/TOI-5205b/Canas2026_aj_ae4976_fig2data/toi5205_coadd_exotic_pixel.txt" ]; then curl -sL -o "$P/TOI-5205b/Canas2026_aj_ae4976_fig2data/ajae4976f2.tar.gz" "https://content.cld.iop.org/journals/1538-3881/171/4/260/revision1/ajae4976f2.tar.gz"; tar -xzf "$P/TOI-5205b/Canas2026_aj_ae4976_fig2data/ajae4976f2.tar.gz" -C "$P/TOI-5205b/Canas2026_aj_ae4976_fig2data"; fi   # Canas et al. 2026, figure 2 data
  if [ ! -e "$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data/ReadMe_dbf2.txt" ]; then curl -sL -o "$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data/apjlae5f8bf2.tar.gz" "https://content.cld.iop.org/journals/2041-8205/1002/2/L32/revision1/apjlae5f8bf2.tar.gz"; tar -xzf "$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data/apjlae5f8bf2.tar.gz" -C "$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data"; fi       # Barat et al. 2026, figure 2 data
  if [ ! -e "$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data/dbf6a.txt" ]; then curl -sL -o "$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data/ajad9c6ef6.tar.gz" "https://content.cld.iop.org/journals/1538-3881/169/6/341/revision1/ajad9c6ef6.tar.gz"; tar -xzf "$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data/ajad9c6ef6.tar.gz" -C "$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data"; fi                  # Gapp et al. 2025, figure 6 data
  [ -e "$P/Kepler-51d/LibbyRoberts2026_aj_ae33c0_mrt/ajae33c0t2_mrt.txt" ] || curl -sL -o "$P/Kepler-51d/LibbyRoberts2026_aj_ae33c0_mrt/ajae33c0t2_mrt.txt" "https://content.cld.iop.org/journals/1538-3881/171/4/221/revision1/ajae33c0t2_mrt.txt"   # Libby-Roberts et al. 2026, table 2
fi
if [ "$WHAT" = all ] || [ "$WHAT" = wasp39 ]; then
  W="$D/wasp39"
  for rec in 7709712 7101283 7185300 7388032 10161743 10055845; do zget "$rec" "$W/$rec/extracted"; done   # Feinstein, Ahrer, Alderson, Rustamkulov 2023; Carter & May 2024; Powell 2024
fi
echo done
