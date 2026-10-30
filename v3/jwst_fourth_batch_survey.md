# Batch 4 survey: planets for the independent JWST test of the frozen rule

Compiled 2026-10-30. Labels come from the papers only. No classifier was run, no script in v3/ that computes verdicts was run, and nothing under v3/results/jwst_panel*, v3/REVISION_FINDINGS.md or the manuscript results was opened. One `git status` check, run to confirm the fetch-script edit, printed some file names under v3/results/jwst_panel/; names only, no contents were read.

**Result: 13 planets qualify, giving 24 scored labels (8 present, 16 absent) out of 52 slots. No tentative present labels. Nobody in this batch has a scored CO2 present label, so CO2 recall cannot be tested on it (see (e)).**

| | present | absent | None |
|---|---|---|---|
| CH4 | 3 | 7 | 3 |
| H2O | 4 | 2 | 7 |
| CO2 | 0 | 4 | 9 |
| NH3 | 1 | 3 | 9 |

**How I read "qualifies".** Requirements 1 to 3 as you wrote them, plus at least one scored label. A planet whose four labels are all None fixes no label, so it is listed in (c) with its numbers, not here. WASP-15 b and KELT-7 b from your starting list fall in that group. Tell me if you want them treated differently; (e) lists what would change.

**How I applied "variant crosses -4".** Every free-chemistry result printed for a planet counts: other codes, reductions, resolutions, single visits and aerosol set-ups. If any printed interval crosses -4, or the limit lies above -4, the label is None. That is the strict reading of your rule. Table (e)1 lists the labels it removes.

Notation: `P` present, `A` absent, `None` not scored. Values are log10 VMR exactly as printed unless marked. Limits are quoted with their level. `$P` = `/Users/fanyang/BioSignatureDetectionModel/v3/data/jwst/panel`.

---

## (a) Qualifying planets

| MCS name | Source (first author, year, arXiv, DOI) | Data location and local path | Modes, range | CH4 | H2O | CO2 | NH3 | Tentative | Twin log10 (CH4, H2O, CO2, NH3) |
|---|---|---|---|---|---|---|---|---|---|
| TrES-4b | Meech 2025, 2503.24280, 10.1093/mnras/staf530 | Zenodo 10.5281/zenodo.15097695; `$P/TrES-4b/Meech2025_zenodo15097695/` | NIRSpec G395H, 2.7765-5.1797 um | **A** <-7.07 (2 sigma) | **P** -2.98 +0.68/-0.73 | **A** -6.86 +0.62/-0.65 | None (not in model) | none | -8.07, -2.98, -6.86, -7.0 |
| HATS-6b | Guzmán Caloca 2026, 2608.16990, 10.3847/1538-3881/ae9858 | Zenodo 10.5281/zenodo.21500595; `$P/HATS-6b/GuzmanCaloca2026_zenodo21500595/` | NIRSpec PRISM, 0.62-5.38 um | **A** -5.38 +0.18/-0.19 | **A** -4.88 +0.25/-0.24 | **A** -7.59 +0.41/-0.44 | **A** -6.03 +0.18/-0.19 | none | -5.38, -4.88, -7.59, -6.03 |
| TOI-5205b | Cañas 2026, 2502.06966, 10.3847/1538-3881/ae4976 | AJ data behind Fig. 2; `$P/TOI-5205b/Canas2026_aj_ae4976_fig2data/` | NIRSpec PRISM, 0.5224-5.5843 um (retrieved 0.6-5.3) | None: -5.89 +0.16/-0.15, but one variant crosses | **A** -10.0 +1.4/-1.3 | **A** -8.6 +0.7/-1.6 | **A** -7.06 +0.44/-1.98 | none | -5.89, -10.0, -8.6, -7.06 |
| TOI-6894b | Zhang 2026, 2606.23774, 10.3847/2041-8213/aea4ae | ApJL data behind Fig. 2; `$P/TOI-6894b/Zhang2026_apjl_aea4ae_fig2data/` | NIRSpec PRISM, 0.8029-5.2955 um | **P** -1.67 +0.16/-0.16 | **P** -1.40 +0.12/-0.15 | None: -4.18 +0.41/-0.48 | **P** -3.19 +0.24/-0.26 | none | -1.67, -1.40, -4.18, -3.19 |
| TOI-199b | Bello-Arufe 2026, 2511.15835, 10.3847/1538-3881/ae4fba | Zenodo 10.5281/zenodo.17653116; `$P/TOI-199b/BelloArufe2026_zenodo17653116/` | NIRSpec G395M NRS1, 2.872-5.168 um | **P** -2.46 +0.85/-1.15 | None: <-2.05 (2 sigma) | None: <-3.47 (2 sigma) | None: <-1.78 (2 sigma) | none | -2.46, -3.05, -4.47, -2.78 |
| LP791-18c | Roy 2026, 2512.10876, 10.1038/s41550-025-02723-3 | Zenodo 10.5281/zenodo.17316562; `$P/LP791-18c/Roy2026_zenodo17316562/` | NIRSpec PRISM, 0.6804-5.3942 um, gap 1.08-1.71 um | **P** -0.86 +0.53/-1.70 | None: <-0.71 (2 sigma) | None: <-1.97 (2 sigma) | None: <-1.09 (2 sigma) | none | -0.86, -1.71, -2.97, -2.09 |
| NGTS-2b | Fairman 2026, 2603.18332, 10.1093/mnras/stag528 | Zenodo 10.5281/zenodo.18864528; `$P/NGTS-2b/Fairman2026_zenodo18864528/` | NIRSpec G395H, 2.7730-5.1663 um | **A** -9.83 +1.58/-1.50 | None: -3.51 +0.73/-0.67 | None: -7.38 +0.94/-0.81, but variants cross | **A** -9.23 +1.92/-1.86 | none | -9.83, -3.51, -7.38, -9.23 |
| HAT-P-30b | Claringbold 2026, 2601.13104, 10.1093/mnras/stag143 | Zenodo 10.5281/zenodo.18682265; `$P/HAT-P-30b/Claringbold2026_zenodo18682265/` | NIRSpec G395H, 2.7662-5.1661 um | **A** <-6.1 (2 sigma) | None: -3.3 +/-0.8 | None: -8.1 +0.9/-1.8, but one variant crosses | None (tested, not reported) | none | -7.1, -3.3, -8.1, -7.0 |
| HAT-P-1b | Ashtari 2026, 2608.28538, 10.3847/1538-3881/aea90a | AJ data behind Fig. 2; `$P/HAT-P-1b/Ashtari2026_aj_aea90a_fig2data/` | NIRSpec G395H, 2.886-5.133 um (retrieved jointly with HST) | **A** -9.55 +1.65/-1.59 | **P** -1.70 +0.43/-0.71 | None: -3.99 +0.53/-0.66 | None (not in model) | none | -9.55, -1.70, -3.99, -7.0 |
| TOI-1130b | Barat 2026, 2605.02036, 10.3847/2041-8213/ae5f8b | ApJL data behind Fig. 2; `$P/TOI-1130b/Barat2026_apjl_ae5f8b_fig2data/` | NIRISS SOSS + NIRSpec G395H, 0.7366-5.1293 um | None: -4 +1/-2 (mass fraction) | **P** -0.7 +0.3/-1.1 (mass fraction; see block) | None: -1.2 +0.6/-1.9 (mass fraction) | None: <-2 (3 sigma, mass fraction) | none | -4.73, -1.48, -2.37, -3.75 (derived, see block) |
| V1298Taub | Barat 2025, 2507.08837, 10.3847/1538-3881/adec89 | Zenodo 10.5281/zenodo.16782958; `$P/V1298Taub/Barat2025_zenodo16782958/` | NIRSpec G395H, 2.7062-5.1465 um (retrieved jointly with HST) | **A** -6.22 +/-0.42 | None (30 sigma detection, no number printed) | None (35 sigma detection, no number printed) | None (limit only in a figure) | none | -6.22, -7.0, -7.0, -7.0 (H2O and CO2 knowingly wrong, see (e)6) |
| Kepler-51d | Libby-Roberts 2026, 2505.21358, 10.3847/1538-3881/ae33c0 | AJ Table 2 MRT; `$P/Kepler-51d/LibbyRoberts2026_aj_ae33c0_mrt/` | NIRSpec PRISM, 0.6015-5.2965 um (retrieved 0.6-4.5) | **A** 66 +72/-45 ppb | None (tested, no abundance) | None (tested, no abundance) | None (tested, no abundance) | none | -7.18, -7.0, -7.0, -7.0 |
| WASP-121b | Gapp 2025, 2506.02199, 10.3847/1538-3881/ad9c6e | AJ data behind Fig. 6; `$P/WASP-121b/Gapp2025_aj_ad9c6e_fig6data/` | NIRSpec G395H, 2.7285-5.1671 um | None (not reported) | None: variants disagree | **A** -8.64 +1.82/-1.99 (day), -8.80 +1.64/-1.89 (night) | None (not reported) | none | -7.0, -5.74, -8.64, -7.0 |

All 13 names are in `mcs_params.parquet`. All DOIs resolve at doi.org (checked 2026-10-30).

### Evidence for each planet

**TrES-4b.** Meech et al. 2025, MNRAS 539, 1381. Chosen: POSEIDON free chemistry, isothermal, cloud deck + haze, NRS offset, on the principal Tiberius R=100 spectrum: the paper's reference test (I). Read from Table 3 and Sec. 4.1.1. Table 3 prints means and 1 sigma, with 2 sigma limits for non-detections. H2O 8.4 sigma, CO2 4.0 sigma (R=100 Tiberius). Other free variants in Table 3, none crossing -4:
- H2O: POSEIDON -2.88 +/-0.61, -2.88 +0.61/-0.72, -2.89 +0.59/-0.53; pRT -3.29 +0.57/-0.59, -3.26 +0.51/-0.56, -3.18 +0.57/-0.65.
- CO2: -6.81 +0.56/-0.54, -7.11 +0.63/-0.67, -6.94 +/-0.56, -7.21 +0.56/-0.51, -7.13 +0.49/-0.52, -7.19 +0.59/-0.64.
- CH4: <-7.01, <-7.01, <-7.47, <-7.54, <-7.20, <-7.53.
Retrieved T 1180 +139/-138 K. Only one JWST paper on this planet was found.

**HATS-6b.** Guzmán Caloca et al. 2026, AJ 172, 193. Chosen: POSEIDON free chemistry, M2.5 (clouds + haze), co-added Eureka1 spectrum: the paper's adopted model. Read from Table 6; Sec. VI.3 repeats the numbers. M2.4 (clouds + TLS, Table 8) has the higher single-visit evidence but is rejected in the paper for unphysical spot temperatures. No printed variant crosses -4:
- M2.5 visits: H2O -5.39 +0.39/-0.47, -4.69 +0.37/-0.37; CH4 -5.54 +0.27/-0.28, -5.45 +0.32/-0.34; NH3 -5.89 +0.25/-0.25, -6.02 +0.30/-0.33; CO2 -7.79 +0.57/-0.69, -8.06 +0.73/-2.22.
- M2.4 co-added and visits: H2O -5.18 +0.28/-0.34, -9.74 +2.66/-2.86, -4.69 +0.38/-0.36; CH4 -5.30 +0.18/-0.19, -5.23 +0.18/-0.19, -5.22 +0.29/-0.31; NH3 -6.22 +0.21/-0.24, -6.05 +0.20/-0.22, -6.11 +0.31/-0.38; CO2 -7.26 +0.39/-0.42, -6.85 +0.49/-0.50, -7.50 +0.74/-1.89.
TauREx results are in figures only. Detected, per Table 7 (leave-one-out): CH4 <12.80 sigma, H2O <7.03 sigma, NH3 <5.25 sigma, CO2 <3.38 sigma. They are `A` because they lie below 10^-4. Retrieved T 514 +35/-27 K (T_eq about 712 K).

**TOI-5205b.** Cañas et al. 2026, AJ 171, 260. Chosen: POSEIDON free chemistry, M3.1 (TLS + clear atmosphere), co-added ExoTiC-JEDI: the paper's favoured model (Sec. V.1.3). Read from Table 4; NH3 from the M3.1 + NH3 run in Appendix F.4.
- CH4 is None because Table 11 (M3.2, clouds) Visit 2 gives -4.84 +1.10/-0.54, whose upper bound is -3.74. All other CH4 values lie below -5: M3.1 -5.78, -5.54, -6.54; M3.2 -5.81, -6.59, -5.87; Eureka (Table 12) -5.99, -5.58, -6.66, -6.12; 40 nm JEDI (Table 13) -6.09, -5.69, -6.70, -6.05. Detected at 10 sigma.
- H2O and CO2 do not cross in any printed variant. The largest upper bounds are H2O -7.47 (Table 12 Visit 1, -9.04 +1.57/-1.88) and CO2 -5.2 (Table 11 Visit 2, -6.4 +1.2/-1.0).
- NH3 has one printed value and no evidence for it (ln B = 0.4 as printed in F.4; -0.4 in Sec. V.1.3).
Retrieved T 773 +51/-54 K. The spectrum is heavily contaminated by the star below 3.5 um, as the paper states.

**TOI-6894b.** Zhang et al. 2026, ApJL 1009, L28. Chosen: "fully free retrieval" in Appendix B, vertically constant VMRs, on 0.8-5.2 um. The paper's headline results are a semi-free retrieval and grid fits, which are not free chemistry (see (e)8). The sulfur-partition free variant gives H2O -1.36 +0.11/-0.13, NH3 -3.14 +0.24/-0.26, CH4 -1.64 +0.16/-0.16, CO2 -4.11 +0.40/-0.45, so no label changes. ln B from Fig. 2 (semi-free model): CH4 322.1, H2O 30.5, NH3 11.0, CO2 5.0. I read that figure for these legend values only, not for abundances. Retrieved T 447 +18/-16 K. Appendix B also reports the free run implies [M/H] = 1.65 +0.08/-0.10, well above the 3-10x solar of the headline models.

**TOI-199b.** Bello-Arufe et al. 2026, AJ 171, 354. Chosen: ExoTR free retrieval on the reference Eureka reduction: the only tabulated free retrieval. Read from Table 2 and Sec. III.1. CH4 Bayes factor 790.16, "Decisive evidence". Limits are 95%. Aurora is figure-only; the text says its CH4 is "comparable ... although generally higher", so no sign of a crossing. Retrieved T 325 +88/-82 K. The spectrum covers NRS1 only.

**LP791-18c.** Roy et al. 2026, Nature Astronomy 10, 371. Chosen: SCARLET free chemistry, nominal retrieval with hazes + Mie clouds, from the **published** Extended Data Table 3 (read from the journal's table image). **The arXiv preprint has different numbers** (CH4 -1.24 +0.89/-1.34, H2O <-0.59, NH3 <-1.14), so use the published ones. Variants in the same table: +TLS CH4 -0.35 +0.15/-0.27, CLR -0.23 +0.23/-0.63; H2O <-0.72, <-0.77; CO2 <-1.45, <-2.43; NH3 <-1.11, <-1.14. CH4 4.53 sigma (Extended Data Table 2). The limits are 95% and all lie well above -4. T used a 355 +/-50 K Gaussian prior; the retrieved value is not in the table.

**NGTS-2b.** Fairman et al. 2026, MNRAS 548, stag528. Chosen: POSEIDON free Model I (with SO) on ExoTiC-JEDI R=400, first row of Table 4 and Sec. 4.3; SO is preferred at 2.0 sigma. The paper reports competing models and does not settle them, so all of Table 4 is treated as variants.
- H2O None: Model I straddles (-4.18 to -2.78); Model II -1.17 +0.11/-0.23.
- CO2 None: Model II JEDI R=400 -4.32 +0.33/-0.46 (upper -3.99), JEDI R=100 -3.85 +0.75/-0.91.
- CH4 A: largest upper bound among all free rows is -5.24 (Model II, mu<3: -7.04 +1.80/-2.88).
- NH3 A: POSEIDON rows only, largest upper bound -5.26 (Model II: -7.71 +2.45/-2.66).
Both CH4 and NH3 are non-detections with broad posteriors. Retrieved T 633 +93/-65 K (T_eq 1468 K).

**HAT-P-30b.** Claringbold et al. 2026, MNRAS 546, stag143. Chosen: pRT free chemistry on Eureka R=400, quoted in Sec. 4.2.2, full table in Appendix A Table 3. Limits are 2 sigma.
- CH4 A: every free run is below -4 (<-5.9, <-6.1, <-4.1, <-5.9, <-5.7; BeAR <-5.7, <-6.0, <-6.1, <-6.2).
- H2O None: -3.3 +/-0.8 straddles (BeAR -2.7 +/-1.0, -2.9 +0.9/-1.0 do not, but the chosen model does).
- CO2 None: nominally absent, but BeAR Eureka R=100 gives -7.6 +6.9/-2.0, the bimodal run the paper says disappears at R=400.
- NH3 None: tested but left out of the primary retrievals, no number printed.
Retrieved T 666 +115/-81 K (T_eq 1630 K); the paper discusses possible modelling causes.

**HAT-P-1b.** Ashtari et al. 2026, AJ 172, 275. Chosen: POSEIDON free chemistry on the panchromatic HST STIS + WFC3 + JWST G395H spectrum, Table 1 and Sec. III.1. SANSAR agrees: H2O -1.68 +0.24/-0.30, CO2 -4.07 +0.34/-0.36, CH4 -9.56 +1.73/-1.58. Text prints H2O -1.70 +0.43/-0.72 and CO2 -3.98 +0.53/-0.66, table prints -0.71 and -3.99; no label depends on the difference. H2O strong detection (Table 2, log10 B 8.91). No G395H-only retrieval was run, and the file holds JWST data only.

**TOI-1130b.** Barat et al. 2026, ApJL 1002, L32. Chosen: pRT free chemistry with jitter term, the fit whose detection significances the paper adopts, Table 2 and Sec. IV.2. **Table 2 is in log mass fraction, not VMR.** I converted using the mu of about 3 that the paper derives from the free-retrieval medians (VMR = mass fraction x mu / m_i). Its equilibrium fit gives mu about 5.5; no label below changes with that value.
- H2O P: 1 sigma lower bound in mass fraction -1.8, which is VMR -2.58 at mu=3 and -2.69 at mu=2.3. Every column is well above -4 (others -0.6 +0.3/-0.5, -0.6 +0.2/-0.5, -0.3 +0.2/-0.5, -0.4 +0.1/-0.2). 7.5 sigma.
- CO2 None: VMR interval -4.27 to -1.77 at mu=3 under the adopted fit. It clears -4 only for mu above about 5.5.
- CH4 None: VMR interval -6.73 to -3.73; "tentative" 2 sigma in the paper.
- NH3 None: limit converts to about -2.75.
Twin values are these conversions and are my arithmetic, not printed numbers. Retrieved T <600 K (3 sigma). The retrieval also used TESS and CHEOPS points, which are not in the files.

**V1298Taub.** Barat et al. 2025, AJ 170, 165. Chosen: PICASO free chemistry on HST WFC3 + JWST G395H (Eureka), the only free retrieval. CH4 from Sec. IV.1 (-6.22 +/-0.42); the abstract and Summary print -6.2 +0.3/-0.5. 6 sigma detection, `A` by threshold. SPARTA run is stated consistent within 1 sigma. Dai et al. 2026 (2606.00974) and Murphy et al. 2026 (2606.03740) are about planets e and c, not b. Retrieved T 444 +/-38 K.

**Kepler-51d.** Libby-Roberts et al. 2026, AJ 171, 221. Chosen: ExoTR free retrieval, tholin haze + CH4, Sec. IV.1 and Table 3. CH4 "66 +72/-45 ppb" converts to log10 -7.18 +0.32/-0.50 (my arithmetic). Preferred over haze alone by only 2.2 sigma, and with stellar contamination added no gas gains evidence. The H2O, CO2 and NH3 runs give evidence only, no abundance. PLATON runs are equilibrium. T fixed at 367 K. Read from arXiv v1, which says "submitted" (see (e)3).

**WASP-121b.** Gapp et al. 2025, AJ 169, 341. Two free retrievals: NEMESIS, split by hemisphere (Table 3, Sec. III.1), and PETRA (Table 6, Sec. III.4). The paper prefers the phase-curve spectrum (PC) to the transit-only one (TR).
- CO2 A: only NEMESIS fits it. TR gives -8.20 +2.11/-2.35 (day) and -9.02 +1.92/-1.54 (night). All upper bounds below -6.
- H2O None: PETRA -5.74 +0.12/-0.11 (PC), -4.95 +0.58/-0.46 (TR) are absent; NEMESIS nightside -3.13 +1.33/-1.32 (PC) straddles and -2.42 +0.69/-0.99 (TR) is present; dayside -8.44 +2.29/-2.21, -7.83 +2.63/-2.49 are absent.
- Twin H2O uses PETRA PC because it is the only single global value; CO2 uses NEMESIS PC dayside.

---

## (b) Parsing recipes

All files are plain data. Read them with `python -I` from outside the directory. For MRT files take the rows after the last line starting with `-----`, split on whitespace. Files were test-loaded; the counts below are what loaded.

| Planet | Use this file (under the local path) | Columns and units | n, range | Reduction, mode |
|---|---|---|---|---|
| TrES-4b | `transmission_spectra/tiberius/tiberius_R100_transmission_spectrum_depths_NRS{1,2}_23Oct24.txt` (concatenate) | um centre; um **full** width; depth fraction; error fraction, symmetric | 61, 2.7765-5.1797 | Principal Tiberius, the labels' retrieval. R400 in the same folder (241). `eureka/` holds Eureka (centre, half-width, fraction, err_down, err_up). `tiberius_v2/` (ppm) appears to be the second reduction of the paper's Appendix; do not use it as primary. The principal choice was confirmed by matching the archive's Tiberius entry |
| HATS-6b | `data/BINNED_coadd.txt` | no header: um centre; half-width (0.02 um, inferred from 0.04 um spacing); depth fraction; error fraction, symmetric | 120, 0.62-5.38 | Eureka1 co-added, the labels' retrieval. `NATIVE_visit{1,2}.txt` are per-visit |
| TOI-5205b | `toi5205_coadd_exotic_pixel.txt` | one header line: wavelength, bin_width (half-width), rp^2_value, rp^2_error; um, um, fraction, fraction, symmetric | 470, 0.5224-5.5843; trim to 0.6-5.3 | ExoTiC-JEDI co-added, the labels' retrieval. The ReadMe says five columns; that holds only for the single-transit files (errorneg, errorpos), the co-added ones have four |
| TOI-6894b | `dbf2.txt` (MRT) | um; (Rp/Rs)^2 in **percent**; error in percent, symmetric | 372, 0.8029-5.2955; paper models 0.8-5.2 | One reduction, not named in the file (see (e)9) |
| TOI-199b | `transmission_spectrum_toi199_eureka_nativespectral.txt` | header: wavelength, bin_width (half-width), rp^2_value, rp^2_errormax; um, um, fraction, fraction, symmetric | 575, 2.872-5.168 | Eureka native, the labels' retrieval. `..._tswift.txt`: um, ppm, ppm; 25 points |
| LP791-18c | `LP79118c/LP79118c_fullRes_spectrum.spec` and `LP79118c/LP79118c_R25_clean_u1u2_EurekaV1_Feb2025_newLD.spec` | ECSV, comma-delimited: `wave`, `waveMin`, `waveMax` um; `yval` ppm; `yerrLow`, `yerrUpp` ppm, asymmetric; `ignore` all 0 | full 374, 0.6804-5.3942, gap 1.083-1.712; R25 40, 0.6925-5.1217, gap 1.082-1.779 | Eureka + ExoTEP. R25 is the Fig. 2 spectrum; which binning the retrieval used is not stated (see (e)9). `spec_ExoTEDRF_R50.spec` is the independent check. Only these three files were kept from the 867 MB zip, which also holds pickles |
| NGTS-2b | `ExoTiC-JEDI_data_products/ExoTiC-JEDI_R400_transit_spectrum.nc` (NetCDF) | coord `central_wavelength` um; coord `bin_half_width` um; `transit_depth`, `transit_depth_error` in (Rp/Rs)^2, symmetric | 239, 2.7730-5.1663 | ExoTiC-JEDI R=400, the labels' retrieval. R100 file: 61, 2.7767-5.1641 |
| HAT-P-30b | `transmission_spectra/Eureka_R400.dat` | header: um centre; um half-width; depth fraction; err_down; err_up | 241, 2.7662-5.1661 | Eureka R=400, the labels' retrieval. `Eureka_R100.dat` 61. `Tiberius_R{100,400}.txt`: um, um, ppm, ppm |
| HAT-P-1b | `dbf2.txt` (MRT) | um; half-width um; `EurTD` ppm; `e_EurTD`; `E_EurTD`; `FireTD`; `e_FireTD`; `TswTD`; `e_TswTD` | 106, 2.886-5.133, gap 3.706-3.833 | Use `EurTD`, the primary Eureka reduction. JWST only |
| TOI-1130b | four `.mrt` files: `toi1130b_niriss_order{1,2}_..._final_modelling.mrt`, `toi1130b_nrs{1,2}_..._50pixel_spectrum.mrt` | um; lower and upper half-widths um; depth ppm; error ppm, symmetric | order 2: 2 pts; order 1: 40; NRS1: 29; NRS2: 39. Total 110, 0.7366-5.1293 | NIRISS exoTEDRF + NIRSpec Eureka at 50-pixel binning, the labels' retrieval |
| V1298Taub | `final spectrum/v1298b_final_combined_spectrum` (space in the folder name) | no header: um; depth ppm; error ppm, symmetric; no bin width | 342, 2.7062-5.1465 | Eureka, matches the archive's 342-point Eureka entry. `v1298_sparta_nrs{1,2}.txt`: SPARTA. `v1298_spectra_emcee_new.txt`: HST WFC3, 17 points, not JWST |
| Kepler-51d | `ajae33c0t2_mrt.txt` (MRT) | um; depth **percent**; lower error %; upper error %; then limb-darkening and spot columns to ignore | 408, 0.6015-5.2965; ExoTR used 0.6-4.5 | Eureka native, the labels' retrieval |
| WASP-121b | `dbf6a.txt` (MRT; R about 600) | um; `TDepth` %, `E_TDepth` (upper), `e_TDepth` (lower); `PDepth` %, `E_PDepth`, `e_PDepth` | 349, 2.7285-5.1671 | Use `PDepth` (phase-curve, the paper's preferred); `TDepth` is transit-only. `dbf6b.txt` is R about 3000. Ignore `._dbf6b.txt` |

---

## (c) Examined and rejected

**1. Fix at least one label, but no public spectrum found.** Worth a request to the authors. See (e)4.

| Planet | Labels it would give | Source | Why not |
|---|---|---|---|
| HATS-75b | all four A: CH4 -5.73 +0.40/-0.42; H2O -8.6 +/-2.0; CO2 -6.13 +0.76/-1.05; NH3 -8.6 +1.8/-2.1 (Model A; Model B also A) | Ashtari 2026, 2604.07268, 10.3847/1538-3881/ae552c, POSEIDON retrieval table | Not on Zenodo or the archive; no figure data at the journal's public content host |
| TOI-5293Ab | NH3 A (-5.7 +0.6/-1.8; -7.9 +2.1/-2.6). CH4 -4.1 +/-0.4 and H2O -4.3 +0.7/-0.8 straddle; CO2 crosses in Visit 1 | Kanodia 2026, 2603.16464, 10.3847/1538-4357/ae801f | Same |
| WASP-178b | H2O A (-5.40 +/-0.82, JWST-only); CO2 A (-7.72 +/-0.65) | Lothringer 2025, 2503.15472, 10.3847/1538-3881/adc117; Saha 2025, 2510.11479 (no DOI found) | Same |
| HD106315c | H2O P (-1.40 +0.40/-0.74); NH3 P, tentative (-2.47 +0.57/-0.89); CH4 A (<-5.86, 2 sigma) | Dyrek 2026, 2609.09893 (no DOI found), MIRI + HST | No data statement or deposit found |
| WASP-69b | H2O -1.70 +0.44/-0.78 evening limb only; morning <-3.20; other limits above -4 (3 sigma) | Wang 2026, 2608.19756 | Limb-resolved only, no public spectrum |

**2. Requirements met or nearly met, but all four labels None.**

| Planet | Why no label | Source |
|---|---|---|
| WASP-15b | H2O: pRT -2.83 +0.59/-0.95 and -3.14 +0.84/-0.82 present, BeAR Eureka R=100 -5.77 +1.0/-0.61 absent. CO2 -4.40 +0.73/-1.02 straddles. CH4, NH3 not printed | Kirk 2025, 2410.08116, 10.1093/mnras/staf208, Zenodo 14779026 |
| KELT-7b | Papers disagree across -4. H2O -7.05 +0.59/-1.93 (Ahrer) vs -1.17 +0.12/-0.16 (Schmidt). CO2 -8.81 +0.44/-0.58 vs -3.35 +0.31/-0.38. CH4, NH3 limits in figures only | Ahrer 2025, 2509.12479, 10.1093/mnras/staf1535, Zenodo 17092001; Schmidt 2026, 2607.06708 |
| GJ3090b | CH4 2 sigma limit "~100 ppm" sits on the threshold; H2O, CO2 only unnumbered lower limits; NIRISS limits above -4 apart from one visit | Ahrer 2025, 2504.20428, 10.3847/2041-8213/add010 |
| TOI-3235b | CH4 -2.60 +0.66/-1.95 straddles; bimodal with TLS (-5.18 and -2.33); others in a figure | Ko 2026, 2608.04194, 10.3847/1538-3881/ae9545 |
| LTT9779b | No printed VMR for the four gases in either paper. A conference poster abstract (Zenodo 21142216) gives log MMR limits; not a paper, not VMR | Radica 2024, 2401.15548, 10.3847/2041-8213/ad20e4; Stamer 2026, 2609.36206 |
| GJ1132b | Prior-dominated; intervals straddle (CH4 -5.02 +1.34/-3.44; H2O -0.96 +0.70/-1.34 vs -5.40 +3.59/-4.27) | Senobar 2026, 2608.24803, 10.1093/mnras/stag1606 |
| LHS1140b | H2O <-2.94, CH4 <-2.78; CO2 -7.62 +3.79/-2.67 straddles | Cadieux 2024, 2406.15136, 10.3847/2041-8213/ad5afa |
| GJ341b | CO2 -2.1 +1.2/-1.9 (Tiberius) vs -8.7 +2.6/-2.1 (Tswift); paper calls it noise | Kirk 2024, 2401.06043, 10.3847/1538-3881/ad19df |
| GJ486b | H2O >10% only for Eureka, and Tiberius allows a clear H2 atmosphere; TLS alternative | Moran 2023, 2305.00868 |
| L98-59b | Only SO2 favoured; CO2 <84%; others unconstrained | Bello-Arufe 2025, 2501.18680 |
| TOI-260b, TOI-125b, TOI-125c, LHS475b, LHS1140c | Free posteriors shown in figures only or described as spanning the prior; no number | Meech 2026, 2602.22329; Chai 2026, 2609.35607; Lustig-Yaeger 2023, 2301.04191; Cadieux 2024 |
| TOI-270b | Only water mole-fraction statements in a 2-gas H2O/H2 model, "inconclusive" | Coulombe 2025, 2509.14224 |
| TRAPPIST-1b, c, d, e, f | Surface or partial pressure limits, or CLR posteriors in figures; no scored VMR | Lim 2023, 2309.07047; Rathcke 2025, 2412.16541; Radica 2025, 2409.19333; Piaulet-Ghorayeb 2025, 2508.08416; Espinoza 2025, 2509.05414; Glidden 2025, 2509.05407; Lim 2026, 2608.17207 |
| WASP-43b | Transit-only MIRI retrieval reports model evidence only ("do not find any H2O contribution"); other JWST retrievals are emission | Binet 2026, 2609.05264; Bell 2024 |

**3. No free-chemistry retrieval: metallicity, mean molecular weight or forward grids only.**

TOI-836b (Alderson 2024, 2404.00093), TOI-776b (Alderson 2025, 2501.14596), TOI-776c (Teske 2025, 2502.20501; Rogers 2026, 2601.14254, H2-He-H2O mix only), HD15337c (Wallack 2026, 2602.22327), L98-59c (Scarsdale 2024, 2409.07552), L168-9b (Alam 2024, 2411.03154), GJ357b (Redai 2025, 2507.07165, grids and H2O-H2 mixtures; Taylor 2025, 2505.24462, forward models), LTT1445Ab (Batalha 2026, 2605.15003, three-gas fits with only approximate limits near 10%), TOI-1685b (Luque 2025, 2412.03411; Fisher 2026, 2512.15338, single-gas grids, fixed-abundance tests), WASP-127b (Fu 2025, 2501.02081, forward fit across the sample), K2-22b (Tusay 2025, 2501.08301, disintegrating planet), HD86226c (Kahle 2025, 2507.13439, HST only).

**4. Not in the MCS.** HAT-P-14 b, TOI-836 c, TOI-1130 c, GJ 436 b, V1298 Tau c and e, WD 1856+534 b, K2-141 b, NGTS-10 b, Kepler-167 e. Your used list was excluded on sight.

**5. In the MCS, but no published JWST transmission spectrum with a free retrieval found.** Includes those in your list without a JWST transmission paper: HAT-P-11b, HAT-P-32b, HAT-P-41b, HAT-P-65b, HATS-72b, HD149026b (emission only), HD80606b (emission and eclipse), HD97658b, Kepler-12b, Kepler-51b, LHS3844b, LTT3780b (eclipse), TOI-1468b, TOI-1468c, TOI-178d to g, TOI-270c, TOI-3714b, TOI-3757b, TOI-3984Ab, TOI-3884b (Murray 2026, 2603.15414, is about the starspot; no planetary free retrieval), TOI-4010b to d, TOI-421c, TOI-561b (phase curve), 55Cnce (emission), WASP-6b, WASP-12b, WASP-18b (emission), WASP-19b, WASP-31b, WASP-63b, WASP-76b, WASP-79b, WASP-101b, WASP-103b (a talk on Zenodo, no paper found), TOI-1899b, TOI-4336Ab, TOI-700c, AU Mic b, HD207496b. Targets not in JWST programme lists were not searched one by one beyond the name matching in (d).

---

## (d) Search

Text for a Methods section:

> We searched for planets in the Ariel Mission Candidate Sample (977 names) with published JWST transmission spectra on 2026 October 30. We queried the arXiv API for astro-ph.EP records whose abstracts mention JWST or a JWST instrument together with transmission, transit, terminator, retrieval, free chemistry or abundance terms (three queries, 907, 652 and 725 records), and for titles naming the programmes COMPASS, BOWIE-ALIGN, GEMS JWST, DREAMS, MANATEE, Hot Rocks, JWST-SUPER, KRONOS, SPACE, Grand Tour, Rocky Worlds, ExoEcho, JWST-TST, HUSTLE and JWST MEP (77 records), 1456 unique in total, and matched titles and abstracts against the sample names. We ran separate title and abstract queries for 93 candidate planets or systems (1544 records). Target coverage was cross-checked against the STScI TrExoLiSTS JWST list (124 targets with transit or phase-curve entries, 89 in the sample, 74 with archived data at the time of the list) and the NASA Exoplanet Archive Atmospheric Spectroscopy table (409 JWST rows, 274 transmission, 65 planets). Data releases were sought through the Zenodo API (16 keyword queries, first 150 hits each, 1424 unique records, then per-target queries), the papers' data statements, and journal figure-data files. Retrieved abundances were read from the arXiv LaTeX source of each paper and, for Roy et al., from the published Extended Data table, which differs from the preprint.

Details not in the text above:
- arXiv queries: `cat:astro-ph.EP AND abs:JWST AND (abs:transmission OR abs:transit OR abs:transiting OR abs:terminator)`; the same with `(abs:"James Webb" OR abs:NIRSpec OR abs:NIRISS OR abs:NIRCam OR abs:MIRI OR abs:SOSS OR abs:G395H OR abs:PRISM)`; `cat:astro-ph.EP AND (abs:JWST OR abs:NIRSpec OR abs:NIRISS OR abs:"James Webb") AND (abs:retrieval OR abs:retrievals OR abs:"free chemistry" OR abs:abundance OR abs:abundances)`; and the `ti:` programme query.
- Exoplanet Archive: TAP table `spectra`, `facility like '%James%'`. TrExoLiSTS: `trexolists.csv`. Its statuses lag; some targets it lists as not yet observed may have been.
- Zenodo terms: JWST transmission spectrum/spectra, JWST NIRSpec G395H transit spectrum, NIRISS SOSS, NIRSpec PRISM, MIRI LRS, NIRCam, JWST exoplanet atmosphere retrieval, BOWIE-ALIGN, JWST COMPASS, GEMS JWST, JWST-TST DREAMS, MANATEE JWST, Hot Rocks Survey JWST.
- General web search for GEMS targets, Grand Tour targets and sub-Neptune free retrievals found nothing beyond the above. GEMS programme site read directly.
- Candidate papers read in source form: 69. DOIs from arXiv metadata and Crossref, each checked to resolve at doi.org.

---

## (e) Doubts

1. **The strict variant rule removes some labels.** Your call whether to relax any:

| Planet, gas | Strict label | Looser reading | What removes it |
|---|---|---|---|
| TOI-5205b CH4 | None | A | One single-visit cloudy run (M3.2 Visit 2); the paper favours cloud-free M3.1 |
| HAT-P-30b CO2 | None | A | One bimodal BeAR run at R=100; the paper says the mode vanishes at R=400 |
| WASP-15b H2O | None, planet rejected | P | BeAR runs, which give about -5.8 once high-metallicity solutions are excluded |
| KELT-7b H2O, CO2 | None, planet rejected | P, P if only Schmidt 2026 is used | Earlier paper on less data |

2. **`A` often means "detected, but below 10^-4".** This applies to HATS-6b (H2O, CH4, NH3 all detected), V1298Taub CH4 (6 sigma), TrES-4b CO2 (4 sigma) and Kepler-51d CH4. A classifier that responds to real low-abundance features may fire on these, and that would score as a false positive under the rule. Worth deciding before unblinding how these are reported.

3. **Most numbers are from arXiv sources, not the published versions.** IOP and OUP article pages returned bot checks (one IOP page loaded before that), and I did not try to get past them. LP791-18c shows this matters: the preprint and the published table differ. arXiv comments say accepted or published for all the others except **Kepler-51d (v1, "submitted")** and **HAT-P-1b (v1, "final revision pending")**. Labels would change only if a value moved across -4; the scored labels closest to it are TrES-4b H2O (pRT lower bound -3.88), HATS-6b H2O (Visit 2 upper bound -4.32) and TOI-199b CH4 (lower bound -3.61). Someone with a browser should check TOI-6894b, HATS-6b and Kepler-51d against the journal pages.

4. **Three "no spectrum" verdicts are not fully verified.** For HATS-75b, TOI-5293Ab and WASP-178b I could not open the article pages. I tried direct file names on the journal's public content host, following the pattern seen on the one page that loaded: figures 1-10 and tables 1-6. Nothing was there. Those same guesses found the HAT-P-1b, Kepler-51d, TOI-5205b, TOI-1130b and WASP-121b files, whose ReadMe or header confirmed the paper. A browser check of the three pages takes a few minutes, and HATS-75b would add four more `A` labels.

5. **The label balance is thin in places.** There are no CO2 present labels, and NH3 present rests on TOI-6894b alone. HAT-P-1b CO2 (-3.99) and TOI-6894b CO2 (-4.18) are the near misses.

6. **V1298Taub twin is knowingly wrong.** H2O and CO2 are detected at 30 and 35 sigma, but no number is printed, so your rule gives -7.0 for both. A twin rendered that way lacks features the planet clearly has. The Zenodo record has posterior files (`atmospheric models.zip`) that I did not use, because labels and twins were to come from the papers. Consider leaving this planet out of the twin comparison or allowing those posteriors. The same logic applies more weakly to any planet where a gas is unreported but likely there.

7. **Joint retrievals versus JWST-only files.** HAT-P-1b and V1298Taub were retrieved with HST data, and TOI-1130b with TESS and CHEOPS. The local files are JWST only, except that V1298Taub also has the HST table. HD106315c would be the same.

8. **TOI-6894b rests on an appendix.** The paper's headline is the semi-free retrieval and 3-10x solar models; the free run implies 35-54x solar. Semi-free absolute VMRs are only in a figure, so I could not test whether they cross -4. NH3 is the label most exposed to this. Under your rule the free run is the one that counts, so I kept the labels.

9. **Files whose exact link to the retrieval is not stated.** TOI-6894b `dbf2.txt` holds one of the two reductions in Fig. 2 without naming it. LP791-18c gives two binnings and does not state which was retrieved (a commented-out draft line in the source says full resolution). The HATS-6b column 2 meaning is inferred.

10. **Table and section numbers** are as rendered in the arXiv HTML of each paper (Nature page for LP791-18c). Journal numbering may differ.

11. **Low retrieved temperatures.** Several chosen models retrieve temperatures far below T_eq: HAT-P-30b 666 K vs 1630 K, NGTS-2b 633 K vs 1468 K, V1298Taub 444 K vs 670 K, HATS-6b 514 K vs 712 K. The papers flag this. It matters for how the twin is rendered; temperatures are in the blocks above.

12. **Possible earlier exposure.** `$P/WASP127b/Fu2025_zenodo13366851/Spectra/WASP 121b.txt` was already on disk before this work. I used Gapp et al.'s own Fig. 6 data instead. If any earlier run touched that file, WASP-121b is not unseen; I did not look at results to check. Directories for GJ1132b, GJ357b, LTT1445Ab, TOI-776b, TOI-776c and TOI-836b also predate this work; all six are rejected here.

13. **Stellar contamination.** TOI-5205b is strongly affected below 3.5 um, and HATS-6 and V1298 Tau show spots or activity. Abundances there depend on the contamination model.

14. **Fetch script.** A 20-line block was added at the end of the panel branch of `v3/fetch_jwst_data.sh`; nothing else changed. I tested all of it in a scratch tree except the LP791-18c 867 MB line, whose steps I ran by hand. Downloads matched the panel files byte for byte where compared. Not committed.
