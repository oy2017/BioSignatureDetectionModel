# Independent data survey: JWST, HST WFC3 G141 and non-TauREx model grids

Compiled 2026-10-30. No classifier and no verdict-computing script was run, and `v3/results/` was not opened. Panel names were read statically from `router_real.py` and `jwst_panel.py` (LABELS keys and EXTRA). Nothing large was saved. Archive heads were streamed into memory to confirm formats (at most 3 MB from a Google Drive tar, and the first members of the 8 GB Zenodo tar).

**Result**

| Source | Planets | Labels | Present | Absent | Comment |
|---|---|---|---|---|---|
| A, clean | 2 (V1298 Tau c, HAT-P-14 b) | 6 | 1 | 5 | Neither is in the Ariel MCS |
| A, outside any H2-rich hot/warm model domain | 1 (WD 1856 b) | 1 | 1 | 0 | I would not score it |
| A, data only on request | 1 new (V1298 Tau e), plus the 4 you named | see A2 | | | No public spectrum in any of them |
| B, strict across five population studies plus one JWST cross-check | 12 | 12, all H2O | 6 | 6 | 10 of the 12 are in the MCS |
| B, Edwards et al. 2023 alone | 20 | 20, all H2O | 14 | 6 | The looser count |
| C | Recommended: Goyal et al. 2019 generic ATMO grid, 73,728 models, per-model VMRs, 9.2 GB | | | | Second: Crossfield et al. 2025 VULCAN + petitRADTRANS grid as a disequilibrium stress test |

A and B together are thin for the molecules that matter. There is still **no new CO2 present label and no new NH3 or CH4 present label on a clean planet**. V1298 Tau c appears in both A and B and counts once for H2O. HD 106315 c has a public HST spectrum (B) but no public JWST spectrum (A2).

**Label rule, as in `jwst_fourth_batch_survey.md`.** Present if the interval is wholly above log10 VMR = -4, absent if wholly below, None otherwise. If any printed free-chemistry result for the planet crosses -4 or disagrees, the label is None.

---

## A. JWST transmission spectra outside the panel and batch 4

### A1. Public spectrum and at least one fixed label

| Planet (MCS?) | Paper | Spectrum | Mode, range | CH4 | H2O | CO2 | NH3 |
|---|---|---|---|---|---|---|---|
| V1298 Tau c (not in MCS) | Murphy et al. 2026, AJ 172, 66, arXiv 2606.03740, DOI 10.3847/1538-3881/ae7a32 | Zenodo 10.5281/zenodo.20547576, `data.zip` 28 MB, folder `data/spectra/` (e.g. `V1298Tauc_SOSS_LCtest2_6spots_R300.txt`). Posteriors in `data/retrievals/` | NIRISS SOSS, about 0.8-2.8 um | **A** -8.38 +2.36/-2.20; -8.45 +2.22/-2.24 | **P** -1.83 +0.68/-0.77; -1.59 +0.60/-0.65 | **A** -7.35 +2.81/-2.89; -7.12 +2.96/-2.90 | **A** -8.30 +2.48/-2.28; -8.43 +2.26/-2.20 |
| HAT-P-14 b (not in MCS) | Liu et al. 2025, AJ 169, 335, arXiv 2504.08903, DOI 10.3847/1538-3881/adcba7 | NASA Exoplanet Archive, Atmospheric Spectroscopy table: bibcode 2025AJ....169..335L, SOSS 137 points (0.60-2.76 um) and G395H 63 points (2.74-5.13 um) | NIRISS SOSS + NIRSpec G395H | **A** (log mass fraction -7.46 +1.96/-1.65; free-T run -7.14 +2.20/-1.87) | None (see below) | **A** (log mass fraction -6.58 +2.18/-2.20; free-T run -7.00 +2.27/-1.95) | None (not in model) |
| WD 1856 b (not in MCS) | MacDonald et al. 2026, Nature, arXiv 2607.01316, DOI 10.1038/s41586-026-10514-7 | Zenodo 10.5281/zenodo.18200586, `WD1856b_JWST_PRISM.zip` 18 MB, `JWST_Data/WD 1856b_planet_spectrum_{FIREFLy,Juniper}.dat` | NIRSpec PRISM, 0.5-5.0 um | **P** "2-20 %", i.e. log -1.70 to -0.70, level not stated | Not printed | Not printed | Not printed |

Notes on A1:

- **V1298 Tau c.** The two columns are the informed and uninformed stellar-heterogeneity retrievals (Aurora, table of atmospheric properties). Both agree on every label. HST agrees on H2O (Edwards 2023 -1.56 +0.32/-0.52). Barat et al. 2024 report only a metallicity and a 2.5 sigma water preference, so they add no VMR. **The three A labels are prior-shaped non-detections.** SOSS does not reach the 4.3 um CO2 band, and CO2's upper bound is only -4.16 in the uninformed run. I would report them, but they test little. The mapping of the paper's fiducial "Case 2" to `LCtest2_6spots` is my inference from the file names, so confirm it against the notebooks in the record. The planet is young and its host is active.
- **HAT-P-14 b.** The paper prints **log mass fraction**, not VMR, and fixes T at 1624 K. Converted with mu = 2.33 (my assumption; the paper does not print mu): CH4 upper bound -6.34 (free-T run -5.78), CO2 upper bound -5.68 (free-T run -6.01). H2O straddles once converted: VMR interval -4.60 to -2.08. In raw mass fraction it would be -3.72 to -1.19 and would read as present, so the unit matters. Clearing -4 would need mu of about 9, which this H2-dominated model does not support. H2O detection is 3.09 sigma, and the paper says a flat line is not excluded.
- **WD 1856 b.** Passes the file's requirements on one label. I would leave it out of any score: a grazing transit of a white dwarf, nightside thermal emission detected at 17-18 sigma across the 3.5-5 um spectrum, a planet of 4.3-10.9 M_J, and an abundance quoted only as a range with CH4 preferred at about 3 sigma. No model grid here covers it, and the classifier was not trained for it. Posterior samples for the other gases are in the deposit (`.npy`), which I did not use because labels come from papers.

**The MCS matters.** All three are outside `data/mcs/Ariel_MCS_Known_2026-05-11.csv`, which is why the earlier survey set aside HAT-P-14 b and V1298 Tau c. Using them means supplying planet and star parameters from their papers, a change to the test's stated sample.

### A2. Candidates without a public spectrum, rechecked

| Planet | Status on 2026-10-30 | Labels at stake | Contact as printed |
|---|---|---|---|
| HATS-75 b | **Still not public; verified.** Article page loaded once (AJ 171, 294, 10.3847/1538-3881/ae552c). It has only two ASCII tables (retrieval posteriors and light-curve fits), no figure data and no repository link, only the raw MAST DOI 10.17909/dtza-4154. Zenodo exact-name search: 0 records | 4 A | No address printed in the arXiv PDF, and none in the journal page HTML I read. First author Reza Ashtari, Johns Hopkins APL. The journal PDF was not read |
| TOI-5293 A b | **No deposit found; page unverified.** ApJ 1006, 119, 10.3847/1538-4357/ae801f. Zenodo: 0. Not in the Exoplanet Archive. 14 figure tarballs and 6 table files guessed on the journal content host: all 404. Article page behind a bot check | 1 A (NH3) | Corresponding author: Shubham Kanodia, skanodia@carnegiescience.edu (arXiv PDF) |
| WASP-178 b | **No deposit found; page unverified.** Lothringer et al. 2025, AJ 169, 274, 10.3847/1538-3881/adc117; Saha et al. 2025, arXiv 2510.11479, "under review", no DOI found. Zenodo holds only a CRIRES+ ground-based record (10.5281/zenodo.11637332). The same file guesses returned 404 | 1 A (CO2). **H2O is no longer A**: Edwards 2023 HST gives -3.01 +0.67/-0.95, present, against JWST-only -5.40 +/-0.82 | Saha: suman.saha@mail.udp.cl (printed as corresponding author in the arXiv source). Lothringer: no address found in the arXiv PDF |
| HD 106315 c | **No data statement or deposit.** Dyrek et al. 2026, arXiv 2609.09893, no DOI found. Zenodo: 0. HST spectra are public (Guilluy 2021, Kreidberg 2022 in the archive) and it counts in B for H2O | H2O P, CH4 A; NH3 P tentative | A. Dyrek, adyrek@stsci.edu (arXiv source) |
| V1298 Tau e (new) | **Not public.** Dai et al. 2026, arXiv 2606.00974: "Additional data products can be obtained from F.D. upon reasonable request"; raw data MAST, GTO 5882. Not in MCS | CH4 A -4.85 +0.54/-0.76 (5.8 sigma detection below threshold); H2O A -8.40 +2.59/-2.36; NH3 A -8.20 +2.38/-2.53; CO2 None -3.93 +0.71/-1.19. The paper cautions about non-detections | fdai@hawaii.edu, printed in the paper |
| WASP-69 b | No change from the earlier survey: limb-resolved only, no public spectrum | none | not checked |

Worth a request, if you want one: HATS-75 b (all four labels, in the MCS). I have written to no one.

### A3. Examined and not usable

- **Binet et al. 2026, arXiv 2609.05264, MIRI survey of 10 planets.** Outside the panel: WASP-43 b, L 168-9 b, GJ 367 b. Reports evidences for trace gases and, as far as I read, no VMR for the four molecules on them. Data cover only about 5-12 um.
- **LTT 9779 b (Stamer et al. 2026, 2609.36206), KELT-7 b, WASP-15 b, GJ 3090 b, TOI-3235 b and others in section (c)2 of the earlier survey.** No change; no labels fixed.
- **Papers on JWST targets with no atmospheric free retrieval:** Kepler-167 e (oblateness and exomoons), NGTS-10 A b (emission).
- **Fu et al. 2025 (WASP-127 b).** Forward fit across a sample, not free chemistry.
- **ExoComp library (2510.26785, Table 4).** Reports C/O and [M/H], not molecular VMRs. Its transit planets added no candidate.

### A4. Search

arXiv API, submitted 2025-01-01 onward, three abstract queries (JWST plus transit or transmission terms; instrument names plus the same; JWST plus retrieval or abundance terms), 483 unique records. Planet names extracted by regular expression, 132 not in the panel or batch 4, read by title and abstract. **This removes the earlier survey's restriction to MCS names, and is how V1298 Tau c and HAT-P-14 b surfaced**, though the regex needs a conventional name in the title or abstract. Cross-checked against the Exoplanet Archive `spectra` table (409 JWST rows, 65 planets with transmission rows, 32 outside the panel and batch 4), and Zenodo with exact-name queries.

---

## B. HST WFC3 G141 population studies

### B1. Sources

| Study | Planets | H2O values | Spectra in machine-readable form | Check |
|---|---|---|---|---|
| Edwards et al. 2023, ApJS 269, 31, arXiv 2211.00649, DOI 10.3847/1538-4365/ac9f1a | 70 | Free-chemistry table, median +/- | Exoplanet Archive: 28 of the 70 under 2023ApJS..269...31E. The other 42 are spectra from earlier papers, listed in the paper's Tables (T18 for most). No Zenodo record found; no MRT at the guessed journal paths | Archive: yes. Table read from arXiv source |
| Tsiaras et al. 2018, AJ 155, 156, arXiv 1704.05413, DOI 10.3847/1538-3881/aaaf75 | 30 | Table 3, maximum a posteriori +/- | `http://bit.ly/HSTDATA`, which redirects to a Dropbox folder serving `FINAL_ONLINE_DATA_4_REFS.zip`, **7.41 GB** (headers only, not downloaded). https://github.com/ucl-exoplanets resolves. www.ucl.ac.uk/exoplanets returned 403 | Link resolves. Internal structure not read |
| Fisher & Heng 2018, MNRAS 481, 4698, arXiv 1809.06894, DOI 10.1093/mnras/sty2550 | 38 fitted, 27 with a water value | Table, median +/- | Earlier reductions; provenance not checked per planet | Table read |
| Welbanks et al. 2019, ApJL 887, L20, arXiv 1912.04904, DOI 10.3847/2041-8213/ab5a89 | 19 | Table, median +/- | Literature HST data | Table read |
| Pinhas et al. 2019, MNRAS 482, 1485, arXiv 1811.00011, DOI 10.1093/mnras/sty2544 | 10 | Table, median +/- | Sing et al. 2016 survey data | Table read |

**Changeat et al. 2022 is the wrong paper for this.** ApJS 260, 3 (arXiv 2204.11729, DOI 10.3847/1538-4365/ac5cc2) is "Five key exoplanet questions answered via the analysis of 25 hot Jupiter atmospheres in **eclipse**". The transmission survey you want is Edwards et al. 2023. **The Exoplanet Archive lists 17 Changeat 2022 spectra as `Transmission`, which looks like a mislabel.** I could not open those files; do not use them as transit spectra without checking.

Direct URLs for individual archive spectra could not be built: every path I tried from `spec_path` returned 404. Use the table UI at https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres (HTTP 200) The page https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/ (HTTP 200) describes wget scripts; I did not confirm that they cover these spectra. The TAP table `spectra` returns metadata and file names, not the data.

### B2. Counts

Of the 71 planets across these tables, 51 are outside the panel and batch 4.

| Reading | Present | Absent | None |
|---|---|---|---|
| Edwards 2023 alone (50 planets) | 14 | 6 | 30 |
| Strict across all five studies, plus Skaf 2020 and Murphy 2025 (51 planets) | 7 | 6 | 38 |
| Same, plus JWST free-chemistry values already on file | **6** | **6** | 39 |

The JWST cross-check removes only WASP-178 b.

### B3. The 12 planets that fix a label

| Planet | MCS | Label | Printed log10 H2O | E23 sigma vs flat | Machine-readable spectra (Exoplanet Archive unless stated) |
|---|---|---|---|---|---|
| HAT-P-32 b | yes | P | E23 -1.18 +0.12/-0.32; T18 -2.84 +/-0.92; FH18 -2.39 +1.09/-1.57 | 7.35 | T18 zip; Alam 2020 (19 points); Damiano 2017 (20) |
| HAT-P-41 b | yes | P | E23 -2.55 +0.92/-1.36; T18 -2.77 +/-1.09; FH18 -0.90 +0.28/-1.20 | 4.72 | T18 zip; Sheppard 2021 (29) |
| HD 106315 c | yes | P | E23 -1.92 +0.58/-1.01; JWST+HST Dyrek 2026 -1.40 +0.40/-0.74 | 5.61 | Guilluy 2021 (25); Kreidberg 2022 (21). Kreidberg calls the water "tentative" |
| V1298 Tau c | no | P | E23 -1.56 +0.32/-0.52; JWST Murphy 2026 above | 14.53 | E23 (25); Barat 2024 (17) |
| WASP-62 b | yes | P | E23 -1.40 +0.23/-0.51; Skaf 2020 -2.03 +0.52/-1.27 | 5.81 | Skaf 2020 (25) |
| WASP-79 b | yes | P | E23 -2.05 +0.40/-0.60; Skaf 2020 -2.43 +0.57/-0.76 | 6.44 | Skaf 2020 (25); Sotzen 2020 (15) |
| GJ 436 b | no | A | E23 -6.89 +2.83/-2.86; T18 -6.74 +/-2.70. FH18: flat line preferred, no value | none reported (<2) | T18 zip; Knutson 2014 (28) |
| HAT-P-2 b | yes | A | E23 -7.23 +2.80/-2.83 | 2.87 | E23 (25); Jacobs 2025 (17) |
| HIP 41378 f | yes | A | E23 -8.80 +1.14/-1.77 | 16.77 | E23 (25); Alam 2022 (30) |
| K2-24 b | yes | A | E23 -7.83 +2.92/-2.64 | 2.46 | E23 (25) |
| LTT 9779 b | yes | A | E23 -8.23 +2.37/-2.33 | none reported (<2) | Bibcode 2023AJ....166..158E (38 points, 0.84-1.63 um, so G102 + G141) |
| WASP-103 b | yes | A | E23 -7.71 +2.86/-2.59 | 6.78 | E23 (25) |

### B4. How much these labels can bear

1. **Absent mostly means flat, not water-poor.** GJ 436 b and LTT 9779 b have no evidence for any atmospheric signal. HAT-P-2 b and K2-24 b are below 3 sigma. In HIP 41378 f and WASP-103 b the strong preference over a flat line is not attributed to water. A flat spectrum with a cloud deck can give this posterior for a planet with plenty of water. Scoring a classifier on these rewards "says absent when flat", which is a different thing from water detection.
2. **G141 covers about 1.1-1.7 um, one water band and nothing diagnostic of CO2.** That is under a tenth of the 0.5-7.8 um range. Any method that needs other bands cannot be tested here without an explicit coverage mask, and the result only speaks to H2O.
3. **The labels themselves are TauREx retrievals for E23 and T18** (Alfnoor wraps TauREx 3). The spectra are real, but if the classifier was trained on TauREx models, the labels share that forward model. Among the 12 usable planets, only HAT-P-32 b and HAT-P-41 b have a water value from a non-TauREx code (FH18, HELIOS-T), and it agrees. GJ 436 b has FH18's flat-line result. The other nine have no such check at population level, apart from JWST for V1298 Tau c and HD 106315 c.
4. **Same data, different labels.** E23 and T18 retrieve different values from the same T18 spectra (HAT-P-32 b -1.18 vs -2.84). That is the size of the systematic the rule has to absorb.
5. **The strict count covers population studies, not all single-planet papers.** I added Skaf 2020 and Murphy 2025 only. Damiano 2018, Alam 2020, Sheppard 2021, Wakeford 2020, Sotzen 2020, Rathcke 2021 and Guilluy 2021 could remove more labels. **Treat 12 as an upper bound.** Checking the seven takes about an hour.
6. **Table values are from arXiv sources**, not the journal versions. E23's Table is printed to two decimals, and three labels sit within 0.1 of the threshold: GJ 436 b (upper -4.04 to -4.06), HAT-P-32 b (FH18 lower -3.96), HAT-P-41 b (E23 lower -3.91).

### B5. Planets that fix no label

The 38 below are blocked by the HST studies alone. The 39th is WASP-178 b, present in E23 (-3.01 +0.67/-0.95) but blocked by the JWST-only -5.40 +/-0.82 in section A2. Fisher & Heng flat-line cases have no value and are omitted.

| Planet | Blocking and other values (median +up/-low, then P, A or x for crossing) |
|---|---|
| CoRoT-1 b | E23 -5.33+2.81/-3.81 x |
| HAT-P-11 b | E23 -1.94+0.68/-1.26 P ; T18 -1.76+1.41/-1.41 P ; FH18 -2.56+2.13/-2.20 x ; W19 -3.66+0.83/-0.57 x |
| HAT-P-17 b | E23 -6.00+3.41/-3.44 x ; T18 -5.86+2.89/-2.89 x ; FH18 -8.25+3.40/-2.91 A |
| HAT-P-38 b | E23 -3.07+1.12/-2.27 x ; T18 -4.29+2.16/-2.16 x ; FH18 -0.41+0.22/-1.80 P |
| HAT-P-3 b | E23 -6.72+3.08/-3.13 x ; T18 -6.93+2.73/-2.73 A ; FH18 -7.88+5.99/-3.47 x |
| HAT-P-7 b | E23 -6.48+3.40/-3.20 x |
| HD 149026 b | E23 -4.54+2.45/-4.12 x ; T18 -5.75+2.91/-2.91 x ; FH18 -4.69+4.11/-5.02 x |
| HD 219666 b | E23 -3.23+1.57/-4.21 x ; M25 -1.43+0.53/-1.29 P |
| HD 3167 c | E23 -3.73+1.07/-0.97 x |
| HD 97658 b | E23 -3.49+0.35/-0.53 x ; FH18 -7.48+4.88/-3.70 x |
| HIP 41378 b | E23 -4.03+2.34/-4.79 x |
| KELT-11 b | E23 -5.09+1.57/-0.87 x |
| KELT-1 b | E23 -6.63+3.23/-3.08 x |
| KELT-7 b | E23 -4.33+1.50/-4.78 x |
| Kepler-51 b | E23 -6.10+3.13/-3.32 x |
| Kepler-79 d | E23 -6.82+2.90/-3.04 x |
| Kepler-9 b | E23 -4.71+2.51/-4.19 x |
| Kepler-9 c | E23 -5.99+2.22/-3.25 x |
| TOI-270 c | E23 -3.73+1.56/-3.78 x |
| TOI-674 b | E23 -3.12+0.78/-1.04 x |
| TrES-2 b | E23 -6.47+3.24/-3.07 x |
| WASP-101 b | E23 -6.72+3.18/-3.07 x ; T18 -6.95+2.61/-2.61 A ; FH18 -9.03+3.13/-2.63 A |
| WASP-117 b | E23 -3.10+1.38/-2.85 x |
| WASP-127 b | E23 -1.14+0.08/-0.12 P ; W19 -2.13+0.65/-3.63 x ; S20 -2.71+0.78/-1.05 P |
| WASP-12 b | E23 -2.89+0.53/-0.63 P ; T18 -3.12+0.92/-0.92 x ; FH18 -3.02+1.09/-1.36 x ; W19 -3.23+1.42/-0.80 x ; P19 -3.16+0.66/-0.69 P |
| WASP-18 b | E23 -6.75+3.46/-3.27 x |
| WASP-19 b | E23 -1.88+0.52/-1.03 P ; FH18 -2.86+1.20/-1.49 x ; W19 -3.43+0.47/-0.52 P ; P19 -3.90+0.95/-1.16 x |
| WASP-29 b | E23 -6.51+2.95/-3.03 x ; T18 -7.93+2.38/-2.38 A |
| WASP-31 b | E23 -3.29+1.36/-3.44 x ; T18 -3.84+1.90/-1.90 x ; W19 -4.55+1.77/-4.33 x ; P19 -3.97+1.01/-2.27 x |
| WASP-33b | W19 -6.64+3.15/-3.25 x |
| WASP-43 b | E23 -3.14+1.41/-5.50 x ; T18 -4.36+2.10/-2.10 x ; FH18 -2.89+1.13/-3.07 x ; W19 -3.68+0.92/-0.88 x |
| WASP-63 b | E23 -3.49+1.48/-4.05 x ; T18 -5.81+2.81/-2.81 x ; FH18 -5.83+2.90/-4.23 x |
| WASP-67 b | E23 -3.60+1.51/-2.75 x ; T18 -6.17+2.82/-2.82 x |
| WASP-69 b | E23 -2.48+0.69/-0.94 P ; T18 -3.94+1.25/-1.25 x ; FH18 -4.24+1.03/-1.09 x |
| WASP-6 b | E23 -2.83+0.87/-1.14 P ; W19 -8.12+2.52/-2.41 A ; P19 -6.91+1.83/-2.07 A |
| WASP-74 b | E23 -3.88+2.13/-4.79 x ; T18 -5.91+2.81/-2.81 x ; FH18 -7.94+3.72/-3.64 A |
| WASP-76 b | E23 -2.96+0.84/-1.33 x ; T18 -2.70+1.07/-1.07 P ; FH18 -5.30+0.61/-0.61 A |
| XO-1 b | E23 -1.39+0.27/-0.69 P ; T18 -2.75+1.64/-1.64 x ; FH18 -1.06+0.29/-0.47 P |

---

## C. Public model grids from forward models other than TauREx

Checked 2026-10-30. "Resolves" means an HTTP 200 on the URL given.

### C1. Grids with spectra you can download

| Grid | URL and check | Models | Parameters | Per-model VMRs | Wavelength, resolution | Size |
|---|---|---|---|---|---|---|
| **Goyal et al. 2019 generic ATMO**, MNRAS 482, 4503, arXiv 1810.12971, DOI 10.1093/mnras/sty3001 | https://drive.google.com/drive/folders/1ZFbkPdqg37_Om7ECSspSpEp5QrUMfA9J resolves, four tars listed, one spectrum and one abundance file read. Mirror DOI 10.24378/exe.883 reaches the Exeter page, which returned a bot check (HTTP 202), not read. Viewer https://exoctk.stsci.edu/generic resolves. Scaling and interpolation code https://github.com/mdhimes/Generic_Grid resolves | 56,320 in the paper. The readme adds 11,264 (0.1x solar) and 6,144 (300, 500 K), so **73,728**. My sum; the tars were not counted | T 300-2600 K isothermal, g 5, 10, 20, 50 m/s2, metallicity 0.1-200x, C/O 0.35-1.0, haze 1, 10, 150, 1100, grey cloud 0, 0.06, 0.2, 1, local or rainout condensation. Equilibrium chemistry. Scales to any planet-star pair | **Yes.** `chem-iso-generic_*.txt.gz`: 51 pressure levels (1e-6 to 10 bar), mean molar mass, 258 species as mole fractions, column names in `Utilities/` | **0.2-1980 um**, 5000 points on a constant wavenumber grid. R about 2000 at 0.5 um, 714 at 1.4, 200 at 5, 127 at 7.8 (measured from one file). The last row is NaN, so check long wavelengths | Spectra 2.16 GB each scheme; abundances 2.35 GB (rainout), 2.56 GB (local). **9.2 GB in all** |
| **Crossfield et al. 2025, "SO2 Shoreline"**, arXiv 2509.14318, data DOI 10.5281/zenodo.17101615 | https://zenodo.org/records/17101615 resolves, tar head read. Journal DOI not found | **936** (as printed) | HELIOS T-P + **VULCAN photochemistry** + **petitRADTRANS** spectra. T_eq 250-2050 K (13), metallicity 0.3-1000x (8), one-at-a-time T_int 100, 300, 500 K, XUV 0.03, 1, 30, Kzz 1e5, 1e7, 1e9, C/O 0.3, 0.55, 0.8. **Single planet and star: HAT-P-26 b bulk parameters, g = 4.58 m/s2, no clouds or hazes** | **Yes, by altitude.** `mtgrid_helios_vertical_profiles.csv` has H2O, CO2, CH4, NH3 and 20 others; also MMR profiles | **0.40-24.98 um, R about 1000** (one file). Columns include `rprs2_only_X` and `rprs2_no_X` for each molecule | 8.03 GB tar, about 19 MB per model |
| Goyal et al. 2018 ATMO, planet-specific, MNRAS 474, 5158, arXiv 1710.10269, DOI 10.1093/mnras/stx3015 | https://drive.google.com/drive/folders/1tvlgSWyEAA0cX_yxnA5URD4Zo3Zj4GcH resolves: `Chemical_Abundances`, `Transmission_Spectra`, `Utilities`, readme. VizieR J/MNRAS/474/5158: **could not be reached from here** | About 460,000, from a secondary source, **not verified** | 117 named planets; isothermal; T, metallicity, C/O, haze, cloud | Yes (folder present; not opened) | **Not verified** | Not verified. Per-planet tars seen |
| Goyal et al. 2020 ATMO self-consistent, MNRAS 498, 4680, arXiv 2008.01856, DOI 10.1093/mnras/staa2300 | https://drive.google.com/drive/folders/1zCCe6HICuK2nLgnYJFal7W4lyunjU4JE resolves. The paper's second link (noctis.erc-atmo.eu) returned **404** | 144 per planet for 89 planets, up to 12,816, less 5-10 % not converged | Radiative-convective T-P, recirculation 0.25-1, metallicity 0.1-200x, C/O 0.35-1.5, local or rainout. No haze or cloud parameter in the abstract | Yes (`Chemical Abundances` folder; not opened) | R about 5000 at 0.2 um down to about 35 at 30 um, as printed | Not verified |
| Crossfield et al. 2023, ApJL, arXiv 2303.17622, DOI 10.3847/2041-8213/ace35f, data 10.5281/zenodo.7760360 | Resolves | 2,662 atmospheres, 3,993 spectra | **WASP-39 b only**. C, O, S each 1-100x; morning and evening | Yes | Not checked | 19.83 GB (spectra about 220 MB, profiles 19.6 GB) |
| PICASO 3.0 WASP-39 b, 10.5281/zenodo.7236759 | Resolves | Small grid, not counted | **WASP-39 b only**. Equilibrium and photochemical, cloudy and clear | Not checked | Not checked | 2.11 GB |
| Fortney et al. 2010, ExoCTK | https://exoctk.stsci.edu/fortney resolves | Not counted | T 500-2500 K, g 10, 25 m/s2, scaling factors (as Goyal 2019 describes it) | **No** | Not checked | Web viewer |

### C2. Asked about, but no usable grid

| Item | Finding | Link |
|---|---|---|
| petitCODE grid, Molaverdikhani et al. 2019, arXiv 1809.09629 | 28,224 cloud-free self-consistent models, and later Kzz and cloud extensions. **Paper's link is dead**: www.mpia.de/homes/karan returned 404. Would need the author | 404 |
| POSEIDON | Code only. No spectrum grid found. Input files 72.1 GB | https://github.com/MartianColonist/POSEIDON resolves; 10.5281/zenodo.19595136 |
| Exo-Transmit | Code with bundled equation-of-state tables (from memory, not rechecked). No spectrum grid found | https://github.com/elizakempton/Exo_Transmit resolves |
| VULCAN | Code. Grid outputs come through the Crossfield records above | https://github.com/exoclime/VULCAN redirects to github.com/shami-EEG/VULCAN |
| petitRADTRANS | Code plus an equilibrium abundance table only (10.5281/zenodo.15654694, 1.64 GB). No spectrum grid | https://petitradtrans.readthedocs.io/ resolves |
| PICASO, Sonora | PICASO is code (https://github.com/natashabatalha/picaso resolves). Sonora and the related Zenodo records (e.g. 6842801, 7076995) are brown dwarf emission, not transmission | |
| Kawashima & Min 2021, 10.5281/zenodo.5722862, 6.53 GB | Retrieval output for 16 planets, not a forward grid | Resolves |
| Baeyens et al. 2021, arXiv 2105.02245 | Pseudo-2D chemistry grid with transmission spectra in the paper. No deposit found in the source grep; not pursued | |
| ATMO `atmo_xarray_grid`, 10.5281/zenodo.7809066 | Emission spectra for self-luminous objects. Wrong geometry | Resolves |

TauREx-based sets (Ariel data challenge, ABC) are left out as you asked.

### C3. Recommendation

**Use Goyal et al. 2019 as the main test, and Crossfield et al. 2025 as a second, separately reported test.**

| | Goyal 2019 generic | Crossfield 2025 | Generate your own with POSEIDON or petitRADTRANS |
|---|---|---|---|
| For | Spans T, g, metallicity, C/O and, critically, **haze and cloud**. Tens of thousands of models. VMRs per model. Covers 0.5-7.8 um at more than Ariel's resolution. Scales to any planet. Different code, opacities and aerosol scheme from TauREx | **Real photochemistry and quenching**, so CH4, NH3, CO2 ratios unlike equilibrium. Per-molecule on and off spectra give a direct feature test. Altitude-resolved VMRs | Sample exactly what the test needs, including priors matching training |
| Against | **Equilibrium chemistry and isothermal only.** Abundances are tied to T, so a label partly encodes temperature; haze and cloud are simple opacity factors. VMRs vary with pressure, so a label needs a stated pressure window | **No aerosols, one planet and star.** Only 936 models, one-at-a-time parameter changes. Only H, C, N, O, S species plus scaled Na and K, so no TiO, VO or metals at the hot end | Your choice of settings becomes part of the result, which weakens the "independent" claim; opacity downloads and runtime |

Why Goyal first: the question is whether the method survives a forward model it has not seen, and aerosol muting looks like the hard case here (I am going by script names such as `jwst_muting_census.py`, not results). It is the only multi-planet public grid I could confirm with both aerosols and VMRs. Crossfield then asks whether it survives disequilibrium chemistry, which Goyal cannot.

Three things to fix **before** looking at any scores:

1. The pressure window for the label. Goyal 2019 itself averages abundances over 0.1-100 mbar; that is a defensible default. Use the same window for both grids.
2. Both condensation schemes or one. My preference is rainout as primary and local as a sensitivity test. The 2018 planet-specific library had a rainout error with a published erratum; I did not check whether the generic grid's files are affected, so read the erratum first.
3. How to sample. Equal weight across 73,728 grid points is not the population the classifier was trained on, and a temperature-stratified score would be more honest. A 300-2600 K grid with fixed scaling factors will not have the same prior as POP-I.

Practical note: the Drive tars include macOS `._` files alongside each model. Skip them on extraction. Google shows a "too large to scan" notice before each download.

---

## Verification status

| Item | Status |
|---|---|
| Zenodo records 20547576, 18200586, 17101615, 7760360, 5722862, 7236759, 7809066, 15654694 | Read through the API; files listed |
| Goyal 2019 Drive tars | Sizes from response headers; one spectrum and one abundance file read from the archive head |
| Goyal 2018 and 2020 Drive folders | Folder listing only |
| T18 Dropbox zip | Headers only (7,411,949,460 bytes) |
| All 28 DOIs in this file | Each redirects at doi.org (checked 2026-10-30). The target of 10.24378/exe.883 is a bot-check page, and IOP and OUP targets may be too |
| Journal article pages | **IOP served a bot check after the first request. I did not try to get past it.** So only HATS-75 b's page was read; TOI-5293 A b and WASP-178 b rest on negative searches. My very first connectivity test sent a generic browser user-agent to six URLs, including that DOI; every later request, including the page read, used the default client |
| VizieR and CDS | **All hosts unreachable from this machine**, so catalogue copies of T18 and Goyal 2018 are unverified |
| Abundances | From arXiv LaTeX sources, not journal versions. WD 1856 b, HAT-P-14 b and V1298 Tau c were not compared with the published text |
| WD 1856 b CH4 | The "2-20 %" is from the main text; the credible level is not stated |
| Project overlap | A name-only grep of v3 outside `results/`, `data/` and `exosim_payload/` for Tsiaras, WFC3, G141, Goyal, Crossfield found **no script**, only five documents (including `jhss/manuscript.md` and `REVISION_PLAN.md`), which I did not read. I cannot rule out earlier use of these data; you can |
