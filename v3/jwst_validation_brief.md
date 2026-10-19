# Literature briefs for the external-validation work package (compiled 2026-10-15)

Two web-literature surveys run on 2026-10-15 to support `REVISION_PLAN.md`, Work package A.
They are secondary notes: every number must be checked against the cited paper before it is
used in the manuscript, and items marked "unverified" or "could not be traced" are not to be
cited at all. Part 1 covers WASP-39 b; Part 2 covers the rest of the JWST transmission-spectrum
literature as a validation panel.

---

## Part 1. WASP-39 b

### 1. Published JWST transmission spectra

All ERS data are from JWST-ERS-1366 (raw data on MAST). Reduced products are on Zenodo
(CC BY 4.0) in the "JWST Transiting Exoplanet Community ERS" community.

| Dataset | Paper | Range / native R | Final binning in paper | Typical precision | Reductions | Archive |
|---|---|---|---|---|---|---|
| NIRISS SOSS | Feinstein et al. 2023, Nature 614, 670 (arXiv:2211.10493) | 0.6–2.8 µm; order 1 R≈700, order 2 R≈1400 | 100 channels (80 order 1 + 20 order 2), R≈300 | Carter+24 quote σ_mean≈310 ppm at native R≈350–1390 | nirHiss, supreme-SPOON, transitspectroscopy | Zenodo 10.5281/zenodo.7709712; GitHub afeinstein20/wasp39b_niriss_paper |
| NIRCam F322W2 (+F210M phot.) | Ahrer et al. 2023, Nature 614, 653 (arXiv:2211.10489) | 2.0–4.0 µm (F322W2 ≈2.4–4.0 spectroscopy; F210M photometric point) | 15 nm bins | median 135 ppm (1.18× photon); F210M 53 ppm | Eureka! (chosen), tshirt, HANSOLO, chromatic-fitting | Zenodo 10.5281/zenodo.7101283; notebook 10.5281/zenodo.7510106 |
| NIRSpec G395H | Alderson et al. 2023, Nature 614, 664 (arXiv:2211.10488) | 3–5 µm, R≈600 | ≈10-pixel bins (344 points) | mean 221 ppm/bin (131–625), 1.46× photon | ExoTiC-JEDI (nominal), Eureka!, Tiberius; 11 combinations averaged | Zenodo 10.5281/zenodo.7185300 |
| NIRSpec PRISM | Rustamkulov et al. 2023, Nature 614, 659 (arXiv:2211.10487) | 0.5–5.5 µm, native R≈20–300; **detector saturated 0.8–2.3 µm** | 207 variable-width channels | 50–200 ppm per channel | FIREFLy (baseline), Tiberius, Eureka!, tshirt | Zenodo 10.5281/zenodo.7388032 |
| MIRI LRS | Powell et al. 2024, Nature 626, 979 (doi 10.1038/s41586-024-07040-9) | 5–12 µm, R≈100; observed 2023-02-14 | 28 bins × 0.25 µm | light-curve scatter ≈890–4300 ppm per bin; transit-depth errors at the several-hundred-ppm level | Eureka! (fiducial), Tiberius, SPARTA | Zenodo 10.5281/zenodo.10055845 |
| **Joint NIR re-analysis (benchmark)** | Carter & May et al. 2024, Nat. Astron. 8, 1008 (arXiv:2407.13893) | 0.518–5.348 µm, four modes | native-resolution light curves (2,888) and R = 100 products | σ_mean at native R: SOSS 310, F322W2 294, G395H 496, PRISM 108 ppm; inter-mode offsets reduced to ≈140 ppm | uniform Eureka! light-curve fitting on supreme-SPOON (SOSS), Eureka! (NIRCam), ExoTiC-JEDI V2 (G395H), FIREFLy (PRISM) | **Zenodo 10.5281/zenodo.10161743** (1.7 GB; time-series spectra, light-curve fits, transmission spectra, model) |

Caveats: Carter+24 flag the PRISM saturated region (0.8–2.3 µm) as unreliable and find a
≈−177 ppm PRISM offset. Roy-Perez et al. 2026 (A&A, arXiv:2602.06722) show retrieved
abundances vary by up to ≈2 dex among six PRISM reductions. Later re-analyses of the same
data: Ma et al. 2025 (ApJS, arXiv:2504.07823; TauREx+YunMa, 0.3–12 µm, favours SiO over SO₂
at 4.1 µm, silicate clouds for MIRI); Deka et al. 2025 (ApJ, NEXOTRANS, arXiv:2504.18815);
Espinoza et al. 2024 (Nature 632, 1017; morning/evening limb spectra from PRISM 2–5 µm,
evening 405 ± 88 ppm deeper); Lueber et al. 2024 (A&A, arXiv:2405.02656); "Knobs and dials II"
(arXiv:2511.05652, PRISM pipeline effects); Binet et al. 2026 (ApJL, arXiv:2609.05264;
homogeneous MIRI survey). A September-2026 NIRISS SOSS nonlinearity-corrected re-reduction was
hinted at in search results but could not be verified. No new JWST transits of WASP-39 b
beyond ERS-1366 were found.

### 2. Retrieved composition

Detections: H₂O 33σ (PRISM), 21.5σ (G395H); CO₂ 28σ (PRISM), 28.5σ (G395H); CO 7σ (PRISM);
Na 19σ (PRISM); K at 0.76 µm (NIRISS; confirmed in Carter+24); SO₂ 2.7σ (PRISM), 4.8σ (G395H),
and 7.7/8.5 µm bands in MIRI; grey cloud 21σ (PRISM). H₂S tentative only (2–4σ, Constantinou,
Madhusudhan & Gandhi 2023, ApJL 943, L10, arXiv:2301.02564).

Abundances: SO₂ log VMR −6.3 to −4.6 (MIRI free retrievals, Powell+24); H₂O log VMR −2.4 to
−1.2 (MIRI); Constantinou+23 (PRISM 3–5 µm): H₂O −4.9 to −3.3, CO₂ −6.3 to −4.5, CO −4.3 to
−2.6, SO₂ −7.0 to −5.9, H₂S −5.3 to −4.0 depending on reduction (Tiberius vs Eureka!), a ≈1 dex
pipeline dependence. Ma+25 free retrieval: H₂O −1.6, CO₂ −3.1, SiO −1.8, Na −3.0 (model-dependent).

CH₄: no detection anywhere. Limits: VMR ≲ 5 × 10⁻⁶ at 0.1–2 mbar (Rustamkulov+23, PRISM);
≤ 5.5 × 10⁻⁵ at 1 mbar (Ahrer+23, NIRCam); best-fit grid models carry ppb-level CH₄;
Feinstein+23: absent 1.7/2.3 µm bands rule out C/O ≳ 0.7. One 2025 cloud study gives log CH₄
≈ −5.85 under one cloud model (arXiv:2501.17728).

NH₃: in retrievals (Powell+24; Constantinou+23; Alderson+23 grids) but never detected; no
formal published upper limit found. Forestano et al. 2025 quote "NH₃ < −6 (log)" citing
Alderson+23, which could not be traced in that paper: **unverified**. Powell+24 state CH₄, NH₃,
HCN, CO, CO₂, C₂H₂, H₂S "showed no significant detections" in MIRI.

Metallicity / C/O: NIRISS 10–30× solar, sub-solar C/O; G395H 3–10×; NIRCam 1–100×, C/O best
fits 0.23–0.35; PRISM ≈10×, C/O ≈ 0.6–0.7 best fit; MIRI/SO₂ photochemistry 7.1–8.0× solar
(Powell+24), ≈10× (Tsai+23); Carter+24 benchmark model: 10× solar, C/O = 0.35.

Clouds/hazes: grey cloud strongly preferred (PRISM, 21σ); NIRISS prefers non-grey, patchy
clouds; Roy-Perez+26: τ = 1 at a few hundred mbar, particle sizes 0.03–3.5 µm depending on
reduction; silicate clouds invoked for MIRI.

Photochemistry: SO₂ from H₂S oxidation (Tsai et al. 2023, Nature 617, 483; arXiv:2211.10490);
the 4.05 µm feature requires VMR 10⁻⁶–10⁻⁵.

### 3. System parameters

Planet: R_p = 1.279 ± 0.040 R_J; M_p = 0.281 ± 0.032 M_J; T_eq = 1166 ± 14 K (NEA default;
other solutions 1076–1120 K); P = 4.0552842 d, a/R* = 11.390 ± 0.012, i = 87.7369°, b = 0.4498,
ρ* = 1.6999 g cm⁻³ (Carter+24 Table 1); transit depth ≈2.1 %.
Star (WASP-39, G8V): T_eff 5400 ± 150 K (also 5485 ± 50 and 5327 in other solutions), log g
4.39–4.48, [Fe/H] −0.12 ± 0.10 or +0.01 ± 0.07, R* = 0.895–1.01 R☉ (0.932 used by Powell+24),
M* = 0.93 ± 0.03 M☉; distance 213.98 pc (Gaia); V = 12.10, J = 10.663, H = 10.307, K_s = 10.202.
Source: NASA Exoplanet Archive overview page. Recommendation: Carter+24 orbital/density values,
Mancini+18 radii.

Ariel status: Edwards et al. 2019 (AJ 157, 242) Table 6 lists WASP-39 b with "maximum
preferred tier 2, Transit"; Edwards & Tinetti 2022 (AJ 164, 15) Table 3 lists "WASP-39 b, Tier
3, Transit"; Radica et al. 2026 (arXiv:2604.07598) use it as the hot-Saturn benchmark and note
one Ariel transit gives "Tier 1.5" S/N, two transits Tier 2.

### 4. Prior ML work on the real spectrum, and Ariel-format conversion

- **Forestano, Matchev, Matcheva & Unlu 2025 (arXiv:2508.04982)**: the closest precedent.
  Regressors (PLS, SVM, KNN, RF, XGBoost, stacking) trained on the Ariel Big Challenge TauREx3
  database (105,887 noise-free spectra, 52 Ariel channels 0.55–7.275 µm) predicting T and
  log VMR of H₂O, CO₂, CH₄, CO, NH₃; applied to Rustamkulov+23 PRISM + Powell+24 MIRI rebinned
  to the 52 channels with TauREx FluxBinner. XGB predictions: H₂O −6.68, CO₂ −4.50, CH₄ −7.98,
  CO −2.80, NH₃ −10.36. Regression, not classification; no Ariel-noise injection.
- Lueber et al. 2024 (A&A 687; arXiv:2405.02656): HELA random forest vs nested sampling on
  G395H and PRISM.
- Deka et al. 2025 NEXOTRANS: ML regressors alongside UltraNest on NIRISS + PRISM + MIRI.
- **Chakravertty & Raj 2026 (arXiv:2610.02245)**: amortized flow posterior trained on
  simulations collapsed on real WASP-39 b (χ²/N = 301) until planet radius was added as a
  latent ("MIRAGE").
- Exoformer (A&A 2026): transformer seeding TauREx nested sampling, tested on WASP-39 b and
  WASP-17 b.
- No paper found that publishes an Ariel Tier-1/2 binned JWST WASP-39 b spectrum other than
  Forestano+25's 52-channel rebin. Changeat et al. 2025/26 (Ariel–JWST Synergy WG,
  arXiv:2509.02657) discuss strategy, not binning. "The effect of spectroscopic binning on
  atmospheric retrievals" (MNRAS 2024, arXiv:2407.09296) for binning effects in general.
- Ariel Tier-1 identifiability: Mugnai et al. 2021 (AJ 162, 288) §4.3 detection limit, H₂O–NH₃
  degeneracy; Bocchieri, Mugnai et al. 2023 (Exp. Astron., arXiv:2309.06817): P-statistic on
  ≈300 Tier-1 planets, AUC ≈ 0.91–0.93 for CH₄/H₂O/CO₂, Tier-1 detection limits ≈10⁻⁴ VMR,
  NH₃–H₂O/CH₄ and CO–CO₂ overlaps as main degeneracies; Radica et al. 2026: for a WASP-39 b
  analogue, H₂O and CO₂ ≲ 1 dex at Tier 1, CO needs Tier 2.5–3, SO₂ undetectable even at Tier 3.

### 5. Ariel channels and tier binning

VISPhot 0.5–0.6, FGS1 0.6–0.80(0.81), FGS2 0.80(0.81)–1.1 µm; NIRSpec 1.1–1.95 µm (R ≥ 15,
Edwards+19 lists 20); AIRS-CH0 1.95–3.9 µm (R ≥ 100); AIRS-CH1 3.9–7.8 µm (R ≥ 30).
Tier binning (Edwards+19 Table 2): Tier 1 R ≈ 1/3/1; Tier 2 R ≈ 10/50/10 (Red Book: 10/50/15);
Tier 3 R ≈ 20/100/30 (Radica+26: native). Tier 1 is S/N ≥ 7 after binning to 1 NIRSpec +
2 AIRS-CH0 + 1 AIRS-CH1 points plus the three photometric points. Tier 1 ≈ 1,000 planets in
≤ 5 transits, Tier 2 ≈ 500–600 (≤ 20), Tier 3 ≈ 50 benchmark planets × 5 visits.

---

## Part 2. A JWST transmission-spectrum panel (excluding WASP-39 b)

Threshold convention: ">1e-4" = log₁₀ VMR > −4 at ≥ 1σ; "<1e-4" = upper limit or posterior
below −4; "amb." = posterior straddles −4 or sources disagree. Ariel membership checked
against `Ariel_MCS_Known_2026-05-11.csv` (arielmission-space/Mission_Candidate_Sample,
977 planets).

### Full coverage (≈0.6–5 µm)

| Planet (type, T_eq, host) | JWST modes / coverage | Papers | CH₄ | H₂O | CO₂ | NH₃ | Clouds / haze / stellar | Data public | Ariel |
|---|---|---|---|---|---|---|---|---|---|
| TOI-270 d (sub-Neptune, ≈350 K, M3V) | SOSS 0.6–2.8 + G395H 2.7–5.2 | Benneke+24 arXiv:2403.03325; Holmberg & Madhusudhan 24 arXiv:2403.03244 | −1.64 +0.38/−0.36, 9.4σ → >1e-4 | −1.10 +0.31/−0.92, 2.5σ (amb.) | −1.67 +0.40/−0.60, 4.8σ → >1e-4 | < −4.27 (2σ) → <1e-4 | largely clear; haze only < 1.1 µm; no spot crossings | MAST 10.17909/dvqh-2r48; no Zenodo spectrum found | yes |
| K2-18 b (sub-Neptune, ≈255 K, M2.5V) | SOSS 0.85–2.85 + G395H 2.9–5.2; G235H 1.7–3.1 (Hu+25); MIRI 6–12 | Madhusudhan+23 arXiv:2309.05566; Schmidt & MacDonald+25 arXiv:2501.18477; Hu+25 arXiv:2507.12622; Madhusudhan+25 arXiv:2504.12267; Luque+25 arXiv:2505.13407 | −1.74 (M23); −2.14 to −0.53 (S25); −1.06, 10.8σ (Hu25) → robust >1e-4 | < −3.06 (M23); < −3.93 (Hu25) → <1e-4 | disputed: −2.09 (M23, 2.9σ); not significant (S25); −3.35, 3.7–4.7σ (Hu25) → amb. | < −4.51 (M23); < −5.59 (Hu25) → <1e-4 | grey cloud + haze ≈3σ; spot crossings modelled; spot/haze degeneracy (Hu+25) | OSF 10.17605/OSF.IO/36DJH; MAST 10.17909/3ds1-8z15 | yes |
| TOI-1231 b (temperate Neptune, ≈300 K, M3V) | SOSS + G395H, 0.65–5.2 | Sarkar, Madhusudhan+26 arXiv:2607.00973 | −0.85 ± 0.2, ln B 55–70 → >1e-4 | < −2.24 (95 %) → amb. | −1.90 +0.40/−0.71, ln B 3–6.6 → >1e-4 | < −3.34 (95 %) → amb. | clouds + haze (ln B 7.9); 2–3 spot crossings per visit | OSF osf.io/mryd6; MAST 10.17909/djm8-jm79 | yes |
| TOI-421 b (hot sub-Neptune, ≈920 K, late G) | SOSS + G395M, 0.83–5.0 | Davenport+25 arXiv:2501.01498 | −8.4 → <1e-4 | −3.81 +1.32/−0.70 → amb. | −9.1 → <1e-4 | not reported | haze-free; cloud deck disfavoured above 10 mbar; quiet star | Zenodo promised on acceptance | yes |
| WASP-166 b (hot super-Neptune, 1270 K, F9V) | SOSS 0.85–2.81 + G395M 2.8–5.17 | Mayo+25 arXiv:2501.00609 | −7.9 ± 1.3 → <1e-4 | −1.30, 15σ → >1e-4 | −2.23, 14.7σ → >1e-4 | −4.00 +0.33/−0.79, 2.3σ → amb. | cloud deck 2.6σ at ≈13 mbar | Zenodo 10.5281/zenodo.14503925 | yes |
| HAT-P-18 b (warm Saturn, 850 K, K2V) | SOSS 0.6–2.8 (+HST, Spitzer) | Fu+22 arXiv:2211.13761; Fournier-Tondreau+24 arXiv:2310.14950 | < −6.1 (2σ) → <1e-4 (Fu+22 3.2σ "hint" attributed to spots) | −4.4 ± 0.3, 12.5σ → amb. | −4.8 ± 0.4 → <1e-4 | < −6.0 → <1e-4 | grey cloud 7.4σ; unocculted spots 5.8σ (f ≈ 0.10, ΔT ≈ 300 K) drive the optical slope; occulted spot crossing | not confirmed | yes |
| HAT-P-14 b (hot Jupiter, ≈1570 K, F5V) | SOSS + G395H, 0.60–5.14 | Espinoza+23 arXiv:2211.01459; Liu+25 arXiv:2504.08903 | nondetection | 3.1σ, no VMR | — | — | grey cloud 1.9σ; small scale height | not confirmed | **no** |
| LTT 9779 b (ultra-hot Neptune, ≈2000 K, G7V) | SOSS + G395H, 0.63–5.17 | Radica+24 arXiv:2401.15548; Stamer+26 arXiv:2609.36206 | unconstrained | present, muted | — | — | clouds at mbar; cloud–metallicity degeneracy | not confirmed | yes |
| KELT-7 b (UHJ, 2050 K, F2V) | G395H + SOSS | arXiv:2509.12479; Schmidt+26 arXiv:2607.06708 | — | −7.05 (tentative) | ≈ −8.5 (tentative) | — | high cloud vs very low metallicity | Zenodo 10.5281/zenodo.17092001 | yes |
| Kepler-51 d (super-puff, ≈350 K, G) | PRISM 0.6–5.3 | Libby-Roberts+25/26 arXiv:2505.21358 | 2.2σ at ppb → <1e-4 | — | — | — | near-featureless; high haze or ring | not confirmed | yes |
| LP 791-18 c (sub-Neptune, 355 K, M6) | PRISM 0.7–5.4 | Roy+25 arXiv:2512.10876 | ≈4.5σ, likely >1e-4 | low | < 1 % | — | haze-dominated (5.4σ), 246–415× solar | not confirmed | yes |
| TOI-732 c / LTT 3780 c (sub-Neptune, ≈305–350 K, M3.4V) | NIRISS + G395H + MIRI, 0.9–12 | Rigby+25 arXiv:2512.15844 | −2.34, ≈3.2σ (degenerate) → >1e-4 (low confidence) | < −1.78 | < −2.56 | < −4.44 → <1e-4 | clouds/hazes ≈4σ; spot crossings | not confirmed | yes |
| LHS 1140 b (HZ super-Earth, ≈230 K, M4.5V) | SOSS (Cadieux+24 arXiv:2406.15136) + G395H (Damiano+24 arXiv:2403.13265) | — | — | — | — | — | H₂-rich rejected > 10σ; faculae 5.8σ, ≈20 % coverage | — | yes |

### Partial coverage (mainly 2.4–5 µm, some with MIRI)

| Planet | JWST modes / coverage | Papers | CH₄ | H₂O | CO₂ | NH₃ | Clouds / stellar | Data | Ariel |
|---|---|---|---|---|---|---|---|---|---|
| WASP-80 b (warm Jupiter, 825 K, K7–M0V) | NIRCam F322W2+F444W 2.4–4.0 (Bell+23); 2.4–10 with MIRI (Triantafillides+26) | Bell+23 arXiv:2309.04042; Triantafillides+26 arXiv:2604.13168 | −4.2 +0.5/−0.7, 6.1σ → amb./borderline; 2026: "very large" (VMR not accessed) | −1.8, 4.6σ → >1e-4 | < −3.09 (Bell); "evidence" 2026 → amb. | < −3.12 (Bell); "evidence" 2026 with CS₂ → amb. | thick patchy cloud; ≈3 % spot filling | MAST (GTO 1185); Zenodo 8264964 (configs only) | yes |
| WASP-107 b (warm super-Neptune, ≈740 K, K6V) | HST 0.9–1.6 + NIRCam 2.45–4.97 + MIRI 5–12 (Welbanks); G395H (Sing); MIRI (Dyrek) | Welbanks+24 arXiv:2405.11018; Sing+24 arXiv:2405.11027; Dyrek+24 arXiv:2311.12515 | −5.8 ± 0.2 (5σ); ≈1 ppm (Sing); none (Dyrek) → <1e-4 | −2.1, 21σ → >1e-4 | −3.9 / −4.4 (29σ) → amb. | −5.0 ± 0.2 (6σ, first NH₃ detection) → <1e-4 | silicate clouds (≈7σ), haze slope | Zenodo 10.5281/zenodo.10780448; 10.5281/zenodo.10891400 | yes |
| GJ 3470 b (warm sub-Neptune, 600 K, M1.5V) | NIRCam 2.45–4.97 (+HST, Spitzer) | Beatty+24 arXiv:2406.04450 | −4.05 +0.25/−0.27, 3.8σ → amb. | −1.08, 6.3σ → >1e-4 | −2.47, 7.3σ → >1e-4 | −7.3, unconstrained → <1e-4 | patchy clouds/haze; SO₂ 4σ | paper table | yes |
| HD 189733 b (hot Jupiter, 1200 K, K1–K2V) | NIRCam 2.4–5.0 | Fu+24 arXiv:2407.06163; Zhang+25 AJ 169, 38 | < −7.9 (99 %) → <1e-4 | −3.69 +0.76/−0.46, 13σ → amb. | −6.24, 11σ → <1e-4 | < −5.3 → <1e-4 | spot crossing excised; unocculted dimming 1–2.7 % corrected | Zenodo 11459715 | yes |
| HD 209458 b (hot Jupiter, ≈1450 K, G0V) | NIRCam 2.3–5.1 | Xue+24 arXiv:2310.03245; Verma+25 arXiv:2505.04413; arXiv:2506.16232 | < −5.6 → <1e-4 | NIRCam-only −1.6, drops ≈3 dex with HST (amb.) | NIRCam-only −5.3 (amb. with HST) | — | clouds 3.6σ | — | yes |
| GJ 1214 b (sub-Neptune, ≈600 K, M4.5V) | G395H (+MIRI, HST) | Schlawin+24 arXiv:2410.10183; Ohno+25 arXiv:2410.10186 | 2.0σ, haze-degenerate → amb. | — | 2.4–3.6σ; [M/H] ≈ 3.5 | — | extreme haze | MAST 10.17909/tqa7-th94 | yes |
| HAT-P-26 b (warm Neptune, ≈1000 K, K1V) | G395H (SOSS in prep) | Gressier+25 arXiv:2509.16082 | < −5.49 (3σ) → <1e-4 | −2.07 (>3σ) → >1e-4 | −2.91 (>5σ) → >1e-4 | < −4.20 (3σ) → <1e-4 | grey cloud ≈10 mbar; SO₂ −4.4 (>5σ) | MAST 10.17909/c23k-dn78 | yes |
| WASP-52 b (hot Saturn, 1300 K, K2V, active) | SOSS (Fournier-Tondreau+25 arXiv:2412.17072) + G395H (Pan+26 arXiv:2609.21810) | — | — | −3.41 ± 1.04 → amb. | −5.09 (>10σ) → <1e-4 | — | two spot crossings; unocculted spots 3–4σ; patchy clouds | not confirmed | yes |
| WASP-17 b (hot Jupiter, ≈1740 K, F6V) | SOSS (Louie+24 arXiv:2412.03675); MIRI (Grant+23 arXiv:2310.08637) | — | — | −2.96 → >1e-4 | — | — | quartz clouds; flat optical slope; H⁻ | Zenodo 14193061 | yes |
| HAT-P-1 b, WASP-15 b, TrES-4 b, WASP-121 b, HIP 67522 b, TOI-836 c, TOI-776 b/c, L 98-59 d, GJ 9827 d | G395H only (GJ 9827 d also SOSS) | see arXiv:2608.28538, 2410.08116, 2503.24280, 2506.02199, 2409.16355, 2404.01264, 2501.14596, 2502.20501, 2408.15707, 2410.03527 | — | various | various | — | — | — | yes |

No published JWST transmission spectrum as of 2026-10-15: WASP-69 b, WASP-77 Ab, WASP-19 b
(emission only), GJ 436 b (emission; not in the 2026 MCS), WASP-127 b (G395H transit taken,
star partly off-slit, analysis in prep).

### Disagreements to keep in mind

- K2-18 b CO₂: 2.9σ (Madhusudhan+23) vs no reliable CO₂ (Schmidt & MacDonald+25) vs 3.7–4.7σ
  (Hu+25). CH₄ is the only consensus species. The DMS/DMDS claim is contested.
- WASP-107 b CH₄: ≈1 ppm in two papers, non-detection in MIRI; CO₂ straddles 10⁻⁴ across codes.
- HAT-P-18 b CH₄: a 3.2σ hint (Fu+22) vanished once unocculted spots were modelled
  (Fournier-Tondreau+24): spot contamination mimicking a molecular feature.
- HD 209458 b: NIRCam-only retrievals ≈3 dex higher than NIRCam + HST (Verma+25): the risk of
  partial coverage.
- WASP-80 b CH₄: −4.2 +0.5/−0.7 is centred just below 10⁻⁴.

### Stellar-contamination flags

Significant: HAT-P-18 b, WASP-52 b, K2-18 b, TOI-1231 b, HD 189733 b, LHS 1140 b, L 98-59 d.
Clean: TOI-270 d, TOI-421 b, GJ 3470 b, WASP-80 b, HIP 67522 b, WASP-166 b.

### Summary at the 10⁻⁴ threshold

(a) Confident CH₄ > 10⁻⁴: TOI-270 d (best: full coverage, clean star), K2-18 b, TOI-1231 b;
weaker: LP 791-18 c, TOI-732 c; borderline: WASP-80 b, GJ 3470 b.
(b) Confident CH₄ < 10⁻⁴: HD 189733 b, HD 209458 b, HAT-P-26 b, HAT-P-18 b, WASP-107 b,
WASP-166 b, TOI-421 b.
(c) NH₃ > 10⁻⁴: none in the JWST literature. The only detections (WASP-107 b at −5.0; a 2026
claim for WASP-80 b, abundance unverified) are below 10⁻⁴. NH₃ can be validated on negatives
only.
(d) CO₂ > 10⁻⁴: TOI-270 d, TOI-1231 b, WASP-166 b, GJ 3470 b, HAT-P-26 b, HIP 67522 b;
ambiguous: K2-18 b, WASP-107 b; negatives: HD 189733 b, HAT-P-18 b, WASP-52 b, TOI-421 b.

---

## Part 3. Data inventory: the files actually used, and where each came from (2026-10-16)

`fetch_jwst_data.sh` rebuilds `v3/data/jwst/` from these sources and checks every file against
`v3/jwst_manifest.md5`. Zenodo archives are fetched with a resume loop (Zenodo drops large transfers
every few MB). Files marked DERIVED were converted from the archive's own products by a script named
here and are not used by the pipeline unless stated.

### WASP-39 b (`v3/data/jwst/wasp39/<record>/extracted/`)

| Record | File in record | Members used | Reader |
| :-- | :-- | :-- | :-- |
| 7709712 (Feinstein 2023, NIRISS) | zenodo_upload.zip | 4_TRANSMISSION_SPECTRA/*.h5 (nirHiss R=300 orders 1–2 fiducial; supreme-SPOON alternate) | `jwst_readers.ers_niriss_h5` |
| 7101283 (Ahrer 2023, NIRCam) | NIRCam_Zenodo_files.zip | 2_TRANSMISSION_SPECTRA/LW-…-15nm_Eureka.h5 | `jwst_readers.ers_h5` |
| 7185300 (Alderson 2023, G395H) | G395H_paper_data.zip | 3_TRANSMISSION_SPECTRA/…_weighted-average.nc | `jwst_readers.ers_nc` |
| 7388032 (Rustamkulov 2023, PRISM) | W39_PRISM_data.zip | transit_spectra/{FIREFLy,Eureka,Tiberius,Tshirt}_transit_spec.h5 (FIREFLy fiducial; the 0.634–2.115 µm saturated region dropped as in Carter & May's satflag); best_fit_model/ kept for reference | `jwst_readers.prism_h5` |
| 10055845 (Powell 2024, MIRI) | miri_lrs_zenodo.zip | Figure2_Spectra/spectra/{Eureka!,SPARTA,Tiberius}-…-spectrum.h5 (Eureka! fiducial; 28 × 0.25 µm bins, 5.1–11.9 µm, ppm); the PICASO grid and photochemical-model spectra kept for reference | `jwst_readers.miri_h5` |
| 10161743 (Carter & May 2024, benchmark) | ERS_DataSynthesis_Zenodo.zip | 4_TRANSMISSION_SPECTRA/Fixed_LimbDarkening/{NIRISS_SOSS_Order1,Order2,NIRCam_F322W2,NIRSpec_G395H_NRS1,NRS2}/R100.csv and NIRSpec_PRISM/bins_scale1+dilution.csv (satflag ≠ 0 dropped); Fit_LimbDarkening as the alternate; 5_MODEL_SPECTRUM/BestFitModel.txt (0.5–12.5 µm, R = 500) for gap filling and MIRI levelling; Archival_Data.txt (HST/Spitzer/VLT) not used | `jwst_readers.carter_csv`, `carter_model` |

### Panel (`v3/data/jwst/panel/<planet>/`)

| Planet | Source | File(s) used | Columns |
| :-- | :-- | :-- | :-- |
| TOI-270 d | Zenodo 15228454, TOI-270d_transit_spectra.zip (Felix et al. 2025 reanalysis) | native_NIRISS.dat, native_NRS1.dat, native_NRS2.dat | wl µm, depth ppm, err ppm, fwhm µm |
| K2-18 b | osf.io/36djh (Madhusudhan et al. 2023, discovery-team originals); MIRI from Zenodo 16277833, Spectra.zip (Jaziri et al. 2025 copy of the Madhusudhan et al. 2025 spectrum) | K2-18b_niriss_soss_native.txt, K2-18b_nirspec_g395h_native.txt (the lowres pair and README kept); K2-18b_miri_lrs_eureka_taurex.txt | wl µm, half-width, depth, err; wl, depth, err, half-width |
| K2-18 b (Hu) | osf.io/hpu8g, spectra-k2-18b.zip (Hu et al. 2025) | spectra/niriss_nameless/…order1_vis1_R300.spec, …order2_vis1_R100.spec; spectra/nirspec_combined/K2_G235H_NRS1_NRS2_ave2_direct.dat, K2_G395H_NRS1_NRS2_ave3_direct.dat | ECSV: wave, waveMin, waveMax, yval ppm, yerrLow/Upp ppm, ignore; lo, hi, mid µm, depth, err |
| TOI-270 d (Holmberg) | osf.io/8fu36 (Holmberg & Madhusudhan 2024) | TOI-270d_nirspec_g395h_native.txt (canonical; the CLD and DT cases, lowres and README kept) | wl µm, half-width, depth, err |
| K2-18 b (alternate) | Zenodo 14735689, K2-18_reanalysis_data.zip (Schmidt & MacDonald 2025) | exoTEDRF R=100: NIRISS orders 1–2, NRS1, NRS2 | wl, half-width, depth, err |
| TOI-421 b | Zenodo 15176438, TOI421b.zip (Davenport et al. 2025) | NIRISS/exoTEDRF/…_Final_50pix.csv; NIRSpec/Deming_spectrum.txt | wl, half-width, ppm, ppm, order; wl, depth, err |
| HAT-P-18 b | VizieR J/ApJ/940/L35 (Fu et al. 2022), https://cdsarc.cds.unistra.fr/ftp/J/ApJ/940/L35/table1.dat | table1.dat | wl µm, depth ppm, err ppm |
| WASP-107 b | Zenodo 10780449, FinalResults.zip (Welbanks et al. 2024); Zenodo 10891400 (Sing et al. 2024) | FinalSpectra/NIRCam_F322W2, NIRCam_F444W, MIRI_LRS (Eureka! fiducial); Sing_2024_Fig2_….csv | wl, width, depth, err; wl, half-width, Rp/Rs, err |
| HD 189733 b | Zenodo 11459715, HD189733b_JWST_data_model.zip (Fu et al. 2024) | Data/transit spectrum/HD189733b_NIRCam_spectrum.txt | wl, half-width, depth, err |
| HAT-P-26 b | AJ 170, 292 figure-data package https://content.cld.iop.org/journals/1538-3881/170/5/292/revision1/ajae0929f2.tar.gz (Gressier et al. 2025) | f2/dbf2a.txt (ExoTiC-JEDI) | wl, depth (fractional despite the ReadMe's "ppm"), err |
| GJ 3470 b | ApJL 970, L10 machine-readable tables (Beatty et al. 2024), https://iopscience.iop.org/2041-8205/970/1/L10/suppdata/apjlad55e9t{3,4}_mrt.txt | t3 (F322W2), t4 (F444W) | bin lo, hi µm, depth ppm, err ppm (−163 ppm offset already applied by the authors) |
| WASP-166 b | Zenodo 15122285, Zenodo_WASP-166b.zip (Mayo et al. 2025) | 0.01797-µm Eureka! tables: NIRISS order 1 and NIRSpec G395M (v81) | ECSV: wl, half-width (named bin_width), rp², errneg, errpos |
| TOI-1231 b | osf.io/mryd6 (Sarkar et al. 2026) | TOI-1231b_{niriss_ord1,niriss_ord2,nirspec_g395h_nrs1,nirspec_g395h_nrs2}_JexoRes.txt (the JexoPipe set kept as the alternate) | wl µm, half-width, depth, err |

The OSF files are fetched by their file ids (`fetch_jwst_data.sh`, "OSF" block) and checksummed like the
rest. Not used: Benneke et al. 2024's TOI-270 d (MAST observations only); Luque et al. 2025's 2.9 GB K2-18 b record (Zenodo 15685551); the
Fournier-Tondreau et al. 2024 HAT-P-18 b spectrum (MAST only); DERIVED tables made during the survey
(TOI-421 b Eureka! depths from Stage-5 posteriors; Dyrek et al. 2024 MIRI CSV from Nature source data).
