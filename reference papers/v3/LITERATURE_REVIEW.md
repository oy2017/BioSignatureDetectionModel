# Literature review for the v3 paper (compiled 2026-09-12)

Purpose: (1) every claim the paper makes about prior work is tied to a paper we have read or
whose relevant passage we have verified; (2) every novelty claim is checked against a search
phrased as its negation. Files: `INDEX.md` maps keys to PDFs; `references.bib` is the BibTeX.
Searches were run 2026-09-11/12 on arXiv, ADS-indexed journals and the web; anything found
that contradicts a claim is recorded under that claim.

---

## 1. The mission and the screens it has proposed

**Ariel and its tiers.** Tinetti et al. 2018 (Exp. Astron. 46, 135; no arXiv) set the science
case: ~1000 planets, 0.5–7.8 µm, a tiered survey. Edwards et al. 2019 (AJ 157, 242) and Edwards
& Tinetti 2022 (AJ 164, 15) define the target list and the tier assignment: a planet reaches
Tier X when its spectrum, binned as Tier X prescribes, reaches SNR ≥ 7 on the assumed
modulation after N transits, computed with ArielRad (Mugnai et al. 2020, Exp. Astron. 50, 303).
The 2026 tier reassessment (arXiv:2604.07598) gives the binning: R ~ 1/3/1 (NIRSpec/CH0/CH1)
at Tier 1, 10/50/10 at Tier 2, native at Tier 3 — the layouts we use. **Verified.**

**The consortium's Tier-1 screen.** Mugnai et al. 2021 (AJ 162, 288; arXiv:2110.00503) present
"a strategy to select candidate planets for reobservation in Ariel's higher resolution Tier",
a band metric for composition without retrieval, and in §II.5/§III.3 four scikit-learn
classifiers (k-NN, MLP, RF, SVC) trained on simulated Tier-1 spectra (POP-III), tested on
POP-I, reaching 64–89 % per molecule (their Table 6). Validation is internal to TauREx 3 +
ArielRad; they state the spectra "will only be used as 'transmission spectral shapes' to test
our methods against". Alfnoor is not public. **Verified from the full text.** This is the
object we rebuild.

**The Data Challenge grids.** Changeat & Yip 2023 (RASTI 2, 45; arXiv:2206.14633) describe the
ABC database of 105,887 TauREx forward models; §2.2: "The trace gases are H2O, CH4, CO, CO2 and
NH3." The 2023 challenge (Aubin et al. 2023, arXiv:2309.09337) inherits the procedure; targets
are R, T and those five abundances. Yip et al. 2023 (PMLR 220) give the lessons learned. The
2024 dataset (Mugnai et al. 2026, arXiv:2605.03719) builds a deliberate train/test shift in
instrument noise and systematics for detrending; no physics shift. **Verified.** The species
omission is therefore the field's, not ours.

**ExoSim 2** (Mugnai et al. 2025, Exp. Astron.; arXiv:2501.12809) is the consortium's
time-domain simulator whose radiometric noise shape we adopt; our comparison with ExoRad is in
`v3/results/noise_curves_comparison.txt` (shape agrees on FGK hosts, diverges on M dwarfs).

## 2. Machine learning for exoplanet atmospheres, and how it is validated

Márquez-Neila et al. 2018 (Nat. Astron.; random forest, WFC3, H2O/HCN/NH3 — note HCN
included), Zingales & Waldmann 2018 (ExoGAN), Cobb et al. 2019 (BNN ensembles), Nixon &
Madhusudhan 2020 (RF assessment; no OOD test), Yip et al. 2021 (DNN sensitivity), Vasist et
al. 2023 (NPE with in-simulator coverage checks), Gebhard et al. 2024/2025 (flow matching, noise-
level conditioning), Hayes et al. 2020 (PCA/k-means classes as priors), Duque-Castaño et al.
2025 (MultiREx and a low-SNR classification screen), the 2025 supervised-ML-with-UQ study
(arXiv:2508.04982), and the 2026 review (arXiv:2606.23766). **Common property: every one is
validated on held-out spectra from its own forward model.** The review's §4.2 lists
"calibration under instrument mismatch" and low-S/N OOD spectra (Gebhard 2023; Orsini 2025)
as open; physics-side mismatch is not mentioned.

**Closest precedent: Ardévol Martínez et al. 2022 (A&A 662, A108; arXiv:2203.01236).** A
retrieval CNN trained with noisy copies, tested on spectra with AlO added, TiO/VO removed, and
unocculted spots (their §5.1–5.3), against nested sampling. They price three mismatches; they
do not absorb, detect, bound with ceilings, hold out, evaluate at the decision tier or on the
target list. Their free-chemistry training models use H2O, CO, CO2, CH4, NH3; HCN and C2H2
appear only in their equilibrium-chemistry models. **Verified from the full text.** Must be
cited in the first paragraph of related work.

**Classical-retrieval misspecification** (context, not competitors): Barstow et al. 2020
(retrieval-code intercomparison), ARES VI (arXiv:2401.03809; 1D retrievals biased by 3D
effects), Kawashima & Min 2021 and the 2025 ten-hot-Jupiter re-analysis (arXiv:2506.12806)
showing that the equilibrium assumption biases retrieved C/O.

## 3. Chemistry of the label and of the omitted species

Madhusudhan 2012 (ApJ 758, 36; arXiv:1109.3183): at C/O ≥ 1 and T > 800 K "C2H2, CH4 and HCN
become major constituents", enhanced by 3–6 orders of magnitude; "C2H2 and HCN can be
considered good tracers of the C/O ratio". Moses et al. 2013 (ApJ 763, 25): the same with
quenching and photochemistry; disequilibrium enhances CH4, NH3, HCN, C2H2. Zahnle & Marley
2014: the CO/CH4 quench timescale we use. Stock et al. 2018: FastChem. **Verified.** Our
FastChem HCN ~1e-5 and C2H2 ~3e-5 on carbon-rich planets sit inside these papers' ranges.

## 4. Stellar contamination, aerosols, opacities, codes

Rackham et al. 2018 (transit light source effect; largest for M dwarfs — the expected host
dependence), Pinhas et al. 2018 (stellar heterogeneity in retrieval). Lee et al. 2013 (the Mie
haze prescription). Al-Refaie et al. 2021 (TauREx 3), Kempton et al. 2017 (Exo-Transmit;
opacities from Freedman et al. 2008/2014 via Lupu et al. 2014 Table 2, including HCN and
C2H2), Chubb et al. 2021 (ExoMolOP). **Verified.**

## 5. The machine-learning methods we apply — none new, all cited as origins

Domain randomization: Tobin et al. 2017. Noise augmentation as regularization: Bishop 1995;
augmentation as invariance: Chen, Dobriban & Lee 2020; variance tuning: Rusak et al. 2020.
Confidence baseline: Hendrycks & Gimpel 2017. Distance scores: Lee et al. 2018 (Mahalanobis),
Sun et al. 2022 (k-NN). Ensembles: Lakshminarayanan et al. 2017. Selective prediction: Geifman
& El-Yaniv 2017. OOD detection ≠ failure detection: Jaeger et al. 2023 (ICLR; FD-Shifts) —
our detection inversion reproduces this on spectra. Held-out corruption protocol: Hendrycks &
Dietterich 2019. Conformal prediction: Angelopoulos & Bates 2021; under covariate shift:
Tibshirani et al. 2019. Target-trained oracle bound: standard in domain adaptation.

## 6. Misspecification for simulation-trained inference (the closest methodological neighbours)

Schmitt et al. 2021 (misspecification detection for amortized Bayesian inference), Cannon et
al. 2022 (impact on neural SBI), Huang et al. 2023 (robust summary statistics), and the 2025
cosmology papers (arXiv:2507.13495, deep ensembles; 2508.05744, flow-based OOD). **The question
is recognised as important in cosmology and has not been asked of exoplanet atmospheres, where
the mismatch is physics rather than simulation fidelity.**

---

## Novelty verdicts (search phrased as the negation; result)

| claim | negation searched | found | verdict |
|---|---|---|---|
| No exoplanet ML screen has been tested outside its simulator with physics mismatch, ceilings, held-out axes, absorb+detect, at the decision tier, on the target list | "out-of-distribution / robustness / misspecification exoplanet ML retrieval or classifier" | Ardévol 2022 (3 axes, priced only); ADC 2024 dataset (instrument only); review lists it as open | **holds**; cite Ardévol first |
| Domain randomization has not been applied to astronomical spectra | "domain randomization spectroscopy astronomy" | robotics/vision only; one radioisotope-ID sim-to-real paper | **holds** |
| Selective prediction / abstention has not been applied to exoplanet screens; conformal not either | "selective classification OR conformal exoplanet" | none | **holds** (include conformal as baseline) |
| The standard Ariel ML grid omits HCN and C2H2 | read ABC §2.2, ADC 2023 targets, Ardévol free-chem models | five gases; HCN/C2H2 absent | **holds** |
| No one has shown a carbon-rich ML screen failing on the omitted species | "HCN C2H2 machine learning classifier bias water CH4" | retrieval-bias papers on equilibrium vs disequilibrium (different task); nothing on ML screens | **holds** |
| "Quenching helps an equilibrium-trained C/O classifier" is unreported | "disequilibrium improves C/O constraint / classification" | Moses 2013 chemistry consistent; no ML statement | **holds**, narrow |
| The detection inversion is new | "OOD detection failure detection" | Jaeger 2023 — established in ML | **not new**; cite, present as reproduced |
| Contamination is worst for M dwarfs | Rackham 2018 | known | **not new**; cite |
| The repair rule of our earlier work | — | retired by our own ceiling test | **withdrawn** |

## Competitor scan (2026)

Searches on 2026-09-12 for "trust / reliability / robustness / OOD / misspecification" with
"exoplanet", "Ariel", "transmission spectra", "triage" returned no paper that overlaps the
programme; the only 2026 items are the review (arXiv:2606.23766) and the ADC 2024 dataset paper
(arXiv:2605.03719), both of which frame the question as open. (The search also returned this
repository, which is public.)

## Papers still to read in full before submission

Mugnai 2021 §II.2 (population recipe details — the CH0 split for Tier 1, the exact noise
scaling), ADC 2023 (Aubin) methods, Jaeger 2023 (to phrase the reproduction correctly),
Tibshirani 2019 (the covariate-shift conformal variant a referee may propose), ARES VI
(whether 3D effects belong in our limitations as an unpriced axis).
