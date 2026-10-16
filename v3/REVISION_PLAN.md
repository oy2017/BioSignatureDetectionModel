# Revision plan: from technical note to original paper

Written 2026-10-15, the day of the JHSS decision (reject, with the editor's offer to move the
manuscript back to revision if three additions are made). This file is the working plan for the
revision. It follows the repository's practice: expectations are written down and committed
before each run, kill criteria are named, and every result has a control.

The manuscript under revision is `jhss/note_src.md`; its supplementary directory is
`../ariel_tier1_trust/`. The decision letter and the review are kept locally and are
git-ignored (`*Scholastica*.pdf`, `*Gmail*.pdf`, `ariel_tier1_trust/*.pdf`).

---

## 0. What was asked, and what each answer has to look like

The editor (Shireesh Apte, 2026-10-15, after discussion with the reviewer) made the
decision conditional on three additions. The reviewer's report adds four further points that
the revision must answer even though the editor did not list them.

| # | Request | What a satisfying answer looks like |
| :-- | :-- | :-- |
| E1 | **External validation on WASP-39 b.** Compare the classifier's predictions with the JWST-derived composition, *provided the observed spectra are made compatible with the classifier's input requirements, with adequate justification of the compatibility procedure*. | A documented, validated conversion of the JWST spectrum to the classifier's 104-bin Tier-3 input; the classifier's verdict per molecule at JWST precision and at Ariel Tier-1 precision; agreement with the published composition, with the ambiguous label (CO₂) treated as ambiguous. |
| E2 | **Identifiability.** Whether physically distinct compositions produce indistinguishable Ariel spectra, and how often, under explicitly defined populations. | A per-planet, per-molecule identifiability test at Tier-1 precision; the fraction of a named population whose label is undecidable; the resulting ceiling on any classifier's accuracy. |
| E3 | **A remedy with measurable value.** A diagnostic that recognizes when the classifier is likely unreliable, shown to improve classification or target selection *compared with existing methods*. | A diagnostic with positive selective credit where the five existing rules had negative credit, and a mission-level selection simulation in which it loses fewer valuable targets than the existing rules at the same Tier-2 budget. |
| E4 | **The two error classes** (the editor's "most interesting possible outcome"): recoverable computational errors (information present, classifier misreads it) versus irrecoverable information limits (compositions indistinguishable at the instrument's precision). | Every classifier error, under every departure, assigned to one class by the E2 test; the prevalence of each; the diagnostic of E3 tuned to tell them apart. |
| R1 | Trustworthiness is a broader claim than robustness to a finite perturbation set; the joint distribution of conditions across Ariel's targets is unknown. | Drop "trust" from the title and claims; report prevalence under *named* populations (the consortium prior, the known Ariel targets) and say that the true distribution is unknown. |
| R2 | Aggregate accuracy hides a collapse in sensitivity to one molecule; the relevant quantity is valuable targets lost under a constrained selection strategy. | Make the mission-level metric (targets lost, slots wasted at a fixed Tier-2 budget) a primary result, alongside accuracy. |
| R3 | Weaknesses belong to the re-implementation and the tested design, not to the unavailable original models. | Say so consistently, not only in the limitations. |
| R4 | The second classifier shows only that two different classifiers respond differently. The confidence/conformal results are well known. | Compress both: the second classifier to a short section or appendix, the decline rules to a paragraph that cites Jaeger et al. 2023 as the expected outcome. |

The reviewer's own words on what would justify reconsideration: "a consequential, non-obvious
result about the propagation of simulator misspecification into Ariel's scientific
decision-making or population-level inference." E2 + E4 + the mission-level metric (R2) are
that result; E1 is the external anchor; E3 is the remedy.

---

## 1. The reframed paper

**Working title.** *Recoverable and irrecoverable errors of a simulator-trained classifier for
Ariel Tier-1 spectra, with an external test on JWST observations of WASP-39 b.*

**The spine.** A classifier's error on a spectrum is one of two things: the information was in
the spectrum and the classifier missed it (recoverable), or the information was not there at
that precision (irrecoverable). The first is a property of the classifier and can be fixed
by retraining, input selection or a fallback; the second is a property of the instrument and
the population, and no classifier can fix it, so the mission must not act on those spectra
as if they were decided. The paper measures both classes for the consortium design,
under its own simulator and under each departure, shows which class each departure
produces, and tests the picture on a real spectrum. The diagnostic separates the two classes
at deployment time and is scored by targets saved.

**What stays from the note.** The re-implementation and its reproduction (Section 3.1); the
departure table and losses, now reported per class; the haze mechanism (Section 3.3, the
result the reviewer singled out as the most informative); the pre-registration practice.

**What is cut or compressed.** The carbon-rich classifier becomes a one-paragraph
counterexample or an appendix. The decline-rule section becomes a paragraph. The word
"trust" leaves the title and the claims.

---

## 2. Work package A: external validation on WASP-39 b

### A1. Data

Primary input: the ERS data-synthesis benchmark spectrum of Carter & May et al. 2024
(Nat. Astron. 8, 1008; Zenodo 10.5281/zenodo.10161743), 0.518–5.348 µm, four JWST modes
(NIRISS SOSS, NIRCam F210M+F322W2, NIRSpec G395H, NIRSpec PRISM) with uniform light-curve
fitting and the inter-mode offsets fitted (residual ≈ 140 ppm). Extended to 7.8 µm with the
MIRI LRS spectrum of Powell et al. 2024 (Nature 626, 979; Zenodo 10.5281/zenodo.10055845),
5–12 µm in 0.25 µm bins (transit-depth errors of several hundred ppm per bin).

Coverage of the Ariel channels from real data:

| Ariel channel | Range (µm) | JWST source | Note |
| :-- | :-- | :-- | :-- |
| VISPhot | 0.50–0.60 | NIRSpec PRISM | PRISM is unsaturated here (saturation is 0.8–2.3 µm) |
| FGS1 | 0.60–0.80 | NIRISS SOSS (order 2) + PRISM | K line at 0.77 µm is in this band |
| FGS2 | 0.80–1.10 | NIRISS SOSS | PRISM saturated; SOSS only |
| NIRSpec | 1.10–1.95 | NIRISS SOSS | PRISM saturated below 2.3 µm; SOSS only |
| AIRS-CH0 | 1.95–3.90 | SOSS (to 2.8) + NIRCam F322W2 (2.4–4.0) + G395H (from 2.8) + PRISM (above 2.3) | best-covered region |
| AIRS-CH1 | 3.90–7.80 | G395H + PRISM (to 5.3), MIRI LRS (5.0–7.8) | 5.3–7.8 µm from MIRI only, coarse and noisy |

Sensitivity inputs: the individual ERS reductions (Feinstein 2023 SOSS; Ahrer 2023 NIRCam;
Alderson 2023 G395H; Rustamkulov 2023 PRISM) and the alternative MIRI reductions (Tiberius,
SPARTA) in the same Zenodo records, to measure how much the verdict depends on the
reduction. Roy-Perez et al. 2026 (arXiv:2602.06722) report up to 2 dex abundance scatter
across PRISM reductions, so the verdict's reduction-dependence must be shown, not assumed.

### A2. The compatibility procedure (the step the editor bolded)

Each step is a stated choice with a measured consequence.

1. **Quantity.** JWST products are transit depth (R_p/R_s)²; the classifier was trained on
   the same quantity from TauREx. No conversion.
2. **Binning.** Each of the 104 Tier-3 bins (`ariel_bins.tier_channels(20, 100, 30)`)
   takes the inverse-variance-weighted mean of the JWST points whose wavelength falls in
   the bin, with the error propagated. Partial overlap at bin edges is weighted by the
   overlap fraction. The training spectra were binned by integration at R ≈ 1,000, so the
   difference between integrating a model and averaging JWST points of finite width is
   measured in step 7.
3. **Coverage.** For every bin, record the number of JWST points and which mode supplied
   them. Bins with no data (expected: none, but check 1.95–2.0 and 5.3–5.5 µm) are filled
   from the Carter benchmark model and flagged; the verdict is reported with and without
   the filled bins.
4. **Offsets between modes.** Use the Carter offsets as the baseline. Sensitivity: redraw
   the mode offsets 1,000 times from their fitted uncertainties and re-run the classifier,
   giving a verdict distribution rather than a single verdict. Per-spectrum normalization
   removes any *global* offset, so only *relative* offsets between modes matter.
5. **Noise regimes.** Three versions of the input:
   (i) *JWST precision as delivered*: the binned errors (an upper bound on what Ariel can
   see, since JWST is more precise everywhere);
   (ii) *Ariel Tier-1*: the per-bin σ from the payload model (`noise.radiometric_sigma`) at
   WASP-39 b's Tier-1 transit count from the Mission Candidate Sample, added as 1,000
   Gaussian draws to the binned JWST spectrum, giving a detection frequency per molecule;
   Radica et al. 2026 find one Ariel transit of WASP-39 b reaches "Tier 1.5", which the
   payload model should reproduce;
   (iii) *Ariel Tier-2 and Tier-3 counts*, for context.
   The JWST errors are not subtracted before adding Ariel noise; they are smaller than the
   Ariel σ in every bin, and the added variance is reported.
6. **Normalization.** Zero mean, unit dispersion per spectrum, exactly as in training.
7. **Round-trip validation of the procedure itself.** Take the Carter benchmark *model*
   (a TauREx-like forward model of WASP-39 b at 10× solar, C/O = 0.35); (a) bin it the way
   the training set was binned; (b) sample it at the JWST wavelength grids of each mode,
   add the per-point JWST errors, and push it through steps 2–6. If the classifier's
   output on (a) and (b) differ, the procedure, not the planet, is the cause. Report the
   bin-level difference between (a) and (b) in ppm and in classifier probability. This is
   the "adequate justification": the conversion is shown not to change the answer on a
   spectrum whose answer is known.
8. **Stellar contamination.** WASP-39 is a quiet G8V star; no correction is applied, and
   this is stated. The 2–20 % spot departures already bound the effect.

Implementation: a new script `v3/jwst_to_ariel.py` (download, bin, coverage map, round-trip
test) and `v3/alfnoor_wasp39.py` (train the four classifiers as in `alfnoor_faithful.do_fit`,
apply them to the converted spectrum under the three noise regimes, write a verdict table).

### A3. Ground-truth labels for WASP-39 b

The classifier's label is "abundance > 10⁻⁴ (also 10⁻⁵, 10⁻³)" for a well-mixed isothermal
atmosphere. A real atmosphere has no single abundance; the label is defined as the volume
mixing ratio at the pressures the transmission spectrum probes (≈ 0.1–10 mbar), taken from
published retrievals. Consensus from the brief of 2026-10-15:

| Molecule | Published constraint | Label at 10⁻⁴ | Label at 10⁻⁵ | Label at 10⁻³ |
| :-- | :-- | :-- | :-- | :-- |
| H₂O | log VMR −2.4 to −1.2 (MIRI free retrievals); 10× solar in all ERS fits; one PRISM reduction gives −4.9 to −3.3 (Constantinou 2023, Tiberius) | present | present | present (ambiguous under the Tiberius reduction) |
| CO₂ | log VMR −6.3 to −4.5 (Constantinou 2023), −3.1 (Ma 2025); 10× solar equilibrium gives ≈ 10⁻⁵–10⁻⁴ at 1,100 K | **ambiguous** | present | absent |
| CH₄ | not detected; VMR ≲ 5 × 10⁻⁶ (PRISM), ≤ 5.5 × 10⁻⁵ (NIRCam) | absent | absent (PRISM limit) | absent |
| NH₃ | not detected; no formal published limit (the "< 10⁻⁶" quoted by Forestano 2025 could not be traced) | absent | absent (unverified) | absent |

CO₂ at 10⁻⁴ is not a usable label: published retrievals of the same data straddle it by
more than a decade. The paper must say that this is itself an identifiability statement
about JWST, and score CO₂ only at 10⁻⁵ and 10⁻³, or report its verdict without a score.

Independent third reference: run the repository's own nested-sampling retrieval
(`v2/retrieve.py`, adapted to the consortium forward model and the 104-bin layout) on the
converted spectrum at JWST precision and at Tier-1 noise, giving P(X > 10⁻⁴) per molecule, as
Mugnai et al. 2021 did for one simulated planet (their Table 8). This ties A to B.

### A4. What is measured

For each of the four classifiers × three thresholds × three noise regimes:

- the verdict per molecule, and its probability where the classifier gives one;
- under Tier-1 noise, the detection frequency over 1,000 draws;
- the same for the input variants already built in `alfnoor_haze_mechanism.py` (all 104
  bins; photometric points removed; AIRS bins only). This is the haze mechanism tested on a
  real spectrum: WASP-39 b has a grey cloud (21σ), Na and K wings in the optical, and an
  optical slope, none of which are in the training grid. If the all-input classifier calls
  H₂O absent while the AIRS-only classifier calls it present, the mechanism of Section 3.3 has
  been seen in real data;
- the Mugnai band metric M_mol on the same binned spectrum (the consortium's own non-ML
  method; their Eq. 3, bands in their Table 3);
- the retrieval posterior of A3.

Reduction sensitivity: repeat with each individual ERS reduction in place of the benchmark.

### A5. Expectations, to be committed before the run

1. At JWST precision all four classifiers report H₂O present, CH₄ absent, NH₃ absent at
   10⁻⁴; CO₂ split across classifiers.
2. Under Tier-1 noise, H₂O detection frequency > 80 %; CH₄ and NH₃ false-positive frequency
   < 20 %; CO₂ near 50 %.
3. Risk, stated as a prediction: the real optical points (cloud slope, Na, K) push the
   all-input classifiers toward the "absent" verdict for H₂O in at least one classifier,
   and the no-photometric variant does not; this is the haze failure mode on real data.
4. The band metric agrees with the classifiers on H₂O and CH₄.
5. The verdict on H₂O and CH₄ is the same for every reduction; CO₂ is not.

A miss on any of these is reported as a miss.

### A6. Beyond one planet: a JWST panel

One planet gives four binary labels, one of them ambiguous, and no positive case for CH₄, the
molecule the haze failure concerns most. The reviewer asked for WASP-39 b; the paper is
stronger with a panel of JWST transmission spectra that provides CH₄-positive planets and a
range of cloudiness and host types. Selection rules, fixed in advance:

- a published JWST transmission spectrum covering at least 0.6–5 µm (so the photometric
  points and AIRS-CH0 come from data), with a public reduced spectrum;
- a published retrieval giving CH₄, H₂O, CO₂ or NH₃ clearly above or below 10⁻⁴;
- the planet is in the Ariel Mission Candidate Sample (checked against the 2026-05-11 list),
  so its Tier-1 transit count and noise exist.

Panel from the literature survey of 2026-10-15 (`jwst_validation_brief.md`, part 2). Labels
at the 10⁻⁴ threshold; "amb." means the published posterior straddles 10⁻⁴ or sources
disagree, and the molecule is not scored for that planet.

| Planet | Coverage (JWST modes) | CH₄ | H₂O | CO₂ | NH₃ | Why it is in the panel |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **TOI-270 d** (sub-Neptune, 350 K, M3V) | 0.6–5.2 (SOSS + G395H) | **present** (log −1.6, 9.4σ) | amb. | present | absent (< −4.3) | the cleanest CH₄ positive: full coverage, quiet star, largely clear |
| **K2-18 b** (sub-Neptune, 255 K, M2.5V) | 0.85–5.2 (SOSS + G395H + G235H) | **present** (three independent analyses, 4–11σ) | absent (< −3.9) | amb. (disputed) | absent | CH₄ positive with cloud + haze and spot crossings; spectra on OSF |
| **TOI-1231 b** (temperate Neptune, 300 K, M3V) | 0.65–5.2 (SOSS + G395H) | **present** (log −0.85) | amb. | present | amb. (loose limit) | CH₄ positive with clouds and haze; spot crossings; spectra on OSF |
| **WASP-166 b** (hot super-Neptune, 1,270 K, F9V) | 0.85–5.2 (SOSS + G395M) | absent (−7.9) | present (15σ) | present (14.7σ) | amb. (2.3σ) | the clean H₂O/CO₂ positive, CH₄ negative; Zenodo |
| **TOI-421 b** (hot sub-Neptune, 920 K, late G) | 0.83–5.0 (SOSS + G395M) | absent (−8.4) | amb. | absent (−9.1) | — | haze-free, quiet star: a near-featureless negative control |
| **HAT-P-18 b** (warm Saturn, 850 K, K2V) | 0.6–2.8 (SOSS only; AIRS-CH1 missing) | absent (< −6.1) | amb. (−4.4) | absent (−4.8) | absent (< −6.0) | the stellar-contamination case: unocculted spots at 5.8σ set the optical slope, and a 3σ CH₄ "hint" vanished once spots were modelled |
| WASP-107 b (warm super-Neptune, 740 K, K6V) | 2.45–12 (NIRCam + MIRI) + HST 0.9–1.6 | absent (≈ 1 ppm, three papers) | present (21σ) | amb. | absent (detected at −5.0, below threshold) | partial coverage; silicate clouds; Zenodo |
| HD 189733 b (hot Jupiter, 1,200 K, K1V) | 2.4–5.0 (NIRCam) | absent (< −7.9) | amb. | absent (−6.2) | absent (< −5.3) | partial coverage; spot-corrected; Zenodo |
| HAT-P-26 b (warm Neptune, 1,000 K, K1V) | 2.8–5.2 (G395H) | absent (< −5.5) | present | present | absent (< −4.2) | partial coverage; SO₂ present (omitted absorber) |
| GJ 3470 b (warm sub-Neptune, 600 K, M1.5V) | 2.45–5.0 (NIRCam) | amb. (−4.05) | present | present | absent | partial coverage; the CH₄-at-threshold case for B3.5 |

The first six have the optical photometers and AIRS-CH0 covered by data and are the main
panel (HAT-P-18 b without AIRS-CH1). The last four cover only AIRS and are used for the
AIRS-only input variant and for the molecules whose bands they contain; their missing bins
are model-filled and flagged, and the verdict is reported with and without them.

What the panel buys, counted at 10⁻⁴: three CH₄ positives and six CH₄ negatives (seven with
WASP-39 b); five H₂O positives and one negative; five CO₂ positives and three negatives;
seven NH₃ negatives and **no NH₃ positive anywhere in the JWST literature** (the only
detections, WASP-107 b and a 2026 claim for WASP-80 b, are below 10⁻⁴). The paper must
state that the NH₃ classifier can be validated on negatives only. About 30 decidable
planet–molecule labels in all, against 3 for WASP-39 b alone.

Two cautions for the panel. All three CH₄ positives are temperate sub-Neptunes around M
dwarfs, for which the payload model's Tier-1 transit count is large (Radica et al. 2026 note
such planets may be prohibitive even for Tier 1); regime (ii) must use the MCS count as it
stands and say so. And several panel planets have features absent from the training grid
(SO₂, CS₂, H₂S, H⁻, non-grey clouds, high mean molecular weight for the sub-Neptunes): the
same mask test as for WASP-39 b (A7) applies, and sub-Neptune spectra sit at the edge of
the consortium prior (H₂/He atmospheres only), which the distance scores should flag; that
is itself a test of the decline rules on real data.

Companion resource: the MCS list downloaded during the survey is at
`/private/tmp/claude-502/-Users-fanyang/18593b96-2825-4253-b1c9-b5ac9e0b41e5/scratchpad/ariel/mcs_2026.csv`
(session-temporary; re-download from arielmission-space/Mission_Candidate_Sample).

### A7. Caveats the paper must state

- The JWST spectrum is not an Ariel observation: different instruments, systematics,
  resolution and epochs. The procedure of A2 makes the *input format* compatible; it does not
  make JWST noise into Ariel noise. Regime (ii) is the closest stand-in.
- Published abundances depend on the reduction and the retrieval model by up to 2 dex;
  the labels are consensus labels with their disagreement shown.
- The classifier's training grid has no SO₂ (4.05 µm, in AIRS-CH0), no CO (4.6 µm), no
  Na or K (optical), no non-grey cloud. WASP-39 b has all of these. Their presence in the
  real spectrum is an omitted-absorber departure in real data; the paper should say which
  bins they affect and whether the verdict depends on them (mask test: replace the SO₂ and
  CO bins with the model-filled continuum and re-run).
- Prior sim-to-real attempts on this very spectrum failed: Forestano et al. 2025
  (arXiv:2508.04982) applied regressors trained on the Ariel Big Challenge grid to the
  PRISM + MIRI spectrum rebinned to 52 Ariel channels and predicted log H₂O = −6.7 against
  a retrieved −2 to −1; Chakravertty & Raj 2026 (arXiv:2610.02245) report an amortized
  posterior collapsing on WASP-39 b (χ²/N ≈ 300) until the planet radius was added as a
  latent. These are the precedents the introduction should cite: the question of
  whether a simulator-trained model survives a real spectrum has been asked, and the
  answers so far were no.

---

## 3. Work package B: identifiability

### B1. Definition

For planet *i* with true parameters θ and molecule *X*, the label "X > 10⁻⁴" is
**identifiable at precision σ** if no atmosphere with the opposite label, drawn from the
stated population, fits the noise-free spectrum within the noise. Operationally:

    Δχ²_i(X) = min over θ' in the opposite class of  Σ_b [ (S_b(θ) − S_b(θ')) / σ_b ]²

where the minimum runs over the nuisance parameters (temperature, radius, cloud-top
pressure, the other three abundances) with X fixed at the class boundary (10⁻⁴ is the
hardest counterfactual because band depth is monotone in abundance; checked on a subset by
also freeing X within the opposite class). The label is identifiable at significance *s* if
Δχ² > s² (one-parameter nested-model test, Wilks). The threshold is calibrated, not
assumed: on a stratified subset of 200 planets, the nested-sampling retrieval gives
P(X > 10⁻⁴) directly, and the Δχ² that corresponds to P = 0.95 (and 0.84) is read off.

The population is explicit: POP-I (the consortium's test prior) for the main result; the
965 known Ariel targets with their own Tier-1 transit counts (the mission's actual
population under the same atmospheric prior) as the second. The paper states that the true
joint distribution of atmospheric conditions is unknown and that prevalence is reported under
these two named priors (answers R1 as far as it can be answered).

### B2. Compute

Rendering costs ≈ 0.4 s per spectrum on 8 workers (`alfnoor_faithful_render.log`: POP-III,
3,908 spectra, 201 s). The Δχ² minimization needs ≈ 300–600 forward calls per planet-molecule
(Nelder–Mead or a bounded Powell in six nuisance dimensions, warm-started at the true θ with
X moved to the boundary): 955 planets × 4 molecules × 500 calls ≈ 2 × 10⁶ calls ≈ 1 day on 8
cores, 10 h on 22. The calibration retrievals: the pilot in `jhss_paper2/OUTLINE.md` ran a
median 14 min per planet at 100 live points with six free parameters; 200 planets ≈ 2 days
on 1 core, hours on 20. Both runs are resumable per planet (CSV checkpoint, as in
`v2/retrieve.py`).

New scripts: `v3/identifiability.py` (the Δχ² test; modes `--population pop1|mcs`,
`--noise tier1|tier2|tier3`, `--departure none|haze3e7|spots20|cloud1e2|...`) and
`v3/identifiability_calibrate.py` (the 200-planet retrievals and the Δχ²–posterior mapping).

### B3. Outputs

1. **Prevalence of undecidable labels.** For each molecule and threshold, the fraction of
   POP-I (and of the known targets) whose label is not identifiable at Tier-1 precision;
   also at Tier-2 and Tier-3 counts, which connects to Radica et al. 2026 (H₂O and CO₂
   constrained to ≲ 1 dex at Tier 1 for a hot Saturn) and to Mugnai et al. 2021's one
   retrieved planet (CH₄ at 10⁻³·¹ gave P(CH₄ > 10⁻⁴) = 48 % at Tier 1: that planet was
   undecidable).
2. **The ceiling.** The Bayes-optimal accuracy implied by the undecidable fraction, against
   the classifier's clean accuracy (81.1 %). The share of the classifier's *clean* errors
   that are irrecoverable is the first number of the paper's new spine.
3. **Error classes under each departure (E4).** Re-run the test with the departure's physics
   in the counterfactual model (the haze, the spots, the deck). Each classifier error on a
   departed spectrum is then recoverable (Δχ² above threshold: the information survived)
   or irrecoverable (below). Expected, and to be pre-registered: haze errors mostly
   recoverable (the ceiling analysis already found 87 % reducible), noise errors mostly
   irrecoverable (13–21 % reducible), spots in between. If the two methods (retraining
   ceiling vs identifiability) agree, the ceilings of Section 3.2 gain a classifier-free
   foundation; if they disagree, that is a result too.
4. **Which bins carry the information.** For undecidable planets, the Δχ² contribution per
   channel shows where the information would have been; for decidable planets the classifier
   got wrong, it shows which bins it ignored. This is the identifiability analogue of the
   input-variant test of Section 3.3.
5. **WASP-39 b.** Its own identifiability at Tier 1 for each molecule, from the consensus
   composition: is CO₂ at log −4.5 decidable at Tier-1 precision? This is the number that says
   whether the ambiguous external label was ever decidable by Ariel.

### B4. Kill criteria

If the Δχ² test and the retrieval posterior do not agree on the subset (rank correlation
< 0.7 between Δχ² and logit P), the cheap test is not usable and the identifiability
result is reported from retrievals on the subset only, with wider error bars.

---

## 4. Work package C: a diagnostic with measurable value

### C1. Candidates, chosen before the runs

The five existing rules (margin, ensemble spread, Mahalanobis, k-NN distance, PCA
reconstruction) all had negative credit under haze, spots and the high deck (Table 3). Four
candidates target the two error classes directly; the first two cost nothing at deployment.

| Candidate | What it flags | Mechanism it rests on | Class it targets |
| :-- | :-- | :-- | :-- |
| **(a) Input-subset disagreement** | planets where the all-input, no-photometric and AIRS-only classifiers disagree | the haze/spot failure enters through the optical points (Section 3.3); the AIRS-only classifier does not see it | recoverable |
| **(b) Optical-excess score** | a one-dimensional novelty score: the three photometric points relative to the NIRSpec continuum, against the training distribution | same mechanism; a 1-D score on the known weak point instead of a 104-D Mahalanobis distance that flagged shift but not error | recoverable |
| **(c) Band-metric consistency** | disagreement between the classifier's verdict and Mugnai's M_mol on the AIRS bands | the consortium's own non-ML method as a check | recoverable |
| **(d) Predicted decidability** | planets whose label is predicted undecidable at their Tier-1 noise, from a cheap regressor trained on B3's Δχ² (inputs: per-channel signal-to-noise, scale height, equilibrium temperature, transit count) | identifiability | irrecoverable |

The two-class diagnostic is (a) or (b) for recoverable errors, routed to a fallback (the
AIRS-only classifier, or a retrieval), and (d) for irrecoverable ones, routed to abstention
("needs Tier 2 to decide"). Which of (a)/(b)/(c) is used is decided by selective credit on
the clean and randomized validation planets *before* the test cases are scored.

### C2. Evaluation: the measurable value

1. **Selective credit**, the paper's existing framework: accepted-planet accuracy above the
   clean selective baseline at equal coverage. The existing rules scored −3 to −26 under the
   aerosol departures; the diagnostic must score positive there to count as a remedy.
2. **Error-class separation**: AUROC of the diagnostic for separating recoverable from
   irrecoverable errors (labels from B3), per departure.
3. **The mission-level selection simulation (R2).** A Tier-1 population of 1,000 planets
   (POP-I rescaled, and the known targets) and a Tier-2 budget of 600 slots (Edwards &
   Tinetti 2022). Selection rule: rank by the classifier's probability that the molecule of
   interest is present (CH₄ and H₂O separately, then jointly). Score: valuable targets lost
   (true positives not selected) and slots wasted (false positives selected), under clean
   spectra and under each departure. Strategies compared at the same budget:
   - classifier alone;
   - classifier + each existing rule (declined planets go to the back of the queue);
   - classifier + the new diagnostic (flagged-recoverable planets re-scored by the fallback;
     flagged-irrecoverable planets ranked by their decidability, i.e., sent to Tier 2 as
     "undecided" rather than dismissed);
   - the band metric alone (Mugnai's method);
   - retrieval on every planet (the gold standard, with its cost in CPU-hours stated).
   Asymmetric cost: a lost target counts *w* times a wasted slot, reported for *w* = 1, 3,
   10, with the ranking of strategies shown to be stable or not across *w*.
4. **On the real spectra.** Whether the diagnostic fires on WASP-39 b (and the panel), and
   whether it was right to.

### C3. Expectations, to be committed before the run

1. (a) or (b) has positive credit under haze at 3 × 10⁷ m⁻³ and spots 20 % (the existing
   rules: −17.5 to −25.8 and −6.5 to −11.0).
2. (d) separates recoverable from irrecoverable errors with AUROC ≥ 0.75 on clean spectra.
3. In the selection simulation under haze, classifier + diagnostic loses fewer than half
   the CH₄-bearing targets that the classifier alone loses, at *w* = 3.
4. Neither (a) nor (b) helps against extra noise (irrecoverable by construction); (d) does,
   by abstaining.

### C4. Kill criterion

If no candidate reaches positive credit under any aerosol departure, the paper reports the
diagnostic negatively ("no cheap rule separates the classes at Tier 1") and keeps the
identifiability and error-class results as the contribution. That is still a publishable
answer to E3, provided the selection simulation shows what the mission loses without one.

---

## 5. Work package D: manuscript changes

1. **Title and framing** per Section 1. Abstract rewritten around the two classes, the
   prevalence numbers, the WASP-39 b result and the diagnostic's value in targets saved.
2. **Introduction** adds: the sim-to-real precedents on WASP-39 b (Forestano 2025;
   Chakravertty & Raj 2026); the identifiability literature for Ariel (Mugnai 2021 §4.3,
   Bocchieri et al. 2023, Radica et al. 2026); the failure-detection result of Jaeger et al.
   2023 as the reason confidence rules were expected to fail.
3. **Methods** adds A2 (the compatibility procedure, with the round-trip test), B1 (the
   identifiability test and its calibration), C1 (the diagnostic), C2.3 (the selection
   simulation).
4. **Results** reorganized: 3.1 reproduction (kept, shorter); 3.2 the departure losses by
   error class; 3.3 the haze mechanism (kept); 3.4 identifiability and prevalence; 3.5
   WASP-39 b (and the panel); 3.6 the diagnostic and the selection simulation; 3.7
   expectations and outcomes (the pre-registration table, extended); 3.8 limitations.
5. **Cut or compress**: the carbon-rich classifier (one paragraph in 3.2 or Appendix B);
   the decline rules (one paragraph in 3.6, as the baseline the diagnostic beats or does
   not).
6. **Attribution (R3)**: "the published design as re-implemented" throughout; never "the
   consortium classifiers" when the claim is about weaknesses.
7. **Journal format**: JHSS original article; the author guidelines set no word limit
   (`jhss/jhss_for_authors.txt`), ≥ 10 keywords, figures submitted separately. Target
   ≈ 7,000–8,000 words, 6 figures, 4 tables. Supplementary directory `ariel_tier1_trust/`
   gains `results/identifiability/`, `results/wasp39/`, `results/diagnostic/`.
8. **References**: Carter & May 2024; Powell 2024; Rustamkulov, Alderson, Ahrer, Feinstein
   2023; Tsai 2023; Roy-Perez 2026; Forestano 2025; Chakravertty & Raj 2026; Radica 2026;
   Bocchieri 2023; Edwards 2019 and 2022 (already cited); Jaeger 2023 (already cited).

---

## 6. Order of work, dependencies, compute

**Dependency to resolve first.** The rendered populations (`v3/data/alfnoor_faithful/`), the
MCS parquet (`v3/data/mcs/`) and the ExoSim noise curves live on the machine that ran the
note; `v3/data/` is git-ignored and empty in this clone. Either run the new work there, or
regenerate here (`python alfnoor_faithful.py --render --jobs 8` is minutes; the MCS set and the
ExoSim grid are the slow parts).

| Week | Work | Compute |
| :-- | :-- | :-- |
| 1 | Reply to the editor (done). Download the Zenodo records. `jwst_to_ariel.py`: binning, coverage map, round-trip test (A2.7). Commit the expectations of A5, B3.3 and C3 as `REVISION_EXPECTATIONS.md`. | minutes |
| 2 | `alfnoor_wasp39.py`: verdicts under the three regimes, input variants, band metric, reduction sensitivity (A4). Launch `identifiability.py` on POP-I at Tier 1 (B2). Launch the 200-planet calibration retrievals. | 1–2 days background |
| 3 | B3 outputs 1–2 and 5. Launch identifiability under the departures (haze, spots, deck, noise ×2). Implement candidates (a)–(d). | 2–3 days background |
| 4 | B3.3 error classes. C2.1–C2.2 scoring; C2.3 selection simulation. Panel planets through the A2 pipeline. | hours |
| 5 | B on the known targets (prevalence under the mission population). Figures: coverage/round-trip; identifiability map; error-class bars per departure; WASP-39 b verdicts; selection curves. | hours |
| 6–7 | Manuscript (D), supplementary update (`ariel_tier1_trust/collect.py`), resubmission. | — |

Everything before the manuscript is roughly five weeks of calendar time with the compute in
the background; the identifiability runs are the long pole and start in week 2.

---

## 7. Risks and how each is handled

| Risk | Handling |
| :-- | :-- |
| The CO₂ label on WASP-39 b is ambiguous | Scored only at 10⁻⁵ and 10⁻³; the ambiguity is itself reported, and B3.5 asks whether Ariel could have decided it |
| Verdict depends on which JWST reduction is used | Run every reduction; report the spread; the benchmark spectrum is the headline |
| PRISM saturation gap (0.8–2.3 µm) | Covered by NIRISS SOSS; the coverage map shows it |
| 5.3–7.8 µm comes only from MIRI, with large errors | At Tier-1 noise these bins are noise-dominated anyway; at JWST precision the verdict is re-run with those bins model-filled, to show they do not drive it |
| Absorbers in the real spectrum absent from training (SO₂, CO, Na, K) | Mask test (A7); reported as an omitted-absorber departure met in real data |
| The Δχ² test disagrees with retrieval posteriors | Kill criterion B4: fall back to retrievals on the subset |
| No diagnostic earns positive credit | Kill criterion C4: report it, keep the selection simulation as the measure of what is lost |
| The classifier calls H₂O absent on WASP-39 b | Not a failure of the study: it is the haze mechanism observed on a real planet, and the input variants show whether it is recoverable |
| Reviewer still finds the result "expected" | The non-obvious claims are quantitative: the prevalence of undecidable labels at Tier 1, the share of each departure's loss that is irrecoverable, the targets saved at a fixed budget, and whether a real spectrum reproduces the haze mode |
