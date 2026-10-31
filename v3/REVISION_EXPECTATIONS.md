# Expectations for the revision, written before any run

Written 2026-10-16, before the JWST spectra were downloaded and before any identifiability or
diagnostic code existed. Companion to `REVISION_PLAN.md`; same practice as
`alfnoor_trust_expectations.md` and `PREREGISTERED_PREDICTIONS.md`: each expectation is scored
met or not met in the results, and misses are reported as misses. Labels follow the classifier's
convention (abundance above 10⁻⁴ unless stated); "the classifier" is the consortium design as
re-implemented, averaged over its four learners, as in the note.

## A. External validation on JWST spectra

### WASP-39 b (Carter & May et al. 2024 benchmark + Powell et al. 2024 MIRI)

A1. At JWST precision, all four learners report H₂O present, CH₄ absent and NH₃ absent at 10⁻⁴;
    the CO₂ verdict is split across learners (the published abundance straddles 10⁻⁴).
A2. Under Ariel Tier-1 noise at WASP-39 b's one transit (1,000 draws), the H₂O detection
    frequency exceeds 80 %, the CH₄ and NH₃ false-positive frequencies are below 20 %, and the
    CO₂ detection frequency lies between 30 and 70 %.
A3. Risk stated as a prediction: the real optical points (gray cloud, Na and K wings, optical
    slope) push at least one learner with all 104 inputs to call H₂O absent, and the variant
    without the photometric points does not. This is the haze failure mode of Section 3.3 seen
    in real data.
A4. The Mugnai band metric agrees with the classifier on H₂O (present) and CH₄ (absent).
A5. The H₂O and CH₄ verdicts are the same for every published reduction (Feinstein, Ahrer,
    Alderson, Rustamkulov, Carter benchmark, and the three MIRI reductions); the CO₂ verdict
    is not.
A6. The round-trip test (the benchmark model sampled at the JWST grids, noised, converted)
    changes no learner's verdict relative to the directly binned model, and changes the mean
    learner probability by less than 0.05.
A7. Masking the bins that carry absorbers absent from the training grid (SO₂ at 4.05 µm, CO at
    4.6 µm, Na and K in the optical) changes no verdict at JWST precision.

### The panel (labels from the survey of 2026-10-15; "amb." is not scored)

| Planet | CH₄ | H₂O | CO₂ | NH₃ | Expected verdicts at JWST precision | Expected trouble |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| TOI-270 d | present | amb. | present | absent | CH₄ present, CO₂ present, NH₃ absent | none: clear, quiet star |
| K2-18 b | present | absent | amb. | absent | CH₄ present, H₂O absent, NH₃ absent | haze + spots: all-input learners may miss CH₄ (A3 mode) |
| TOI-1231 b | present | amb. | present | amb. | CH₄ present, CO₂ present | clouds + haze + spot crossings: same risk |
| WASP-166 b | absent | present | present | amb. | CH₄ absent, H₂O present, CO₂ present | none expected |
| TOI-421 b | absent | amb. | absent | — | CH₄ absent, CO₂ absent | near-featureless: learners may call everything absent, which is correct here |
| HAT-P-18 b | absent | amb. | absent | absent | CH₄ absent, CO₂ absent, NH₃ absent | unocculted spots at 5.8 %: the all-input learners' verdicts move when the optical points are removed |
| WASP-107 b (AIRS only) | absent | present | amb. | absent | CH₄ absent, H₂O present, NH₃ absent | partial coverage; filled bins flagged |
| HD 189733 b (AIRS only) | absent | amb. | absent | absent | CH₄ absent, CO₂ absent, NH₃ absent | partial coverage |
| HAT-P-26 b (AIRS only) | absent | present | present | absent | CH₄ absent, H₂O present, CO₂ present, NH₃ absent | partial coverage; SO₂ present |
| GJ 3470 b (AIRS only) | amb. | present | present | absent | H₂O present, CO₂ present, NH₃ absent | partial coverage |

P1. Over the scored planet–molecule labels of the panel (about 30), the classifier at JWST
    precision agrees with at least 75 % (its clean in-simulator accuracy is 81 %).
P2. Its accuracy on the CH₄ positives (three planets) is lower than on the CH₄ negatives, and
    every CH₄ positive missed is recovered by the no-photometric or AIRS-only variant: the
    misses are the recoverable class.
P3. The three CH₄ positives (temperate sub-Neptunes around M dwarfs, high mean molecular
    weight) are flagged by the Mahalanobis and k-NN distance scores as outside the training
    distribution (above the threshold that declines 10 % of clean planets), while the hot
    giants (WASP-39 b, WASP-166 b, HAT-P-18 b, HD 189733 b) are not.
P4. HAT-P-18 b: removing the photometric points changes at least one learner's verdict on at
    least one molecule (spot contamination enters through the optical points).
P5. Under Ariel Tier-1 noise at each planet's MCS transit count, the detection frequency of a
    present molecule is above 50 % for the hot giants and below 50 % for at least one of the
    temperate sub-Neptunes (their Tier-1 noise is large).
P6. The NH₃ classifier produces no false positive on any panel planet at JWST precision.

## B. Identifiability

B1. The cheap Δχ² test and the nested-sampling posterior agree: Spearman correlation between
    Δχ² and logit P(X > 10⁻⁴) above 0.7 on the 200-planet calibration subset, for every
    molecule. (Kill criterion in the plan if not met.)
B2. On POP-I at Tier-1 noise, the fraction of planets whose label is undecidable at the
    calibrated 95 % level is between 10 and 30 % for CH₄ and CO₂, and between 20 and 40 % for
    H₂O and NH₃ (H₂O was the hardest molecule in the reproduction; NH₃ overlaps H₂O).
B3. The implied ceiling on accuracy is at least 8 points above the classifier's clean accuracy
    (81.1 %), so most of its clean errors are recoverable, not irrecoverable.
B4. Under haze at 3 × 10⁷ m⁻³, at least 70 % of the classifier's errors fall in the recoverable
    class (the retraining ceiling found 87 % of the loss reducible); under noise × 3, at most
    30 % do (13 % reducible); spots 20 % in between.
B5. The two methods of locating the information (the Δχ² contribution per channel and the
    input-variant test) agree that the molecular information of hazy planets sits in the AIRS
    bins.
B6. WASP-39 b at Tier-1 precision: H₂O at 10⁻⁴ is decidable; CO₂ at 10⁻⁴ is not (its abundance
    is within a decade of the threshold); CH₄ absence at 10⁻⁴ is decidable.
B7. Undecidable fractions fall with tier: Tier-2 counts halve them relative to Tier 1 for every
    molecule.

## C. The diagnostic

C1. Candidate (a), input-subset disagreement, or (b), the optical-excess score, has positive
    selective credit under haze 3 × 10⁷ m⁻³ and spots 20 % at the threshold that declines 10 % of
    clean planets (the five existing rules: −6.5 to −25.8).
C2. Candidate (d), predicted decidability, separates recoverable from irrecoverable errors on
    clean spectra with AUROC ≥ 0.75.
C3. In the Tier-2 selection simulation (600 slots of 1,000) under haze, classifier + diagnostic
    loses fewer than half the CH₄-bearing targets that the classifier alone loses, at a
    lost-target weight of 3.
C4. Neither (a) nor (b) has positive credit under noise × 2 or × 3 (nothing to recover); (d) does,
    by abstaining.
C5. The diagnostic fires on WASP-39 b for at least one learner if and only if A3 occurs, and
    fires on HAT-P-18 b (spots) and K2-18 b (haze); it does not fire on WASP-166 b.
C6. Across the lost-target weights 1, 3 and 10, the ranking of strategies in the selection
    simulation is unchanged.

## F. The amplitude flag through each planet's coverage, and physically confined hazes

Added 2026-10-30 after the four-reviewer pass, before either run (`amplitude_flag.py`, `haze_confined.py`). Earlier versions of the
flag were chosen after its panel values were seen; this rule is fixed here so that no further choice is made from the panel.

Rule. A reduction's ratio is the catalogue's five-scale-height modulation over the noise-corrected dispersion of its measured bins.
Its upper threshold is the largest ratio of any clear POP-I planet through that reduction's measured bins, and its lower cut is the first
percentile of POP-I with its own decks through the same bins. Every reduction with a measured dispersion gets a ratio, including the two without a
clear twin. Flagged planets go to the design retrained with haze; the rest take the shape flag. The 99th-percentile and AIRS-range forms are
reported as sensitivities and are not candidates to replace the rule.

F1. At least seven of the ten giants the census places beyond the prior stay flagged. At least one of the four planets within 0.25 of the old threshold drops out.
F2. The fixed-denominator panel score stays within 2 labels of the 53.5 of 66 scored with the full-bin threshold.
F3. The 95 % planet-bootstrap interval against AIRS-only everywhere still includes zero; the interval against the published classifier still excludes it.

Hazes: Lee-Mie 3 × 10⁷ m⁻³ below 1 Pa; power law at constant mixing ratio with slope 4 and log₁₀ σ₀(1 µm) of -29 and -28 (about 10³ and 10⁴ times H₂ Rayleigh).

G1. The optical-point mechanism survives: under each haze the all-input design loses more accuracy than the no-optical design, which loses more than the AIRS-only one.
G2. The all-input loss is smaller than the 17.7 points of the column-filling haze, between 5 and 15 points at the weakest of the three.
G3. The optical flag fires on fewer planets than the 100 % of the column-filling haze at the same clean threshold, between 40 and 95 % at the weakest.
G4. The AIRS-only design stays within 2 points of its clean accuracy under each haze.
G5. The design retrained on the column-filling haze recovers at least half of the all-input loss under each confined haze.

## H. The independent test of the frozen routing method (written 2026-10-30, before any fourth-batch label, spectrum or verdict was seen by the code or by me)

The rule under test is the one of section F (commit b09b253) as implemented at the commit that adds this section: the amplitude flag of `amplitude_flag.py`, with each new
reduction's upper threshold (largest ratio of any clear POP-I planet through its bins) and lower cut (first percentile of POP-I with its own decks through its bins); the shape flag at
the clean thresholds of `alfnoor_diagnostic.py`; the haze-retrained and AIRS-only learners already trained; a majority of four learners; right 1, split or withheld 0.5, wrong 0. Nothing is
retrained or re-thresholded on the fourth batch. New planets enter only through `jwst_panel.LABELS`, `jwst_panel.segments` and the catalogue. Labels come from the papers by the rule of the
manuscript's Section 2.6 and are committed before any verdict is computed. The batch is the set of planets the survey finds qualifying, all of them, whatever they turn out to do; planets
rejected are listed with the reason. The fourth-batch score is reported whichever way it goes.

On the 66 labels the method was built on, it gained 0.13 labels per label over the published classifier at JWST precision (8.5 of 66) and 0.04 at Tier-1 noise (2.7 of 66), and was not
separable from the AIRS-only input (+1.5, interval -1.5 to 5.7) or the haze+deck-retrained design (+2.0, -2.2 to 5.9). Expected on the fourth batch:

- H1. At JWST precision the method scores above the published classifier (difference > 0).
- H2. At JWST precision the method is within two labels, either way, of the AIRS-only input and of the haze+deck-retrained design.
- H3. The gain over the published classifier at Tier-1 noise is smaller than at JWST precision.
- H4. The method moves at most one label from right to wrong against the published classifier.
- H5. The amplitude flag fires on at least half of the giants among the new planets (planets above 0.1 Jupiter masses with a published H2/He-dominated atmosphere).
- H6. Any new methane-present label on an H2-dominated planet is called absent by the all-input design and present by the AIRS-only input.

## I. A method built for each reduction (written 2026-10-30, after the fourth batch was seen; no verdict of this design computed yet)

Reason: the frozen method fills unmeasured bins with a flat value and feeds learners trained on full-coverage spectra at another noise; five of the eight new present labels fail the mask control, so those learners could not read even a clear twin through that mask. This design removes that step. All 35 planets have been seen, so scores here are development scores; only unseen spectra or an independent simulator can confirm a gain.

Rule, fixed before the first run. For each reduction, four learners of the same kinds and settings per molecule, trained on the POP-III natives (the faithful and haze+deck sets pooled) binned to the same 104 bins, cut to that reduction's measured bins only with no fill, each spectrum's depth deviations rescaled by the ratio of the planet's five-scale-height modulation to the training planet's, with the reduction's own per-bin errors added as noise (in quadrature with the Tier-1 sigma at Tier-1). Label above 1e-4; majority of four; right 1, split 0.5, wrong 0. "matched" uses the z-score of the published design. "matched+amp" also draws a muting factor log-uniform in 0.1 to 1 on half the training rows, and gives the learners the logarithm of the spectrum's dispersion over its noise as one more input. matched+amp is the primary design.

Expected:
- I1. matched+amp finds more present labels than the frozen method on the 35 planets at JWST precision.
- I2. It raises the score on the fourth batch above 16.0 of 24 at JWST precision, with no more false present calls than the frozen method's 2.5 lost there.
- I3. Its AUROC at Tier-1 on the fourth batch exceeds the frozen method's 0.60.

Added before its first run, the same design for Ariel, where coverage is complete and one set of learners serves every planet (`matched_ariel.py`): the pooled natives at the Tier-1 noise of each training planet, the muting factor on half the rows, and the dispersion-over-noise input; scored as `pipeline_sim.py` on POP-I.
- I4. Clean accuracy within 3 points of the published design (81.2), and above it under haze3e7 by at least 5 points and under mute3.

## J. Confirmation of the per-reduction design on a forward model it never saw (written 2026-10-30, before any verdict on this grid)

Design under test: matched+amp of section I, unchanged. Comparators: published, AIRS-only, shape flag, haze+deck-retrained and the frozen method. Data: the Goyal et al. (2019) generic ATMO grid, rainout condensation (36,864 models from a different code, opacities and aerosol scheme; local condensation as a later sensitivity test). For each of the 35 reductions, 200 models drawn at random with grid temperature within 200 K of the planet's and the grid gravity nearest the planet's, rescaled to the planet by the grid's published scaling, binned to the 104 bins, passed through the reduction's mask with its real errors added (and the Tier-1 sigma at Tier-1). Truth: log10 of the mole fraction averaged over 0.1 to 100 mbar, above -4. Learners and thresholds as trained on the real run; nothing is refitted to this grid.

Primary measure: balanced credit, the mean of the credit on present and on absent labels, with right 1, split or withheld 0.5, wrong 0, over all molecules; 95 % bootstrap over planets. Expected:
- J1. matched+amp above the frozen method at JWST precision, interval excluding zero.
- J2. matched+amp above the frozen method at Tier-1, interval excluding zero.
- J3. matched+amp above AIRS-only and above haze+deck-retrained at JWST precision.
- J4. matched+amp's false-present rate no higher than the frozen method's at JWST precision.
Sensitivities fixed now: labels within 0.5 dex of the threshold removed; each molecule alone. These grid abundances are equilibrium values tied to temperature, so this tests a different forward model, not real atmospheres.

## K. Real spectra unseen by any design (written 2026-10-30, before any verdict on them)

Design under test: matched+amp of section I, unchanged, against the frozen method and the other designs. Spectra and labels: `independent_test.py` (LABELS), from `independent_data_survey.md`; provenance in `results/independent_test/SOURCES.md`. 17 labels on 12 planets (7 present, 10 absent): V1298 Tau c JWST (H2O present; CH4, CO2, NH3 absent), HAT-P-14 b JWST (CH4, CO2 absent), and HST G141 water on 11 planets. LTT 9779 b is not scored: its absent label and a companion paper's 3.1-sigma water disagree. Scored at the observed precision only; right 1, split or withheld 0.5, wrong 0. The HST labels are mostly TauREx retrievals of one water band, and absent there mostly means a flat spectrum, so the HST part tests water at G141 coverage and nothing more; the frozen designs must see it through a flat fill of about nine tenths of their bins, which is the handicap the new design was built to remove.
- K1. matched+amp scores above the frozen method on the 17 labels.
- K2. matched+amp's balanced credit is above 0.5 on the HST labels.
- K3. It finds at least 4 of the 7 present labels with at most 3 of the 10 absent labels wrong.

## L. Second round: a broader training population (written 2026-10-30, before its first run)

Sections I to K measured matched+amp. On the rainout grid it beat the frozen method at both noise levels, while on the 17 unseen real labels it did not clearly beat it, and absolute recall stayed low (0.40). The training population has CO switched off, no gas above 1e-2 and a single mean molecular weight near hydrogen, all of which real spectra and the grid violate. This round changes the population.

Development data: the rainout grid only, by the balanced credit of section J. The real spectra are not consulted for any choice in this round, though they have been seen; choices cannot be blind to what the earlier scores suggested. Candidates, in order: B, 8,000 new spectra of POP-III planets with CH4, H2O, CO2, NH3 and CO each log-uniform in -9 to -1, a third clear, a third with a Lee-Mie haze, a third with a grey deck at 5 to 500 Pa; B+bal, the same with classes balanced in training; B+bal+ens, adding a vote from the haze+deck-retrained design. The candidate with the highest balanced credit at Tier-1 on rainout is frozen and committed, then scored once on data no design has seen: the local-condensation grid, and the photochemical Crossfield et al. (2025) grid. The 107 real labels are then reported as they fall, as descriptive results.

## M. Corrections from the pipeline audit, and the third round (written 2026-10-30, before any of these ran)

An audit of the data path found errors. They are corrected here, for every design, frozen ones included, and every score is recomputed; earlier scores remain in the git history. The planets on which they were noticed are no longer unseen for any purpose.

Corrections. (1) `jwst_to_ariel.align_offsets` gave NRS1 and NRS2 separate offsets although the two detectors of one grating do not overlap, which put an artificial step between them; on TOI-270 d it reversed the 3.3 um CH4 contrast (+87 ppm in the published file, -110 after alignment; +83 in the independent Holmberg reduction). The detectors of a grating now share one offset. (2) WASP-107 b NIRCam and MIRI files give half-widths, read as full widths, which dropped 13 bins. (3) The Felix TOI-270 d files' fourth column is an instrument profile width, not a bin width, and is no longer read as one. (4) HIP 67522 b has 186 Earth masses in the catalogue; Thao et al. (2024) give 13.8, and `mcs_corrections.py` corrects it. (5) K2-18 b H2O in the Madhusudhan and Schmidt reductions were labelled absent on limits above 1e-4 (-3.06 and -2.44), against the stated rule; both labels are withdrawn, leaving 88 scored labels. Other findings were checked and left unchanged, GJ 3470 b NH3 among them (its 1-sigma upper bound -4.52 meets the rule).

Expectation for corrections alone: small. About +1 present label on the real panel, TOI-270 d CH4; HIP 67522 b CO2 absent may be recovered by the amplitude flag.

Third-round design, from two audit findings: in the training populations the target gas is usually the most abundant of the four because draws are independent, unlike the grid; and hard votes on an imbalanced population set a threshold nobody chose (grid CO2: AUROC .86, recall .18). Population C: 16,000 POP-III planets with temperature log-uniform in 250 to 2500 K; composition from FastChem (`generate_grid.Chemistry`), metallicity log-uniform -1 to +3 dex solar, C/O uniform 0.2 to 1.5, half equilibrium and half quenched; every gas then scattered by a normal of 0 to 1.5 dex (width uniform per planet), with a 15 % chance per gas of being replaced by a log-uniform draw in -9 to -1, standing for photochemistry and other disequilibrium; abundances capped at -0.3 and CO retained; a third clear, a third with a Lee-Mie haze, a third with a grey deck at 5 to 500 Pa. Mean molecular weight then follows from the composition.

Candidates: population C or C plus B, each with (a) the four-learner vote or (b) five multi-label neural networks with early stopping, each trained with three fresh noise and muting draws per row, averaged. Candidates of kind (b) use one threshold per molecule and reduction, set on held-out simulated spectra at that reduction's noise: the threshold giving the highest balanced accuracy among those with a simulated false-present rate of at most 10 %. Selection: highest mean of the balanced credit at JWST precision and at Tier-1 on the rainout grid, with the corrected masks. No real label is consulted for selection.

The choice is committed, then scored once on the local-condensation grid and on the photochemical Crossfield et al. (2025) grid, which no design has seen. Real labels are reported as they fall; they have been seen and cannot confirm. Chemistry in C makes verdicts partly follow correlations between gases, so a grid built on equilibrium chemistry may flatter it: the photochemical grid is the fairer test, and the false-present rate on labels outside a molecule's correlated range, and the band-masked control, are reported beside the score.

Expectations: on rainout, balanced credit above .70 at JWST precision and .65 at Tier-1 (matched+amp: .661, .597), with false-present no more than .15. M1: on local condensation it beats the frozen method at both precisions with the interval excluding 0. M2: on Crossfield it beats the frozen method at Tier-1 with the interval excluding 0; I rate this less likely than not to hold at the rainout size. M3: on the real labels, present credit rises above the method's without absent credit falling more than 3 labels. If M1 or M2 fails, the paper says so.

Addendum to M (written 2026-10-30, after the section L scores on the rainout grid with the masks before correction, before any third-round score): B+bal had the highest balanced credit of round two (JWST .762, Tier-1 .673; the method .641, .565), at false-present .139 and .157 (the method .067, .028). B+bal+ens scored lower (.712, .624). By the rule of section L, B+bal is that round's choice. It is refit on the corrected masks and joins the candidates of M under the same selection rule, with the same requirement of false-present no more than .15, which at Tier-1 it does not meet on the old masks. If the selected design exceeds .15, the paper reports that rate beside the score.

Selection under M (written 2026-10-30, before any verdict on either held-out grid): C+B-mlp is chosen, with the highest mean balanced credit on rainout, (.765 + .698) / 2 = .732, against C-mlp .724 and B+bal-vote .720. Its false-present rate is .107 at JWST precision and .109 at Tier-1. Its settings are frozen as in `round3_learner.py` at this commit. The vote candidates on C were not run: the rule's other four candidates already spanned the choice and the vote on C costs about 15 core-hours. M3 as written can already be read on the seen real labels: at JWST precision present credit rises from 22.5 to 30.0 and absent credit falls from 59.0 to 54.0, of 105 labels, so absent falls by more than the 3 labels allowed. M3 fails at JWST precision. At Tier-1, of 88 labels, present 7.2 to 17.3, absent 50.7 to 46.2, also more than 3.

Results under M, read once (written 2026-10-30). M1 met: on local condensation C+B-mlp minus the frozen method is +.137 (.110, .156) at JWST precision and +.105 (.089, .119) at Tier-1. M2 met, against my rating: on the photochemical grid +.115 (.089, .136) and +.133 (.120, .144). The false-present rate on local condensation, .173 at both precisions, exceeds the .15 limit; on the photochemical grid it is .081 and .116. All held-out verdicts are in `results/goyal_test/goyal_{local,crossfield}.txt`.

N. Two checks on that result, before they run. N1, stability: two further seeds of the whole design (networks and thresholds) on all three grids and the real labels; expectation, balanced credit moves by less than .01 and the real JWST score by no more than 2 labels of 105. N2, what the verdict reads: for each molecule the design is refit with that molecule's bands removed from the input (CH4 2.15-2.45, 3.1-3.7; H2O 1.3-1.5, 1.75-2.05, 2.5-3.0, 5.5-7; CO2 4.2-4.45; NH3 2.9-3.1, 5.8-6.5 um), at JWST precision. What survives comes from other bands and correlations between gases. Expectation: on the grids a clear part of the score survives, more for H2O and NH3 than for CH4 and CO2, since correlation is what the audit warned about; balanced credit stays above .5 for most molecules. The share of the gain over the frozen method that depends on the bands is reported with the score.

Results under N (written 2026-10-30; `results/round3/round3_checks.txt`). N1 narrowly missed: balanced credit moves by up to .011 across three seeds, the real JWST score from 83 to 86 of 105 (allowed 2); the unseen 12 and 17 planets do not move at all (19.0, 16.0). N2: on the grids most of the score survives removal of the main bands (rainout .736 of .765, local .678 of .711, photochemical .726 of .773), so most of the gain over the frozen method is not read from those bands. Removal leaves a molecule's other bands in the input, so this does not yet separate secondary bands from correlation between gases. On the real labels removal halves present credit (30.0 to 15.0 of 41; 15 lost, none gained) and leaves absent credit level.

N3, written before it runs: the photochemical grid gives each model's spectrum with one molecule's opacity removed and its chemistry unchanged (`rprs2_no_<gas>`). The frozen design, refit with the same seeds, reads these at both precisions. A "present" call on a spectrum without the gas's opacity can only come from other gases: call this the correlation rate, measured where the gas is truly above 1e-4. Reading the molecule's own absorption is the difference between the present rate on the full spectrum and the correlation rate. Expectation: correlation rate about .35 at JWST precision, against a full rate of .63, so somewhat more than half of what it calls present relies on the correlation; CH4 and CO2 lowest, H2O and NH3 highest. The result sets the wording of the claim: if the correlation rate is close to the full rate, the method is inference from chemistry rather than detection.

Result of N3 (written 2026-10-30): my expectation was wrong, in the design's favour. Where the gas is truly above 1e-4, at JWST precision, it is called present on .627 of full spectra and on .114 with that gas's opacity removed (expected .35); 82 % of present calls flip to absent. By gas, called present with opacity removed: CH4 .003 (.803 full), CO2 .152 (.682), H2O .159 (.582), NH3 .020 (.161). At Tier-1: .096 of .516. So the verdict mostly reads the molecule's own absorption, across more of the spectrum than the main bands of N2. About a fifth of present calls (a quarter for H2O and CO2) rest on other gases. This is one grid, without clouds or hazes; the same test was not run on the frozen method. `results/round3/counterfactual.txt`.

O. The design at Ariel coverage, before it runs: C+B-mlp at complete coverage, Tier-1 noise, scored on POP-I and the departures of `pipeline_sim.py`, as in section I. POP-I draws gases independently with a ceiling of -2, unlike C, and the threshold is set on C+B, so I expect clean accuracy below the published 81 %, perhaps 72 to 78 %, and a smaller loss under haze and clouds than the published 63.5 % at haze 3e7. If clean accuracy falls below 70 %, the design is not offered for Ariel's own population without retraining.

Result of O (written 2026-10-30): my expectation was wrong again, in the design's favour. On POP-I, C+B-mlp reaches 84.2 % clean (published 81.2 %; expected 72 to 78 %), 80.5 % at haze 3e7 (published 63.5 %, retrained haze+deck 73.6 %) and 74.6 % muted threefold (published 67.7 %). Recall rises most: clean 81.3 %, haze 3e7 59.2 %, against 63.8 % and 35.3 % for matched+amp. POP-I comes from the same forward model as training and was not used to select the design. `results/round3/round3_ariel.txt`.

P. Does the design's own probability mark unreliable verdicts (written 2026-10-30, before computing it)? On the held-out grids and the real labels: AUROC of the probability for the label, AUROC of the distance from the threshold for ranking errors, and accuracy on the most confident 70 %. Expectation: label AUROC above .85 on the grids and above .75 on the real labels; error-ranking AUROC near .70, higher than the .44 to .52 of the distance rules under haze but on a different shift; a gain of at least 5 points on the kept 70 %.

Q. Error classes of the design at Ariel coverage (written 2026-10-30, before computing them): by the clean-spectrum posterior of Section 2.5, on clean spectra and under the column-filling haze. Expectation: its gain over the re-implemented classifier is mostly recoverable errors removed, with the errors on undecidable labels falling little; under haze its recoverable errors are less than half of the re-implemented classifier's 675.
