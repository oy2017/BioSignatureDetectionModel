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
