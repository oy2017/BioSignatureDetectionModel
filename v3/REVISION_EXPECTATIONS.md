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
