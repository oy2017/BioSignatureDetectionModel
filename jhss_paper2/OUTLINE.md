# Paper 2: retrieval-derived labels for a synthetic-trained triage classifier

**Venue: undecided.** JHSS is one candidate (its referee described this study), but the
choice is deferred until the results exist. Draft blind: no author names or affiliation
in any file until a venue is chosen and the submission is being assembled.

Outline written 2026-09-08. Companion to paper 1 (`../nhsjs/`, submitted to NHSJS). Paper 2
reuses paper 1's regenerated grid, noise model, and frozen classifier, and contains
every retrieval result; paper 1 contains none. Nothing here starts until paper 1's
Stage 3 has frozen the classifier.

## Why JHSS is a candidate venue

JHSS rejected the R = 200 benchmark on 2026-09-08 for insufficient novelty and said that
retrieval-derived labels, realistic channel-level spectra, and physically consistent
atmospheres "would amount to requesting a fundamentally new study". Paper 2 is that
study on two of the three counts. Photochemical self-consistency is still absent and must
be stated in the first page as the known gap; it is the most likely reviewer objection.

The retrieval work has never been published. The 50-planet pilot appears only in the
non-archival ML4PS 2026 workshop draft; the JHSS manuscript mentioned it as prototyping
in the response letter, without numbers.

## Research question

Can a classifier trained on synthetic Ariel-configured spectra stand in for Bayesian
atmospheric retrieval when ranking CH₄–O₃ candidates, when does the proxy fail, and does
training on retrieval posteriors instead of injected thresholds make it a better proxy?

## Hypotheses

- **P1.** Retrieval-derived labels will agree with injected labels on most planets, with
  disagreements concentrated near the thresholds and at low feature amplitude, and the
  disagreement rate will rise once nuisance parameters and a cloud are freed.
- **P2.** The threshold-trained classifier's calibrated probability will track the
  retrieval posterior P(+) across the test population.
- **P3.** Agreement with retrieval will degrade under domain shift faster than accuracy
  against injected labels, and retrieval itself will absorb some shifts (a cloud deck)
  that the classifier cannot.
- **P4.** A classifier trained on posterior P(+) as a soft label will predict retrieval
  outcomes better than the threshold-trained one, at the same inference cost.
- **P5.** Retrieving with an independent forward model (Exo-Transmit) will widen the
  label disagreement, bounding the forward-model-mismatch contribution.

## Experiments

| Stage | What | Compute (22 cores) |
| :-- | :-- | :-- |
| R0 | Diagnose the pilot discrepancy: 12 flips in the July 28 trial vs 3 in the July 29 balanced run. Reproduce both on ten planets; identify the cause (error bar, sample, live points) before anything scales | half a day |
| R1 | Regression test: the 50 pilot planets' parameters re-rendered at the Ariel configuration, retrieved with the new code, compared with the pilot | one hour |
| R2 | Test-set retrievals: stratified 2,000 planets from paper 1's pooled test sets, seed 42; nine free parameters (six gases, T, radius, grey cloud-top pressure); uniform priors; nestle multi-ellipsoid, 100 live points; hard cap on likelihood calls per planet so the 8-hour tail seen in the pilot cannot stall a worker; checkpoint per planet to CSV | about three days |
| R3 | Retrievals under shift: the same 500 planets re-rendered under the 10⁴ Pa cloud deck, and under the ExoMol opacity swap; retrieval with the training opacities (mismatch) | about one day each |
| R4 | Forward-model mismatch: 300 planets rendered with Exo-Transmit, retrieved with MultiREx | half a day |
| R5 | Training-set retrievals for the emulator: 5,000 stratified training planets | about one week |
| R6 | Emulator: XGBoost trained on P(+) (regression, and as a soft label); compared with the threshold-trained classifier on agreement with retrieval, in domain and under R3's shifts | hours |

Runtime basis from the pilot: median 14 min, 90th percentile 48 min, maximum 463 min
per planet with six free parameters; expect roughly double with nine.

## Exhibits (target 6 figures, 3 tables; check the chosen venue’s limits)

1. Workflow: paper 1's frozen classifier → retrieval on the same planets → comparison.
2. Retrieval label vs injected label: agreement vs margin to threshold, fixed vs freed
   nuisance parameters.
3. Classifier probability vs retrieval posterior, 2,000 planets, density plot with r.
4. Agreement under shift: clean, cloud deck, opacity swap, Exo-Transmit, for both
   classifier-vs-retrieval and retrieval-vs-truth.
5. Emulator vs threshold-trained classifier: agreement with retrieval, calibration
   against P(+), in domain and under shift.
6. Cost: retrieval CPU-hours vs classifier milliseconds, per planet, with the accuracy
   each buys.

Tables: retrieval configuration and priors; agreement matrix (truth / retrieval /
classifier / emulator); shift results.

## Overlap and disclosure

- Paper 2 cites paper 1 for the grid, noise model, and classifier, as "submitted to
  NHSJS" or published, whichever holds at the time.
- Both submission forms state that the other manuscript exists and asks a different
  question on shared synthetic data.
- No result appears in both. Paper 1's threshold-sensitivity and margin analyses are
  against injected labels; paper 2's are against retrieval labels.

## Open questions before starting

- Whether to free the stellar parameters in retrieval (they are observationally
  constrained in practice; fixing them is defensible, and it keeps runtime down).
- Whether the emulator (R5–R6) is in scope for the first submission or held for a
  revision; R5 is the single largest compute item.
- Venue: decide once R2 has run. JHSS asked for exactly this study but rejected paper 1's
  predecessor; NHSJS would take it as a second paper; a workshop is the low-risk option.
