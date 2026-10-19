# Findings of the revision, as they come in

Dated entries; each names the result file, the control that supports it, and the caveats that still
apply. Expectations are scored against `REVISION_EXPECTATIONS.md`. Everything here is preliminary
until the entry says otherwise.

## 2026-10-16: first pass over the real JWST spectra

Inputs: WASP-39 b from the Carter & May et al. 2024 benchmark and the Powell et al. 2024 MIRI spectrum
(the discovery-paper products as alternates); nine panel planets from the files listed in
`results/jwst_panel/panel_summary.txt` (TOI-1231 b and the original K2-18 b and TOI-270 d spectra
are on OSF, which is down until 2026-10-19). Conversion as in
`jwst_to_ariel.py` with modes aligned through their overlaps and uncovered bins filled flat at the
measured mean. Verdicts: `wasp39_assemble.py`, `jwst_panel.py`; controls: `jwst_controls.py`.

### F1. WASP-39 b: three learners of four are right, the band metric is right, the MLP is wrong, and the misses under Ariel noise come from the optical points

Inputs (2026-10-16, all archives in): the Carter & May et al. 2024 benchmark re-reduction of the four ERS
modes (R = 100, fixed limb darkening, PRISM dilution-corrected with the saturated 0.63–2.12 µm dropped)
plus the Powell et al. 2024 MIRI LRS spectrum (Eureka!), 0.52–7.8 µm; 103 of the 104 bins are measured
(the one 2.01–2.03 µm bin is half-covered and filled). Modes aligned through their overlaps onto NIRISS
order 1: PRISM −177 ppm, which is the PRISM offset Carter & May report from their joint fit, so the
alignment reproduces theirs; NIRCam +122, NRS1 −19, NRS2 −57; MIRI, which shares one bin with the
rest, +326 ppm through the benchmark model (left as published in the sensitivity matrix: no verdict
changes). Files: `results/wasp39/{coverage,verdicts}_benchmark.*`, `sensitivity.{txt,csv}`,
`cloud_twins.txt`; `results/jwst_panel/controls.txt`.

At JWST precision with all 104 inputs, at 10⁻⁴: H₂O present for KNN, RFC and SVC and **absent for the
MLP** (P = 0.000); CH₄ absent, CO₂ absent and NH₃ absent for all four. At 10⁻⁵ CO₂ is present for two
to three learners, where the published abundance (10⁻⁴·⁵, a decade across reductions) puts it. At
10⁻³, which the published H₂O abundance (10⁻²·⁴ to 10⁻¹·²) also exceeds, all four learners call H₂O
absent with all inputs, and the RFC and SVC call it present when the three optical points are dropped:
the note's optical-point mode, at the one threshold where the verdict is close. NH₃,
left open by the preliminary set, is settled: absent in all 24 configurations, with each of the three
MIRI reductions. The band metric agrees with the majority (M_H₂O = 4.0 against the flag at 0.8;
M_CH₄ = −1.2; M_CO₂ = −0.1).

The MLP's miss is not the data's. Over the 24 configurations of the sensitivity matrix (two product
sets, four PRISM and three MIRI reductions, two NIRISS pipelines, fitted and fixed limb darkening, offsets
on and off, model and flat fill, with and without MIRI, PRISM alone, and the bins of Na, K, SO₂ and CO
masked singly and together) the H₂O verdict is KNN-RFC-SVC present, MLP absent in 23; only the
preliminary three-mode, 15-bin-filled set gave four of four. The CH₄ and NH₃ verdicts never change;
CO₂ flips to present for the RFC in three configurations (its score sits at the 0.5 boundary). The
benchmark's own best-fit model, binned directly onto the 104 bins with no conversion at all, gets the
same MLP verdict (score 0.28, absent). The round trip (that model sampled at the JWST grids and converted)
introduces a bias of 22 ppm rms, 104 ppm at worst in a MIRI bin, and changes no verdict noise-free;
with the JWST errors, the two learners whose CO₂ score sits at the boundary cross it in 28–30 % of
realizations, and nothing else moves.

What the MLP objects to is the shape of a real hot Saturn. The clear H₂/He twin at the published
abundances is called H₂O-present by all four (MLP score 1.00), as are twins with a grey deck down to
10 Pa, whose features are half the clear twin's (MLP 0.97), and clear twins at every water abundance
from 10⁻³ to 10⁻¹·⁵. The real spectrum's dispersion over the measured bins is 324 ppm against 1,190 ppm
for the clear twin and 618 ppm for the 10-Pa deck: 3.7 times more muted than the clear twin and more
muted than any cloud the note's departure study rendered, from ten-times-solar metallicity and clouds
together. The real spectrum is nevertheless inside the training distribution by the distance rule
(10-NN distance 6.2 against the decline threshold 11.25; the twin 4.7), so the rule does not flag this
error.

Under Ariel Tier-1 noise at WASP-39 b's one transit, H₂O is detected in 80, 34, 33 and 87 % of draws
(KNN, MLP, RFC, SVC; mean 58 %) with all inputs, and in 87, 66, 94 and 98 % (mean 86 %) without the
three optical photometric points. The twin is detected in 100 % with the real errors and with twice
them. So the optical points cost the RFC 61 points and the MLP 32 points of detection frequency on the
real spectrum, which is the note's haze mechanism (Section 3.3) in a weaker form: it does not flip the
verdict at JWST precision, but it decides the Tier-1 outcome. CH₄ and NH₃ false positives are 0 %, CO₂
is detected in 0–23 % (mean 7 %).

Expectations: A1 met for CH₄, NH₃ and CO₂ (absent at 10⁻⁴, split at 10⁻⁵), not met for H₂O (three of
four, not four). A2: CH₄ and NH₃ below 20 % met; H₂O above 80 % not met (two learners of four reach it;
mean 58 %); CO₂ between 30 and 70 % not met (7 %). A3 not met as written at 10⁻⁴ (the MLP's absent verdict
appears with every input variant, so it is not the optical-point mode), but met at 10⁻³ (all four absent
with the optical points, two present without) and in the Tier-1 frequencies at 10⁻⁴, 58 % with the
optical points against 86 % without. A4 met. A5 met: the H₂O, CH₄ and
NH₃ verdicts are the same for every reduction, and CO₂, as predicted, is not. A6 met for verdicts and
not for the probability bound (the MLP's H₂O score moves by 0.21 through the conversion, the others by
at most 0.12). A7 met.

Caveat. The MIRI level relative to the near-infrared modes is set through a model the benchmark did not
fit to MIRI; the alternative (MIRI as published) changes the CO₂ verdict of one learner and nothing else.

### F2. The three real CH₄-bearing planets are missed with all inputs in six of six reductions and found without the optical points in five of six; the distance rule puts all six at or beyond its threshold

TOI-270 d (CH₄ at 9.4σ, log VMR −1.6), K2-18 b (4–11σ in three analyses, −1.7 to −1.1) and TOI-1231 b
(ln B 55–70, −0.85), through six reductions: TOI-270 d from Felix et al. 2025 and from the discovery
team (Holmberg & Madhusudhan 2024, G395H only); K2-18 b from the discovery team's own files
(Madhusudhan et al. 2023), from Schmidt & MacDonald 2025 and from Hu et al. 2025; TOI-1231 b from
Sarkar et al. 2026 (`results/jwst_panel/panel_summary.txt`, `controls.txt`).

| reduction | all inputs (KNN MLP RFC SVC) | no optical | AIRS only | distance (threshold 11.25) |
| :-- | :-- | :-- | :-- | :-- |
| TOI-270 d, Felix | 1000, absent | absent | split 2–2 | 12.70, flagged |
| TOI-270 d, Holmberg | 0000, absent | absent | present | 11.01 |
| K2-18 b, Madhusudhan | 0000, absent | absent | present | 11.25, at the threshold |
| K2-18 b, Schmidt | 1000, absent | split | present | 11.54, flagged |
| K2-18 b, Hu | 1000, absent | absent | present | 11.69, flagged |
| TOI-1231 b, Sarkar | 1100, split | absent | present | 11.21 |

With all inputs the classifier calls CH₄ absent on five reductions and splits on the sixth; the Tier-1
detection frequencies are 7–16 %. With the AIRS channels alone, the same learners retrained without the
optical and NIRSpec points, it calls CH₄ present on five of six and splits on the sixth. That is the
note's optical-point mechanism (Section 3.3, haze mode) on real planets: all three have clouds or hazes
in their retrievals (TOI-1231 b: clouds and haze at ln B 7.9; K2-18 b: grey cloud and haze at about
3σ), and the three optical photometric points carry that shape into the learners. The muting census
(F7) shows these sub-Neptunes are not more muted than their twins (0.6–2.2, inside the population's
range), so unlike the warm giants the problem is the shape, not the amplitude. Expectation P2 (the
no-optical or AIRS-only variant recovers the misses) is met for the AIRS-only variant, not for the
no-optical one; P3 (the distance rule flags the misses) is met for three of six, and the other three sit
at 11.0–11.25 against a clean median of 9.93, that is, at the 90th percentile of clean distances.

Control: the clear H₂/He twins with the published abundances, through the same coverage mask and flat
fill, scattered with the real per-bin JWST errors (19–45 ppm after binning), are detected as
CH₄-bearing in 68, 95, 100, 100 and 100 % of 200 draws (TOI-270 d Felix, K2-18 b three ways,
TOI-1231 b) and never flagged; the G395H-only Holmberg reduction's twin, with 63 of 104 bins filled, is
detected in 19 % and is the one case where coverage limits the test. So the misses are not the noise,
not the coverage (except for that one), and not the classifier's in-simulator skill: they are the shape
of the real spectra, and the information is in the spectrum, because the AIRS-only learner finds it. In
the editor's taxonomy this is a recoverable error. Of the F5 candidates (`results/diagnostic_real.txt`),
the input-subset disagreement flags the CH₄ cell on four of the six reductions (K2-18 b Madhusudhan,
K2-18 b Hu, TOI-1231 b, TOI-270 d Holmberg; the MLP's |p_all − p_AIRS| reaches 0.86–1.00), while the
molecule-independent optical-excess score flags only the two K2-18 b reductions with the largest
optical excess (1.51 and 1.58 against 1.29–1.39 for Hu and TOI-1231 b, below its threshold); so on the
real sub-Neptunes the disagreement candidate, not the optical score, is the one that would route
them to the AIRS-only learner.

Switching K2-18 b from the third-party copy to the discovery team's files moved its distance from 12.3
to 11.25 and its all-input verdict from 0100 to 0000, and raised the AIRS-only verdict from absent to
present; the copy's −41 ppm instrument offset was replaced by the −92 ppm offset our alignment finds
between SOSS and G395H. Caveats: labels from the literature (the CH₄ detections are among the most
secure in the JWST literature); 5–30 bins flat-filled (63 for the G395H-only reduction).

### F3. With AIRS-only coverage the classifier cannot test H₂O, and says so by construction

WASP-107 b, HAT-P-26 b and GJ 3470 b (NIRCam or G395H only, 50–69 of 104 bins flat-filled) are all
called H₂O-absent although H₂O is present at 6–21σ. Their twins through the same mask are also missed
(1 of 4 learners), while the same twins with full coverage are detected by all four: the H₂O bands
at 1.4, 1.9 and 2.7 µm sit in the uncovered NIRSpec region. This is a coverage artifact, not a
classifier failure, and the paper scores these planets only on CH₄ and CO₂, whose bands the data
cover. On those: CH₄ absent (correct, all three), CO₂ present for WASP-107 b (3 of 4; literature
ambiguous), HAT-P-26 b (2–3 of 4; present) and GJ 3470 b (2 of 4 all-input, 3 of 4 AIRS-only; present).

### F4. On real spectra the classifier's default is "absent"

Over the 35 panel labels the literature decides (thirteen inputs, ten planets, K2-18 b three ways,
TOI-270 d twice), the all-input classifier is right on 22 (63 %), which is below the share of absent
labels (22 of 35). It reproduced none of the thirteen "present" labels outright (one split). Without the
optical points it is right on 25 of 36, and with the AIRS channels alone on 29 of 37 (78 %), where it
finds CH₄ on five of six sub-Neptune reductions (F2) and CO₂ on four of five warm giants. Even where it is right, the Tier-1 detection frequencies
of present molecules are low (CO₂ on WASP-107 b 49 %; everything else below 30 %).

WASP-166 b is the clean case of this: a hot super-Neptune with 0.86–5.16 µm coverage, inside the
training distribution (distance 8.0), H₂O at 15σ and CO₂ at 14.7σ in the literature, and the classifier
calls both absent (H₂O 0 of 4 learners; Tier-1 frequency 13 %). Its clear H₂/He twin through the same
mask is itself only marginal (H₂O 2 of 4 learners noise-free, 49 % with the real errors), because the
mask removes the optical points and everything beyond 5.16 µm, where the twin's strongest H₂O band
sits; with full coverage the twin is detected by all four. The real spectrum's features are 2.4 times
more muted than the twin's (89 against 213 ppm). So part of this miss is coverage and part is muting;
the MIRI-less panel cannot separate them for this planet.

Together with F1's four-fold muting, F4 says the real spectra carry smaller features, relative to their
continua, than the clear-grid training spectra, and the classifier's operating point is tuned to the
latter. Expectation P1 (≥ 75 % of scored labels) not met at 63 % with all inputs, met at 78 % with the AIRS
channels alone; P6 met (no NH₃ false positive on any panel planet).

### F5. The diagnostic: a one-number optical-excess score, routed to the AIRS-only classifier, is never worse than the classifier and recovers a third of the CH₄ targets haze loses

Candidates (a) input-subset disagreement, (b) optical excess (the three normalized photometric points
against the normalized NIRSpec continuum, as a z-score on the clean training set), (c) band-metric
consistency and (d) predicted decidability, scored on POP-I under clean, haze 3 × 10⁷, spots 20 %,
cloud 10² Pa, compound, noise × 2 and × 3, against the note's five rules
(`alfnoor_diagnostic.py`; `results/diagnostic.txt`, `diagnostic_*.csv`, `diagnostic_real.txt`).

Selective credit (C1): no candidate has positive credit under haze or spots at the 10 % clean decline
rate (disagreement −6.8 and −6.3, the best of the nine; the five old rules −17.8 to −25.7 under haze).
C1 not met. The reason is instructive: the optical-excess score flags 100 % of hazy and 66 % of spotted
planets while flagging 10 % of clean, 1.5 % of clouded and 2–3 % of noisier ones, so declining what it
flags empties the hazy population (coverage 0 %) and credit is not the right measure of a shift
detector. Used as a router instead, which is what C1's design intended: the flagged planets are
re-scored by the AIRS-only learner (the same learner retrained without the optical points). Under haze
the CH₄ accuracy rises from 66.6 to 78.3 % and the CH₄ recall from 11.6 to 53.7 % (compound: 5.4 to
50.3 %; spots: 35.5 to 53.1 %), while under clean, cloud and noise the routed classifier is within 0.5
points of the frozen one, because almost nothing is flagged. The disagreement candidate does nearly
the same (77.7 % and 51.1 % under haze) at the cost of three retrained learners per molecule; the
band-metric candidate makes things worse.

Selection (C3, C6): 60 % of the 939 POP-I planets get a Tier-2 slot, ranked by the mean present
probability of KNN, MLP and RFC. Under haze the classifier alone loses 38 of 359 CH₄-bearing targets;
with the optical-excess routing 26; the AIRS-only classifier used everywhere 27; the best old rule
(kNN, declined to the back) 42, and the margin and ensemble rules 105 and 112, worse than the band
metric alone (103). Under clean spectra the routed classifier loses 19 against 20, the AIRS-only
classifier 27; under noise × 2 it loses 37 against 37, the AIRS-only classifier 57. So the diagnostic
buys the AIRS-only classifier's immunity to haze without paying its clean and noisy cost; it is never
worse than the classifier in any case. C3 (fewer than half the lost targets) not met: 26 against 38
is a third, not a half, and the remaining misses are the planets the AIRS-only learner also misses.
C6 is met trivially: with a fixed slot count, slots wasted equal targets lost plus a constant, so the
cost ranking is the same at every lost-target weight; the weight only matters if the budget can flex.
For H₂O no candidate helps (78 lost under haze for the classifier, 82 with disagreement): the haze
loss on H₂O is not routed through the optical points to the same degree.

Error classes (C2, C4) and the comparison with a retrieval of every planet (E3): see F6, which
supersedes the uncalibrated prevalence (93–96 % irrecoverable at 3σ) first reported here and shows
that on clean simulated spectra the retrieval, not the classifier with any diagnostic, is the
selection method.

Real spectra (C5): the optical-excess score flags HAT-P-18 b (z = 2.3, the spotted host) and the
K2-18 b reanalysis (1.6), not WASP-39 b (0.8), WASP-166 b (0.9) or the primary K2-18 b (0.8); it is
molecule-independent, so it says nothing about the CH₄ misses of F2. The band-metric candidate flags
exactly the one wrong learner on WASP-39 b (the MLP's H₂O verdict, |p − flag| = 1.00) and no other
WASP-39 b cell, and on the panel it fires mostly on H₂O cells of flat-filled planets, where the band
metric itself is unreliable. The disagreement candidate flags no WASP-39 b cell. C5 as written (fires on
WASP-39 b iff A3; fires on HAT-P-18 b and K2-18 b; not on WASP-166 b) is met for the optical score on
HAT-P-18 b, the reanalysed K2-18 b and WASP-166 b, and not for the primary K2-18 b.

### F6. Identifiability, measured by a retrieval of every planet: a third to a half of the consortium labels are undecidable at Tier-1 noise, the retrieval of every planet is the Tier-2 selection oracle, three quarters of the classifier's clean errors are irrecoverable, and no diagnostic routes a retrieval budget better than chance

The Δχ² identifiability test ran on all 955 POP-I test planets for the four molecules (3 restarts, 8
workers, 15 h, 207 core-hours; `identifiability.py`; `results/identifiability_none_tier1.{csv,txt}`).
For each label it finds the best-fitting atmosphere of the opposite class, with the molecule held at
the 10⁻⁴ boundary and temperature (0.7–1.05 T_eq), grey cloud top (5 × 10² to 10⁶ Pa), the other three
abundances (10⁻⁹ to 10⁻²) and an additive offset free, and reports the χ² gap at the planet's Tier-1
per-bin noise over the 104 bins. A nested-sampling retrieval of every one of the 955 planets (same
free quantities, the four abundances over 10⁻⁹ to 10⁻², 100 live points, median 3,960 likelihood
calls and 19.9 min per planet, 365 core-hours, 29 h on 8 workers; `identifiability_calibrate.py`;
`results/identifiability_calibration.{csv,txt}`, `identifiability_calibration_thresholds.csv`) gives
for every label the posterior probability of the true class, p_true, which is what "decidable" should
mean, and the posterior probability of "present", which is what a retrieval-based selection would rank
by. The 200-planet pilot (`identifiability_calibration_200.*`) is superseded; every number below is
the population's, not a mapping from a sample.

Calibration (B1): Spearman between the Δχ² gap and logit p_true is 0.86 (CH₄), 0.80 (H₂O), 0.88 (CO₂)
and 0.89 (NH₃), above the 0.7 kill criterion. B1 met. The test's scale was wrong by an order of
magnitude: p_true crosses 0.95 at Δχ² of 0.45, 0.98, 0.90 and 0.56, not at the nested-model 3.84 or
the 3σ value 9, because the test represents the opposite class by its single best-fitting point and
so over-credits that class, while the retrieval integrates over its volume. The uncalibrated 3σ
fractions (78–88 % undecidable) and the error-class prevalence first reported in F5 (93–96 %
irrecoverable) are superseded. The cheap test is also not cheap: 207 core-hours against the
retrieval's 365, so on a population of this size the retrieval is the method and the Δχ² test is at
most a check on it.

Undecidable fractions on the population (B2), labels with p_true below the level, counted directly:

| | p_true < 0.95 | p_true < 0.84 | Δχ² test alone, thresholded at its conditional mean | B2 expected |
| :-- | :-- | :-- | :-- | :-- |
| CH₄ | 32 % | 17 % | 45 % | 10–30 % |
| H₂O | 54 % | 28 % | 72 % | 20–40 % |
| CO₂ | 44 % | 27 % | 59 % | 10–30 % |
| NH₃ | 32 % | 19 % | 42 % | 20–40 % |

B2 met at the 0.84 level for all four, at the 0.95 level for NH₃ and (by two points) CH₄, not for H₂O
and CO₂. Only 17 % of the planets have all four labels decidable at 0.95, 3 % have none; 33 % have
both CH₄ and H₂O decidable. Decidability is set first by the label's distance from the 10⁻⁴ boundary
and second by the planet's noise: labels within half a decade of the boundary are undecidable in
89–92 % of cases, labels half to one decade away in 43–70 %, one to two decades away in 12–36 %, and
"absent" labels two to three decades below it in 1 % (CO₂), 3 % (NH₃), 8 % (CH₄) and 31 % (H₂O), the
last being the planets whose noise leaves water unconstrained at any abundance. The earlier
statement that proximity does not matter was an artefact of the 3σ scale and is withdrawn.

Ceiling (B3): the retrieval itself scores 93.7 % (CH₄), 91.6 % (H₂O), 93.8 % (CO₂) and 93.9 % (NH₃)
on the clean spectra (its own expected accuracy, the mean of max(p, 1 − p), is 92.5, 88.3, 89.2 and
92.3 %, so it is slightly under-confident); it is confidently wrong (p_true < 0.05) on one label of
3,820. The four learners score 84.3, 75.3, 80.7 and 84.6 %: gaps of 9, 16, 13 and 9 points, 12 on
average against the 8 B3 asked for. B3 not met; the pilot's "met within a point" used the
isotonic-mapped ceiling, which the full retrieval shows was 2–4 points too low. So the classifier
is well short of the information limit on clean spectra.

Error classes (E4, B4), each learner's errors split against the posterior of the same spectrum:
recoverable if p_true > 0.95 (a retrieval of that spectrum gets the label right), undecidable if
0.05 ≤ p_true ≤ 0.95, misled if p_true < 0.05 (`alfnoor_diagnostic.py` section d″;
`results/diagnostic_retrieval_classes.csv`); share recoverable, mean over the four learners, with the
range over learners for the clean case:

| | clean | haze 3 × 10⁷ | spots 20 % | cloud 10² Pa | noise × 2 | noise × 3 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ | 25 % (21–29) | 58 % | 46 % | 48 % | 47 % | 54 % |
| H₂O | 20 % (14–34) | 43 % | 33 % | 33 % | 32 % | 36 % |
| CO₂ | 20 % (16–22) | 39 % | 29 % | 28 % | 30 % | 35 % |
| NH₃ | 19 % (11–34) | 54 % | 46 % | 45 % | 41 % | 47 % |

Misled errors are 0 % (NH₃ 0.3–0.8 %). On clean spectra three quarters to four fifths of the
classifier's errors are on labels the spectrum cannot decide; under every departure the recoverable
share rises because the departure adds errors on decidable labels. The two thresholds disagree on
individual labels (the Δχ² version of the same table, `diagnostic_classes.csv`, gives 50/34/31/36 %
irrecoverable on clean CH₄/H₂O/CO₂/NH₃ against the posterior's 75/80/80/81 %): the rank correlation
of 0.8–0.9 is not a label-by-label agreement, and the posterior is the definition. B4 asked for at
least 70 % of the haze errors to be recoverable: not met for any molecule (CH₄ 58 %, NH₃ 54 %, H₂O
43 %, CO₂ 39 %); its noise × 3 clause (at most 30 % recoverable) is not met either, but these
departure columns still carry the clean spectrum's decidability, and the departure identifiability
runs (haze 3 × 10⁷ running, then spots, cloud, noise × 2 and × 3) replace them.

Self-consistency: on decidable labels the learners score 93.5–95.7 % (CH₄), 76–93 % (H₂O), 92–95 %
(CO₂) and 91–98 % (NH₃); on undecidable ones 60–68, 61–66, 61–70 and 58–63 % against majority rates
of 57, 60, 53 and 55 %, so a few points above the prior. Recall of present labels on decidable CH₄,
CO₂ and NH₃ is 82–84 % and on undecidable 35–49 %: the "absent by default" of F4 is the classifier
reverting to the prior where the spectrum gives it nothing. The learners are right on 20–39 % of the
labels the retrieval gets wrong, all of them undecidable, which is the prior again.

Selection (C3, C4, E3): with the retrieval's "present" posterior as the ranking, the 60 % Tier-2
selection loses 0 of 359 CH₄ targets (oracle 0; classifier 20), 2 of 364 H₂O targets (oracle 0;
classifier 51) and 39 joint targets (oracle 3; classifier 95). On clean simulated spectra the existing
method, a retrieval of every planet, is the selection oracle to within noise, at 365 core-hours for the
population, two days on a laptop; the classifier and every diagnostic on it are dominated. The
question that remains open is what a diagnostic buys at a fixed retrieval budget. Triage
(`diagnostic_triage.csv`): the fraction f of planets each rule ranks most suspect is re-scored by the
retrieval and the rest keep the classifier; CH₄ targets lost at the 60 % selection:

| ordering | f = 10 % | 20 % | 30 % | 50 % | 70 % |
| :-- | :-- | :-- | :-- | :-- | :-- |
| random | 16 | 12 | 7 | 3 | 2 |
| retrieval's own undecidability (unattainable) | 7 | 7 | 4 | 1 | 0 |
| measured Δχ² (costs 57 % of the retrieval) | 13 | 8 | 4 | 2 | 0 |
| kNN density | 13 | 8 | 5 | 3 | 1 |
| Mahalanobis | 13 | 9 | 7 | 3 | 1 |
| predicted decidability (bulk features) | 16 | 13 | 9 | 2 | 1 |
| margin | 17 | 13 | 9 | 6 | 2 |
| ensemble | 17 | 15 | 11 | 3 | 1 |
| disagreement | 17 | 14 | 12 | 6 | 1 |
| optical excess | 15 | 12 | 8 | 5 | 2 |
| band metric | 19 | 15 | 13 | 8 | 2 |

Nothing attainable beats a random budget by more than three targets, and the confidence-based rules
(margin, ensemble, disagreement, band metric) are worse than random at every budget up to 50 %:
they send the retrieval to the planets the classifier is unsure about, which are the undecidable
labels, where the retrieval cannot help either; the retrieval pays off on the planets the classifier
is confidently wrong about, and no rule finds those. For H₂O the same holds (random 35 lost at 30 %,
predicted decidability 29, measured Δχ² 32, margin 42, band metric 44). C3 and C4 not met; C2
(predicting decidability from bulk features, AUROC 0.47–0.65 against 0.8 asked) not met. For the
paper, E3's "improves target selection against existing methods" cannot be claimed against a
retrieval on clean simulated spectra; what the diagnostic has is F5's immunity to the haze departure,
and whether a retrieval with the same clear-atmosphere forward model shares that failure on hazy
spectra is the test still to run (the departure spot-check retrievals).

### F7. Muting census: six of the fourteen real JWST spectra are more muted than any atmosphere the simulator population can produce at their bulk parameters, and the distance rule cannot see it because it normalises amplitude away

For each real planet with a converted spectrum, the dispersion of its measured Tier-3 bins (standard
deviation of the depths, with the per-bin errors removed in quadrature) against the same dispersion of
its clear H₂/He twin at the published abundances over the same bins; the twin under grey decks from the
population's lowest allowed cloud top (5 × 10² Pa) down to 1 Pa; the twin at the population's coldest
allowed temperature (0.7 T_eq, the floor of its prior) with the 5 × 10² Pa deck, which is the most muted
atmosphere the population can draw at those bulk parameters; and the cloud-only muting of a random
sample of 150 POP-I planets, their rendered spectrum against their own clear twin
(`jwst_muting_census.py`; `results/jwst_panel/muting_census.{txt,csv}`, `muting_population.csv`).

| planet | real, corrected (ppm) | clear twin (ppm) | muting | most the prior allows | beyond the prior | AIRS bins only: muting / prior |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| HAT-P-26 b | 159 | 847 | 5.3 | 2.4 | yes | 5.3 / 2.4 |
| WASP-107 b | 263 | 1171 | 4.5 | 2.2 | yes | 4.5 / 2.2 |
| WASP-39 b | 319 | 1190 | 3.7 | 2.0 | yes | 3.3 / 1.9 |
| HAT-P-18 b* | 210 | 778 | 3.7 | 3.3 | marginally | 3.5 / 3.1 |
| GJ 3470 b | 53 | 182 | 3.5 | 2.0 | yes | 3.5 / 2.0 |
| TOI-421 b* | 35 | 119 | 3.4 | 3.7 | no | 3.5 / 3.5 |
| WASP-166 b | 69 | 213 | 3.1 | 1.8 | yes | 3.0 / 1.8 |
| HD 189733 b* | 104 | 182 | 1.8 | 2.3 | no | 1.7 / 2.3 |
| K2-18 b, Hu | 36 | 71 | 2.0 | 2.7 | no | 2.2 / 2.6 |
| K2-18 b, Schmidt | 42 | 71 | 1.7 | 2.7 | no | 1.5 / 2.6 |
| TOI-1231 b | 41 | 63 | 1.6 | 1.9 | no | 1.5 / 1.8 |
| TOI-270 d, Holmberg | 48 | 50 | 1.0 | 1.9 | no | 1.0 / 1.9 |
| K2-18 b, Madhusudhan | 94 | 68 | 0.7 | 2.6 | no | 0.7 / 2.6 |
| TOI-270 d, Felix | 86 | 57 | 0.7 | 2.0 | no | 0.6 / 1.8 |

\* twin water abundance from the literature with the other gases at the floor (HAT-P-18 b: log H₂O −3.03,
Fu et al. 2022, the paper whose spectrum is used; a 2024 reanalysis finds −4.4, which would shrink the
twin and the muting; HD 189733 b: 3–5 × solar, Fu et al. 2024; TOI-421 b: solar, mean molecular weight
2.3–2.7, Davenport et al. 2025); the others use the published retrievals. Changing the twin's water
abundance between 10⁻² and 10⁻³ moves its dispersion by 10–30 %, an order of magnitude less than the
muting. The last column repeats the measure over the AIRS bins alone (1.95–7.8 µm), which excludes the
optical slope: it changes nothing by more than 0.4, so the muting is in the molecular bands, not in a
haze slope. Within the population the cloud-only muting over the AIRS bins has median 1.00, 90th
percentile 1.09, 99th 2.12 and maximum 2.59 of 150.

Every warm giant with a published abundance set (HAT-P-26 b, WASP-107 b, WASP-39 b, GJ 3470 b, WASP-166 b)
is muted by a factor of 3 to 5 against its clear twin, and by 1.5 to 2.2 beyond the coldest, cloudiest
atmosphere the population allows at its mass, radius and irradiation. For five planets no grey deck at any
pressure in the model reaches the real amplitude, because their photospheres sit above the population's
cloud floor. Within the population, clouds mute the median planet by 1 %, the 90th percentile by 23 %,
the 99th by a factor of 2.4, and the most muted of 150 by 3.1; none of the sampled planets is as muted
as any of the five giants. The three sub-Neptunes with retrieved methane-rich atmospheres (K2-18 b,
TOI-270 d, TOI-1231 b, six reductions) are the only real planets whose amplitude the population covers,
because their twins are already heavy, and the classifier missed CH₄ on all six for the different reason
of F2 (the shape the optical points carry).

What this says: in feature amplitude the real warm giants lie outside the simulator population, although
their bulk parameters lie inside it. The population's atmospheres are H₂/He with trace gases at 1 % at
most, temperatures of 0.7–1.05 T_eq and a grey deck no higher than 5 × 10² Pa; the real planets need the
remaining factor of 1.5–2.2 from what the population does not contain, metallicities of tens to hundreds
of times solar, hazes, or patchy clouds, which is what their JWST retrievals report. The note's distance
rule, and every input-space rule in it, works on normalised spectra and so removes exactly this
quantity: WASP-39 b is in-distribution by distance (F1) and the most muted giants here are not flagged
either. This is the measurement behind the reviewer's "joint distribution of Ariel targets unknown": the
first fourteen JWST transmission spectra (ten planets) converted to the Ariel grid say the simulator population is
systematically too clear for warm giants, and the one learner that fails on WASP-39 b fails on the planet
the population is least able to represent. It also explains F4 (absent by default): on muted spectra the
bands fall under the noise the classifier was trained to expect, and it reverts to the prior.

Caveats: ten planets in fourteen reductions, five warm giants with retrieved abundances carrying the result, with no
exception among them; the twin's temperature is the MCS equilibrium temperature, and the 0.7 T_eq column
allows for a cooler terminator only up to the population's own floor, which is the comparison the
question needs (a cooler real terminator is part of what the population does not represent). The
optical-slope concern is closed by the AIRS-only column.

### What these add up to, provisionally

A simulator-trained classifier that reproduces its published in-simulator accuracies is mostly right
on the one real hot Saturn in its training domain (three learners of four, and the non-ML band metric,
on H₂O; all four on CH₄, CO₂ and NH₃), with one learner wrong on H₂O for a reason the twins locate in
the real spectrum's muting rather than in the data, the conversion or the noise; it is wrong on both
real planets that carry the molecule it was built to find, because those planets lie outside the
domain; and a cheap distance rule, useless for ranking errors inside the simulator (the note's Section
3.4), identifies exactly those two planets on real data and not the hot Saturn. Under Ariel noise the
optical photometric points decide the hot Saturn's Tier-1 outcome (58 % against 86 % without them),
which is the note's haze mechanism showing on a real planet. A one-number diagnostic on the known weak point, the optical
excess, routed to the learner without that weak point, is never worse than the classifier and recovers
a third of the CH₄ targets haze would lose from Ariel's Tier-2 queue; the old rules lose more targets
than ranking by the band metric alone. With a nested-sampling retrieval of every planet
(365 core-hours), a third to a half of the consortium labels are undecidable at Tier-1 noise, the
classifier sits 9–16 points below the retrieval's accuracy, three quarters of its clean errors are
irrecoverable, the errors the departures add are about half recoverable, the retrieval of every
planet is the Tier-2 selection oracle on clean spectra, and no diagnostic routes a partial retrieval
budget better than chance, the confidence rules worse. The muting census (F7) turns the one-planet WASP-39 b
result into a population statement: five of five warm giants with retrieved abundances are muted beyond
anything the simulator population can draw at their bulk parameters, which no normalised-input rule can
see. The remaining pieces are the departure identifiability runs, the retrieval spot checks under haze,
and the manuscript.
