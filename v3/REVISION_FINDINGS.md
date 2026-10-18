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

### F2. The two real CH₄-bearing planets are missed, and the distance rule flags both

TOI-270 d (CH₄ at 9.4σ, log VMR −1.6) and K2-18 b (CH₄ at 4–11σ in three analyses, log VMR −1.7 to
−1.1) are called CH₄-absent by the classifier with all inputs (1 of 4 learners present on each), without
the optical points, and in the AIRS-only variant (TOI-270 d split 2–2). Their 10-nearest-neighbour
distances from the training set, 12.7 and 12.3, exceed the threshold that declines 10 % of clean
simulated planets (11.25): both are flagged as outside the training distribution.

Control (`results/jwst_panel/controls.txt`): the clear H₂/He twins with the same published abundances,
through the same coverage mask and the same flat fill, scattered with the real per-bin JWST errors
(30 and 45 ppm after binning), are detected as CH₄-bearing in 68 % and 96 % of 200 draws and are
never flagged; at twice the real errors they are still detected in 43 % and 60 %. So the miss is not
the noise, not the coverage, and not the classifier's in-simulator skill: it is the shape of the real
spectra. Both planets are temperate sub-Neptunes around M dwarfs with high mean molecular weight and
clouds or haze, a kind of atmosphere the consortium grid (H₂/He, four absorbers, gray cloud) does not
contain. In the editor's taxonomy this is a recoverable error: the information is in the spectrum
(JWST sees it), the classifier's training prior does not reach it, and a rule the note already
tested would have declined both planets. Expectation P3 met; P2 not met (the no-optical and AIRS-only
variants do not recover the misses, so this is not the haze mode of Section 3.3).

Robustness to the reduction: the independent Schmidt & MacDonald 2025 reanalysis of K2-18 b
(exoTEDRF, R = 100) gives the same answer: distance 11.5 (flagged), CH₄ called by 1 of 4 learners,
Tier-1 frequency 18 %; its twin with the real errors is detected in 100 % of draws.

Caveats. TOI-270 d is the Felix et al. 2025 reanalysis at native resolution, not Benneke et al.'s
product; the primary K2-18 b spectrum is a third-party copy of the 2023 spectrum with a −41 ppm offset
applied to one instrument; 17, 5 and 16 of 104 bins are flat-filled. The discovery-team originals are on
OSF and will be run when it is back. The labels are the literature's; the CH₄ detections are among the
most secure in the JWST literature.

### F3. With AIRS-only coverage the classifier cannot test H₂O, and says so by construction

WASP-107 b, HAT-P-26 b and GJ 3470 b (NIRCam or G395H only, 50–69 of 104 bins flat-filled) are all
called H₂O-absent although H₂O is present at 6–21σ. Their twins through the same mask are also missed
(1 of 4 learners), while the same twins with full coverage are detected by all four: the H₂O bands
at 1.4, 1.9 and 2.7 µm sit in the uncovered NIRSpec region. This is a coverage artifact, not a
classifier failure, and the paper scores these planets only on CH₄ and CO₂, whose bands the data
cover. On those: CH₄ absent (correct, all three), CO₂ present for WASP-107 b (3 of 4; literature
ambiguous), HAT-P-26 b (2–3 of 4; present) and GJ 3470 b (2 of 4 all-input, 3 of 4 AIRS-only; present).

### F4. On real spectra the classifier's default is "absent"

Over the 28 panel labels the literature decides (ten inputs, nine planets, K2-18 b twice), the
all-input classifier is right on 19 (68 %), which is below the share of absent labels (19 of 28). It
reproduced one of the nine "present" labels. Even where it is right, the Tier-1 detection frequencies
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
latter. Expectation P1 (≥ 75 % of scored labels) not met at 68 %; P6 met (no NH₃ false positive on any
panel planet).

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

Error classes (C2, C4): see F6, which supersedes the uncalibrated prevalence (93–96 % irrecoverable
at 3σ) first reported here.

Real spectra (C5): the optical-excess score flags HAT-P-18 b (z = 2.3, the spotted host) and the
K2-18 b reanalysis (1.6), not WASP-39 b (0.8), WASP-166 b (0.9) or the primary K2-18 b (0.8); it is
molecule-independent, so it says nothing about the CH₄ misses of F2. The band-metric candidate flags
exactly the one wrong learner on WASP-39 b (the MLP's H₂O verdict, |p − flag| = 1.00) and no other
WASP-39 b cell, and on the panel it fires mostly on H₂O cells of flat-filled planets, where the band
metric itself is unreliable. The disagreement candidate flags no WASP-39 b cell. C5 as written (fires on
WASP-39 b iff A3; fires on HAT-P-18 b and K2-18 b; not on WASP-166 b) is met for the optical score on
HAT-P-18 b, the reanalysed K2-18 b and WASP-166 b, and not for the primary K2-18 b.

### F6. Identifiability, calibrated: at Tier-1 noise a third to a half of the consortium labels are undecidable, the classifier sits 6–10 points below the information ceiling, and the haze errors are mostly on labels the spectrum could decide

The Δχ² identifiability test ran on all 955 POP-I test planets for the four molecules (3 restarts, 8
workers, 15 h; `identifiability.py`; `results/identifiability_none_tier1.{csv,txt}`). For each label
it finds the best-fitting atmosphere of the opposite class, with the molecule held at the 10⁻⁴
boundary and temperature (0.7–1.05 T_eq), grey cloud top (5 × 10² to 10⁶ Pa), the other three
abundances (10⁻⁹ to 10⁻²) and an additive offset free, and reports the χ² gap at the planet's Tier-1
per-bin noise over the 104 bins. Nested-sampling retrievals of 200 planets (half random, half with a
label within 0.7 dex of the boundary; same free quantities; 100 live points, median 3,900 likelihood
calls and 14 min per planet, 8 h on 8 workers; `identifiability_calibrate.py`;
`results/identifiability_calibration_200.{csv,txt}`, `identifiability_calibration_thresholds.csv`)
give the posterior probability of the true class, p_true, which is the quantity "decidable" should mean.

Calibration (B1): Spearman between the Δχ² gap and logit p_true is 0.85 (CH₄), 0.76 (H₂O), 0.88 (CO₂)
and 0.88 (NH₃), all above the 0.7 kill criterion, so the cheap test ranks labels as the retrieval does
and keeps its role on the population. B1 met. Its scale was wrong by an order of magnitude: p_true
crosses 0.95 at Δχ² of 0.44, 0.97, 0.96 and 0.75 for the four molecules, not at the nested-model 3.84
or the 3σ value 9. The reason, visible before the retrievals ran, is that the test represents the
opposite class by its single best-fitting point and so over-credits that class; the retrieval
integrates over its volume. The uncalibrated 3σ fractions (78–88 % undecidable) and the error-class
prevalence first reported in F5 (93–96 % irrecoverable) are superseded by what follows. The retrieval
is being extended to all 955 planets (29 h) so that the headline numbers come from it directly.

Undecidable fractions on the population (B2), labels with p_true below the level:

| | below 0.95, calibrated Δχ² threshold | below 0.95, isotonic map | below 0.84, isotonic map | B2 expected |
| :-- | :-- | :-- | :-- | :-- |
| CH₄ | 29 % | 43 % | 16 % | 10–30 % |
| H₂O | 56 % | 75 % | 42 % | 20–40 % |
| CO₂ | 42 % | 45 % | 38 % | 10–30 % |
| NH₃ | 32 % | 41 % | 26 % | 20–40 % |

B2 met for CH₄ and NH₃ at the threshold, not for H₂O and CO₂; the two mappings differ because the
threshold is a 50 % crossing and the isotonic map carries the spread. The undecidable fraction is set
by the planet's noise, not by proximity to the boundary: labels within half a decade of 10⁻⁴ are
undecidable in 95–99 % of cases at 3σ, but so are 46–76 % of labels two decades above it.

Ceiling (B3): a Bayes classifier with the retrieval's posteriors would score, mapped onto the
population, 90.7 % (CH₄), 84.9 % (H₂O), 87.8 % (CO₂) and 91.7 % (NH₃); the four learners score 84.3,
75.3, 80.7 and 84.6 %. Gaps of 6, 10, 7 and 7 points, 7.5 on average against the 8 B3 asked for: met
within a point. So the classifier is not at the information limit on clean spectra; a quarter to a
third of its clean errors are recoverable in principle.

Self-consistency (`alfnoor_diagnostic.py` section d′; `results/diagnostic_consistency.csv`): at the
calibrated threshold the four learners score 87–94 % on decidable CH₄, CO₂ and NH₃ labels and 67–87 %
on decidable H₂O labels, against majority rates of 54–64 %; on undecidable labels they score 72–74 %
(CH₄) and 68–73 % (H₂O) against majority rates of 70 and 68 %, that is, the prior, and 64–72 % (CO₂)
and 66–68 % (NH₃) against 54 and 58 %, so at the strict 0.95 level some information remains in those
two (the retrieval puts it between p_true 0.84 and 0.95). At the 3σ threshold the same check failed
(76–83 % against 60–68 %), which is what exposed the scale error. The recall of present labels falls
from 83–86 % on decidable CH₄, CO₂ and NH₃ to 36–50 % on undecidable ones, so the "absent by default"
of F4 is the classifier reverting to the prior where the spectrum gives it nothing.

Error classes (E4, B4), the share of each learner's errors on labels undecidable at the calibrated
threshold, mean over the four learners (`results/diagnostic_classes.csv`):

| | clean | haze 3 × 10⁷ | spots 20 % | cloud 10² Pa | noise × 2 | noise × 3 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ | 49 % | 26 % | 33 % | 34 % | 33 % | 29 % |
| H₂O | 66 % | 47 % | 55 % | 55 % | 57 % | 54 % |
| CO₂ | 70 % | 52 % | 60 % | 62 % | 59 % | 55 % |
| NH₃ | 68 % | 34 % | 41 % | 42 % | 47 % | 42 % |

On clean spectra half (CH₄) to two thirds (the others) of the errors are irrecoverable; under every
departure the share falls, because the departure adds errors on labels the spectrum could decide. B4
asked for at least 70 % of the haze errors to be recoverable: met for CH₄ (74 %), not for NH₃ (66 %),
H₂O (53 %) or CO₂ (48 %). B4's second clause (noise × 3: at most 30 % recoverable) is not met as
tabulated, but the table is not yet the right test of it: these columns use the clean-spectrum
decidability, whereas the plan's definition carries the departure into the counterfactual (the noise
× 3 label is undecidable at the × 3 noise, not the × 1 noise). The departure runs of `identifiability.py`
(haze3e7, spots20, cloud1e2, noise_x2, noise_x3) replace these columns when they finish; the clean
column and the haze column's direction stand.

Predicting decidability without the spectrum (C2, C4): a random forest on bulk and noise features
predicts the Δχ² out of fold with Spearman 0.33–0.42 and separates the two error classes with AUROC
0.46–0.67 at the calibrated threshold, against 0.61–0.86 at 3σ: the cheap predictor finds the
hopeless planets (tiny signal) but not the boundary of decidability. C2 (AUROC ≥ 0.8) not met.
Routing predicted-undecidable planets to the back of the queue loses targets, so C4's second clause
is not met; the retrieval-everywhere strategy, which the full retrieval run enables, is the proper
upper bound for C4 and is pending.

### F7. Muting census: six of the eleven real JWST spectra are more muted than any atmosphere the simulator population can produce at their bulk parameters, and the distance rule cannot see it because it normalises amplitude away

For each real planet with a converted spectrum, the dispersion of its measured Tier-3 bins (standard
deviation of the depths, with the per-bin errors removed in quadrature) against the same dispersion of
its clear H₂/He twin at the published abundances over the same bins; the twin under grey decks from the
population's lowest allowed cloud top (5 × 10² Pa) down to 1 Pa; the twin at the population's coldest
allowed temperature (0.7 T_eq, the floor of its prior) with the 5 × 10² Pa deck, which is the most muted
atmosphere the population can draw at those bulk parameters; and the cloud-only muting of a random
sample of 150 POP-I planets, their rendered spectrum against their own clear twin
(`jwst_muting_census.py`; `results/jwst_panel/muting_census.{txt,csv}`, `muting_population.csv`).

| planet | real, corrected (ppm) | clear twin (ppm) | muting | most the prior allows | beyond the prior |
| :-- | :-- | :-- | :-- | :-- | :-- |
| HAT-P-26 b | 159 | 847 | 5.3 | 2.4 | yes |
| WASP-107 b | 263 | 1171 | 4.5 | 2.2 | yes |
| WASP-39 b | 319 | 1190 | 3.7 | 2.0 | yes |
| HAT-P-18 b* | 210 | 778 | 3.7 | 3.2 | yes, marginally |
| GJ 3470 b | 53 | 182 | 3.5 | 2.0 | yes |
| TOI-421 b* | 35 | 119 | 3.4 | 3.7 | no |
| WASP-166 b | 69 | 213 | 3.1 | 1.8 | yes |
| HD 189733 b* | 104 | 182 | 1.8 | 2.3 | no |
| K2-18 b, reanalysis | 42 | 71 | 1.7 | 2.7 | no |
| K2-18 b | 72 | 68 | 0.9 | 2.6 | no |
| TOI-270 d | 86 | 57 | 0.7 | 2.0 | no |

\* twin abundances assumed (solar-metallicity water, no retrieved set in the survey); the others use the
published retrievals. Changing the twin's water abundance between 10⁻² and 10⁻³ moves its dispersion by
10–30 %, an order of magnitude less than the muting.

Every warm giant with a published abundance set (HAT-P-26 b, WASP-107 b, WASP-39 b, GJ 3470 b, WASP-166 b)
is muted by a factor of 3 to 5 against its clear twin, and by 1.5 to 2.2 beyond the coldest, cloudiest
atmosphere the population allows at its mass, radius and irradiation. For five planets no grey deck at any
pressure in the model reaches the real amplitude, because their photospheres sit above the population's
cloud floor. Within the population, clouds mute the median planet by 1 %, the 90th percentile by 23 %,
the 99th by a factor of 2.4, and the most muted of 150 by 3.1; none of the sampled planets is as muted
as any of the five giants. The two sub-Neptunes with retrieved methane-rich atmospheres (K2-18 b,
TOI-270 d) are the only real planets whose amplitude the population covers, because their twins are
already heavy, and the classifier missed CH₄ on both for the different reasons of F2 (coverage and
domain).

What this says: in feature amplitude the real warm giants lie outside the simulator population, although
their bulk parameters lie inside it. The population's atmospheres are H₂/He with trace gases at 1 % at
most, temperatures of 0.7–1.05 T_eq and a grey deck no higher than 5 × 10² Pa; the real planets need the
remaining factor of 1.5–2.2 from what the population does not contain, metallicities of tens to hundreds
of times solar, hazes, or patchy clouds, which is what their JWST retrievals report. The note's distance
rule, and every input-space rule in it, works on normalised spectra and so removes exactly this
quantity: WASP-39 b is in-distribution by distance (F1) and the most muted giants here are not flagged
either. This is the measurement behind the reviewer's "joint distribution of Ariel targets unknown": the
first eleven JWST transmission spectra converted to the Ariel grid say the simulator population is
systematically too clear for warm giants, and the one learner that fails on WASP-39 b fails on the planet
the population is least able to represent. It also explains F4 (absent by default): on muted spectra the
bands fall under the noise the classifier was trained to expect, and it reverts to the prior.

Caveats: eleven planets, eight with retrieved abundances; the twin's temperature is the MCS equilibrium
temperature and the real terminators are cooler, which the 0.7 T_eq column allows for only up to the
population's own floor; dispersion mixes the optical slope with the infrared bands, and a haze-dominated
optical slope could raise the real dispersion, which would make the muting here a lower bound for the
infrared.

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
than ranking by the band metric alone. With the identifiability test calibrated against retrievals,
a third to a half of the consortium labels are undecidable at Tier-1 noise, the classifier sits 6–10
points below the information ceiling, half to two thirds of its clean errors are irrecoverable, and
the errors the departures add are mostly recoverable. The muting census (F7) turns the one-planet WASP-39 b
result into a population statement: five of five warm giants with retrieved abundances are muted beyond
anything the simulator population can draw at their bulk parameters, which no normalised-input rule can
see. The remaining pieces are the retrieval on every planet (running), the departure identifiability
runs, and the manuscript.
