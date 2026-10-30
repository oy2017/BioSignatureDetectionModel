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

Selection (C3, C6): 60 % of the 955 POP-I planets (573 slots) get a Tier-2 slot, ranked by the mean present
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

**E3 fixed-budget test, committed before the data (2026-10-24, `e3_budget.py`).** The selection simulation
above gives the classifier a fallback (the AIRS-only learner) but no retrieval, and the triage table of F6
gives the retrieval a budget on clean spectra only, where every rule is within noise of random. The test that
E3 actually needs is a mixed population: each POP-I planet hazed with probability q (0.1, 0.2, 0.5), a
retrieval budget for a fraction f of the planets (10–50 %), each rule choosing which planets get it, and the
retrieval being the fallback a pipeline could run on any spectrum, the training model plus a haze with free
density (`identifiability_calibrate.py --free-haze`, running on the hazed population, 954 planets, then on
the clean one). Metrics: CH₄ (H₂O, joint) targets lost at the 60 % selection, the pipeline's accuracy, and
its recall on the hazed targets. Baselines: classifier alone, the five existing rules and random as
orderings, an oracle that knows which planets are hazed, the existing fallback (a clear-model retrieval,
of every planet and routed), and the free-haze retrieval of every planet.

Expectations, from the preview on the 124 planets with fixed-density posteriors (CH₄, q = 0.2, f = 20 %:
recall on hazed targets 4 % classifier alone, 27 % random budget, 87 % optical-excess budget, 88 % oracle,
83 % Mahalanobis, 4 % margin and ensemble; targets lost 4.0 → 2.1 against 2.6 random; the clear-model
retrieval of every planet loses 10.8 and recovers no hazed target): on the population with the free-haze
fallback, (i) the optical-excess ordering at f = 20 % raises CH₄ recall on hazed targets from below 15 % to
above 70 % and halves the targets the classifier alone loses; (ii) it beats every confidence rule (margin,
ensemble, band metric) by more than 40 points of hazed recall and random by more than 30; (iii) the
distance rules (Mahalanobis, PCA, k-NN), used as routers rather than decliners, come within 20 points of it,
so the claim is about the use (route a shift flag to a departure-aware retrieval), with the optical score
the most selective flag (99 % of hazed against 2 % of clean planets inside the top 20 %); (iv) routing the
existing clear-model retrieval is worse than no retrieval; (v) the free-haze retrieval of every planet loses
at most as many clean targets as the clean retrieval plus 5. If (i) fails, E3 is reported as not met.

**The free-haze retrieval on the hazed population (2026-10-27; `identifiability_calibrate.py --departure
haze3e7 --free-haze`; `results/identifiability_calibration_haze3e7_freehaze.{csv,log,txt}`).** All 954
hazed spectra completed, none at the cap or on a sampler abort (median 34 min and 6,033 calls per planet; 9
workers, 70.5 h wall, 635 core-hours). The haze density is recovered on every planet: median log₁₀ n 7.47
against 7.48 true, 68 % interval 0.11 dex wide and holding the true value on 95 % of planets, no posterior
at the haze-free end. Against the clean retrieval of the same planets: accuracy CH₄ 93.7 → 92.3, H₂O
91.6 → 87.3, CO₂ 93.8 → 93.7, NH₃ 93.9 → 91.4 %; recall of present CH₄ 88 → 82, H₂O 82 → 69, CO₂ 88 → 85,
NH₃ 88 → 81 %; decidable (p_true > 0.95) CH₄ 68 → 64, H₂O 46 → 34, CO₂ 56 → 63, NH₃ 68 → 61 %; confidently
wrong 0.0–0.3 %; median posterior shift of log X −0.07 to −0.15 dex. Of the 224 planets with CH₄ present
and decidable on the clean spectrum, 222 are recalled (99.1 %), 75 % stay decidable, median p_true 0.996.
Evidence median log Z −15.8 against −12.4 clean (clear-model −252); four planets (GJ 1252 b, Kepler-79 d,
TOI-544 b, TOI-6255 b) have log Z below −200 with the radius held fixed and keep their labels but one
(TOI-544 b H₂O). This supersedes the 124-planet fixed-density pilot above, which it agrees with (pilot
CH₄ 98 → 98 %, population 94 → 92 %), and completes E4: the haze's errors are recoverable by a retrieval
that models it, with one free parameter, and the same retrieval without that parameter is confidently
wrong on 99 % of the present labels.

**E3 fixed-budget test, outcome (2026-10-27; `e3_budget.py --fallback freehaze --draws 20`;
`results/e3_budget_freehaze.{csv,txt}`).** 954 planets, q = 0.1 / 0.2 / 0.5, 20 draws; the clean planets
carry the clean retrieval's posteriors until the free-haze run on the clean spectra completes (due 31
October). At q = 0.2, CH₄ (359 targets): classifier alone loses 28.9 and recalls 3 % of the hazed
targets; clear-model retrieval of every planet 68.5 and 0 %; free-haze retrieval of every planet 0.0 and
82 %. Budget orderings at f = 10 / 20 / 30 / 50 % (targets lost; hazed recall at f = 20 %): random
24.8 / 20.1 / 16.2 / 9.4 (17 %); optical 19.9 / 12.6 / 8.4 / 3.9 (80 %); oracle 20.6 / 12.3 / 8.9 / 4.5
(81 %); Mahalanobis 21.2 / 12.6 / 8.2 / 3.2 (64 %); PCA 21.4 / 14.3 / 10.2 / 4.2 (58 %); k-NN
21.5 / 13.5 / 7.0 / 2.8 (54 %); disagree 23.7 / 17.9 / 13.8 / 6.5 (37 %); band metric
24.6 / 21.3 / 18.1 / 10.2 (13 %); margin 24.9 / 19.6 / 16.4 / 11.8 (3 %); ensemble 26.5 / 22.2 / 18.2 / 12.8
(3 %); optical → clear-model 47.0 / 71.0 / 70.4 / 69.6 (0 %); random → clear-model 30.2 / 30.4 / 31.9 / 37.5.
Detection inside the top 20 %: optical 98 % of hazed, 1 % of clean; Mahalanobis 73 / 7; k-NN 77 / 6; PCA
67 / 9; disagree 44 / 14; margin and ensemble 0–1 / 25. H₂O at f = 20 %: optical 39.8 against classifier
72.8, oracle 39.5, random 58.1. At q = 0.1 optical 11.8 against oracle 12.2 (classifier 23.6); at q = 0.5 a
20 % budget cannot cover the hazed half (21.6 against oracle 22.9, classifier 34.2) and the ordering reaches
the oracle at f = 50 % (5.0 against 5.2).

Scoring the expectations: (i) met, recall 3 → 80 %, targets lost 28.9 → 12.6; (ii) met, 77 points over
margin and ensemble, 67 over the band metric, 63 over random; (iii) met for Mahalanobis (16 points), not
for PCA (22) and k-NN (26), and the selectivity claim met (98 / 1 %); (iv) met, 71.0 against 28.9; (v)
met on the hazed planets (0 CH₄ and 2.0 H₂O targets, the clean retrieval's own numbers), the clean-planet
half provisional until the clean free-haze run. E3 is met in the sense the editor asked for: the optical
score recognises the planets on which the classifier and the existing retrieval are unreliable (98 % of
the hazed in its top fifth), and used to allocate a retrieval that models the departure it improves both
classification (hazed-target recall 3 → 80 %) and target selection (29 → 13 targets lost, the oracle's
12) against the classifier alone, the existing rules, a random budget and the existing retrieval. What it
does not do is beat the free-haze retrieval of every planet, which costs 635 core-hours and loses nothing;
the diagnostic's value is at a budget below that.

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
of 0.8–0.9 is not a label-by-label agreement, and the posterior is the definition. The departure
columns of this table carry the clean spectrum's posterior, so they only say that the departures add
errors on labels the clean spectrum could decide; whether the departed spectrum still can is the
question below.

**Identifiability under haze (2026-10-20).** The Δχ² test rerun on the hazed population with the
haze in the counterfactual model as well (`identifiability.py --departure haze3e7`; 954 planets, one
errored; 251 core-hours; `results/identifiability_haze3e7_tier1.{csv,txt}`), so that it asks whether
the label is decidable given a correct model of the departure, with the haze density held at its
true value. At the calibrated thresholds the labels are about as decidable as on the clean spectra:

| decidable labels | clean | haze 3 × 10⁷ | both | clean only | haze only | median log₁₀ Δχ² ratio |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ | 71.3 % | 73.9 % | 68.0 % | 3.2 % | 5.9 % | +0.07 |
| H₂O | 43.9 % | 37.9 % | 33.5 % | 10.4 % | 4.4 % | −0.16 |
| CO₂ | 58.8 % | 76.0 % | 58.4 % | 0.4 % | 17.6 % | +0.42 |
| NH₃ | 71.1 % | 71.7 % | 65.6 % | 5.5 % | 6.1 % | −0.08 |

Present labels decidable: CH₄ 277 → 281 of 359, H₂O 194 → 166 of 364, CO₂ 205 → 280 of 386, NH₃
254 → 249 of 372. The haze costs H₂O a little (it sits under the NIR slope the haze adds) and makes
CO₂ more decidable, not less; the likely reason is that with the haze density known its optical and
NIR slope pins the scale height, which removes part of the temperature–abundance degeneracy the
counterfactual used, and a test that also fitted the haze density would give back some of that. The
information needed for the labels survives the haze, in other words: what the classifier and the
clear-model retrieval lose on hazy spectra is lost by their model, not by the spectrum.

**Departure-aware error classes (d‴, `alfnoor_diagnostic.py` section 3d,
`results/diagnostic_departure_classes.csv`).** Each learner's errors on the hazed spectra against the
haze-model Δχ² at the calibrated thresholds, split into errors the learner also makes on the clean
spectrum (shared) and errors the haze introduces; means over the four learners:

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| accuracy clean → haze | 84.3 → 66.6 % | 75.3 → 62.1 % | 80.7 → 63.6 % | 84.6 → 61.6 % |
| recall on decidable present labels, clean → haze | 83 → 14 % | 65 → 1 % | 82 → 13 % | 84 → 2 % |
| errors per learner, clean → haze | 150 → 319 | 236 → 362 | 184 → 347 | 147 → 366 |
| errors the haze introduces | 218 | 170 | 218 | 250 |
| of which recoverable (label decidable under the haze model) | 86.5 % | 63.5 % | 81.0 % | 78.7 % |
| errors shared with the clean spectrum | 100 | 192 | 130 | 116 |
| of which recoverable | 51 % | 29 % | 53 % | 40 % |
| clean errors the haze removes | 49 | 43 | 54 | 30 |
| all haze errors recoverable | 75 % | 45 % | 70 % | 67 % |

This is the editor's two-class split measured rather than argued: the errors the haze introduces are
four fifths recoverable for CH₄, CO₂ and NH₃ (two thirds for H₂O), while the errors the learners
share with their clean run are half or less recoverable on the Δχ² definition and a quarter or less
on the posterior one. The haze does not take the methane out of the spectrum; it takes the
classifier's recall of decidable present labels from 83 % to 14 %, which is the mechanism of F2 and
F5 on the population. B4 (at least 70 % of the haze errors recoverable) is met for CH₄ and CO₂ on
all errors and for CH₄, CO₂ and NH₃ on the introduced errors, on the Δχ² definition; the posterior
definition on the modelled-haze retrieval (below) is the stricter check, and agrees.

The white-noise cases need no run: the test's σ enters only as a uniform factor, so Δχ² under noise
× k is the clean Δχ² / k² with the same counterfactual, and the diagnostic computes them that way.
They are the opposite class. Under noise × 2 the undecidable fraction rises to 50/78/72/51 %
(CH₄/H₂O/CO₂/NH₃) and under × 3 to 65/87/86/67 %; the errors the noise introduces are 61/32/23/53 %
recoverable at × 2 and 56/23/10/41 % at × 3, all errors 45/20/17/36 % and 41/13/8/28 %. B4's noise
× 3 clause (at most 30 % of the errors recoverable) is met for H₂O, CO₂ and NH₃ and missed by CH₄
(41 %). So the two classes separate on the population as the editor asked: a haze error is in the
spectrum's reach and a noise error is not, and the same test tells them apart label by label.

**Identifiability under spots (2026-10-21).** The Δχ² test rerun with 20 % unocculted spots in the
spectrum and in the counterfactual model (`identifiability.py --departure spots20`; 955 planets, none
errored; 176 core-hours; `results/identifiability_spots20_tier1.{csv,txt}`). The spots change nothing
the labels need. At the calibrated thresholds (clean fractions here over all 955 planets; the haze
table's are over the 954 with a haze result):

| decidable labels | clean | spots 20 % | both | clean only | spots only | median log₁₀ Δχ² ratio |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ | 71.3 % | 71.6 % | 70.8 % | 0.5 % | 0.8 % | +0.05 |
| H₂O | 44.0 % | 46.5 % | 43.6 % | 0.4 % | 2.9 % | +0.05 |
| CO₂ | 58.8 % | 60.5 % | 58.2 % | 0.6 % | 2.3 % | +0.05 |
| NH₃ | 71.1 % | 72.7 % | 70.6 % | 0.5 % | 2.1 % | +0.05 |

Present labels decidable: CH₄ 277 → 275 of 359, H₂O 194 → 202 of 364, CO₂ 205 → 212 of 386, NH₃
254 → 256 of 372; the counterfactual temperature moves by 3 K at the median. The contamination
factor of unocculted cool spots deepens the transit by a few per cent at every wavelength, so Δχ²
rises by about 12 % at the median and a few labels just under the threshold cross it; fewer than 1 %
of labels in any molecule lose decidability.

Departure-aware error classes under the spots (section 3d, `results/diagnostic_departure_classes.csv`;
means over the four learners):

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| accuracy clean → spots | 84.3 → 74.6 % | 75.3 → 67.8 % | 80.7 → 70.3 % | 84.6 → 68.8 % |
| recall on decidable present labels, clean → spots | 83 → 44 % | 65 → 30 % | 82 → 45 % | 84 → 28 % |
| errors per learner, clean → spots | 150 → 242 | 236 → 307 | 184 → 284 | 147 → 298 |
| errors the spots introduce | 134 | 110 | 146 | 180 |
| of which recoverable (label decidable under the spot model) | 79.3 % | 65.0 % | 57.0 % | 79.9 % |
| errors shared with the clean spectrum | 108 | 197 | 138 | 118 |
| of which recoverable | 49 % | 35 % | 29 % | 36 % |
| clean errors the spots remove | 41 | 38 | 46 | 29 |
| all spots errors recoverable | 66 % | 47 % | 43 % | 62 % |

The spots are the haze mechanism at lower amplitude: the labels stay as decidable as on the clean
spectra (undecidable 28/54/39/27 % against 29/56/41/29 %), the classifier loses 10–16 points and
half to two thirds of its recall on decidable present labels, and the errors it adds are four fifths
recoverable for CH₄ and NH₃, two thirds for H₂O and 57 % for CO₂, against 10–56 % for the errors
threefold noise adds. The spots' errors fall on undecidable labels more often than the haze's do (all
errors 43–66 % recoverable against 45–75 %), which is a smaller departure adding fewer errors on the
decidable labels, not a loss of information: the decidable fractions did not move.

**Identifiability under the deep deck (2026-10-22).** The Δχ² test rerun on the population rendered
with a grey deck at 10² Pa, two decades above the prior's 5 × 10² Pa top, with the deck held at its
true pressure in the counterfactual model and temperature and the other abundances free
(`identifiability.py --departure cloud1e2`; 955 planets, none errored; 106 core-hours;
`results/identifiability_cloud1e2_tier1.{csv,txt}`). Unlike the haze and the spots, the deck removes
information. At the calibrated thresholds:

| decidable labels | clean | deck 10² Pa | both | clean only | deck only | median log₁₀ Δχ² ratio |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ | 71.3 % | 69.8 % | 64.7 % | 6.6 % | 5.1 % | −0.02 |
| H₂O | 44.0 % | 27.9 % | 26.1 % | 17.9 % | 1.8 % | −0.36 |
| CO₂ | 58.8 % | 53.2 % | 48.7 % | 10.2 % | 4.5 % | −0.10 |
| NH₃ | 71.1 % | 63.0 % | 58.0 % | 13.1 % | 5.0 % | −0.22 |

The loss falls almost entirely on absent labels: present labels decidable CH₄ 277 → 286 of 359, H₂O
194 → 173 of 364, CO₂ 205 → 198 of 386, NH₃ 254 → 260 of 372, against absent labels decidable 404 →
381 of 596, 226 → 93 of 591, 357 → 310 of 569 and 425 → 342 of 583. The deck hides the deep
atmosphere where a small abundance would have shown, so the floor on an absence rises and an
"absent" label that the clear spectrum could prove becomes undecidable; a present label above the
threshold still shows in the bands above the deck. The counterfactual temperature moves by 23–55 K
at the median, so the deck also changes the temperature–abundance degeneracy the test exploits.

Departure-aware error classes under the deck (section 3d, `results/diagnostic_departure_classes.csv`;
means over the four learners):

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| undecidable labels under the deck | 30 % | 72 % | 47 % | 37 % |
| accuracy clean → deck | 84.3 → 75.1 % | 75.3 → 67.3 % | 80.7 → 73.5 % | 84.6 → 69.8 % |
| recall on decidable present labels, clean → deck | 83 → 45 % | 65 → 36 % | 82 → 58 % | 84 → 34 % |
| errors per learner, clean → deck | 150 → 238 | 236 → 312 | 184 → 253 | 147 → 288 |
| errors the deck introduces | 129 | 107 | 104 | 166 |
| of which recoverable (label decidable under the deck) | 81.7 % | 51.2 % | 51.5 % | 76.1 % |
| errors shared with the clean spectrum | 110 | 205 | 148 | 122 |
| of which recoverable | 51 % | 27 % | 26 % | 39 % |
| clean errors the deck removes | 40 | 30 | 36 | 25 |
| all deck errors recoverable | 68 % | 36 % | 36 % | 61 % |

So the deck is a third case between the haze and the noise. On CH₄ and NH₃, whose bands sit high,
its errors are four fifths recoverable as under haze; on H₂O and CO₂ only half, because the deck has
made undecidable a sixth of the H₂O labels and a tenth of the CO₂ labels that the clean spectrum could
decide, and the classifier's errors on those are the atmosphere's, not its own. The three departure
families therefore separate on the population: haze and spots take nothing from the spectrum
(recoverable 57–87 % of introduced errors, decidability unchanged), threefold noise takes information
through the instrument (10–56 %, decidability halved), and a deep deck takes it through the
atmosphere (51–82 %, decidability down 2–16 points, on absent labels). The table of departures is
complete.

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
spectra is the test below.

**The clear-model retrieval on hazy spectra (2026-10-20; `identifiability_calibrate.py --departure
haze3e7`, `results/identifiability_calibration_haze3e7.{csv,txt}`).** Sixty planets, half of them
CH₄-present and decidable on the clean retrieval, half random; the same nested-sampling retrieval,
with the training forward model (clear H₂/He, grey deck, no haze), on the 3 × 10⁷ m⁻³ hazed spectrum
at Tier-1 noise. Three planets errored in the sampler on the optical excess, one hung past five hours
in nestle's uncapped rejection loop and was stopped (the script now has a per-planet wall-clock
cap), so 55 have posteriors. Against the clean retrieval of the same planets:

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| accuracy clean → hazed | 98 → 36 % | 87 → 62 % | 87 → 55 % | 96 → 64 % |
| recall of present labels | 97 → 0 % | 73 → 5 % | 72 → 0 % | 89 → 0 % |
| confidently wrong (p_true < 0.05) | 0 → 64 % | 0 → 38 % | 0 → 45 % | 0 → 36 % |
| median posterior shift of log X | −4.7 dex | −3.4 dex | −4.2 dex | −3.4 dex |

Of the 30 planets with CH₄ present and decidable on the clean spectrum, the hazed retrieval recovers
none, with the posterior of the true class at 0.000 on every one. The accuracies equal the absent
fractions: every abundance is driven to the prior floor, the retrieval says "absent" for everything,
and it says so with full confidence. Its evidence shows the misfit (median log Z −13 on clean
spectra, −260 on hazed, down to −2 × 10⁵), so an analyst looking at the fit would see it; a pipeline
that reads only the posterior would not. The departure is largest where the classifier's failure
was: with the retrieval's free offset removed, the haze-minus-clean difference is 18 σ per bin in
the three optical points (median over planets; 4 to 170), 4.6 σ over NIRSpec, 1.8 σ over AIRS; the
χ² per bin of the departure is 12 over all 104 bins and 3 over the AIRS bins alone. So the AIRS
bins are not untouched (the haze at this density adds some 450 ppm there, partly absorbed by the
offset), but the optical and NIR slope is what the clear model cannot produce, and the fit buys it
down with the infrared.

For the paper this closes the gap in E3. On hazy spectra the "existing method" with the same
physics as the training set is not a fallback: it fails on the same planets, for the same reason,
and more confidently than the classifier (which at least reverts to "absent" with low margin). The
labels are decidable (the haze identifiability above), so these are recoverable errors in the
editor's sense, recoverable by a model that carries the haze, which is the modelled-haze retrieval
(running on the same sixty) and, at the classifier's cost, the AIRS-only routing of F5. The remaining
caveat: the retrieval
holds the planet radius fixed with only a ±5 × 10⁻³ depth offset, so a retrieval with a free radius
and a haze parameter, as the consortium would run on a flagged planet, is a different and fairer
comparison, which is what the modelled run measures.

**The clear-model retrieval on the hazed population (2026-10-24; `identifiability_calibrate.py
--departure haze3e7 --n 1000`, `results/identifiability_calibration_haze3e7.{csv,txt}`; 500
core-hours).** The sixty-planet pilot above is superseded. The same nested-sampling retrieval, with
the training forward model (clear H₂/He, grey deck, no haze), on the 3 × 10⁷ m⁻³ hazed spectrum of
every one of the 954 planets with one, at Tier-1 noise. It failed to complete on 18: 14 sampler
aborts on a degenerate likelihood ("array must not contain infs or NaNs" in nestle's ellipsoid
update, when every live point sits at the same misfit) and 4 at the three-hour wall-clock cap, against
no failure in 955 on the clean spectra. (One planet ran past four hours under the cap because the
alarm's exception was raised inside the forward-model call and swallowed by its catch-all; the
likelihood now re-raises it; the run was stopped and its last eight planets rerun.) On the 936 it
completed, against the clean retrieval of the same planets:

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| accuracy clean → hazed | 93.6 → 62.5 % | 91.5 → 61.5 % | 93.7 → 59.3 % | 93.8 → 60.3 % |
| recall of present labels | 88.1 → 0.6 % | 81.3 → 0.6 % | 88.2 → 0.3 % | 88.3 → 0.3 % |
| present labels confidently wrong (p_true < 0.05) | 99.4 % | 99.4 % | 99.7 % | 99.7 % |
| all labels confidently wrong | 0.0 → 37.5 % | 0.0 → 38.5 % | 0.0 → 40.7 % | 0.1 → 39.7 % |
| median posterior shift of log X | −3.6 dex | −3.5 dex | −4.1 dex | −3.8 dex |

The accuracies are the absent fractions: the retrieval calls every present label absent, with the
posterior of the true class below 0.05 on 99 % of them. Of the 219 planets with CH₄ present and
decidable on the clean spectrum it recovers two (recall 0.9 %, median p_true 0.000). The evidence
shows the misfit (median log Z −12.3 on clean spectra, −252 on hazed, down to −4 × 10⁸). The pilot's
headline numbers (98 → 36 % accuracy, 64 % confidently wrong) were those of a sample chosen half
from CH₄-present decidable planets; on the population the accuracy loss is 31–34 points and the
confident-error rate 38–41 %, and the mechanism is the same.

**The modelled-haze retrieval on the same sixty (2026-10-21; `identifiability_calibrate.py --departure
haze3e7 --model-departure`, `results/identifiability_calibration_haze3e7_modelled.{csv,txt}`; 60 of 60,
median 33.6 min and 6,062 calls per planet).** The same retrieval with the haze in the forward model
(density held at its true value, as in the Δχ² counterfactual; not a free parameter) on the same hazed spectra, against the clean retrieval of the same planets:

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | :-- | :-- | :-- | :-- |
| accuracy clean → hazed, modelled | 98 → 98 % | 88 → 87 % | 88 → 92 % | 97 → 92 % |
| recall of present labels | 97 → 97 % | 74 → 74 % | 74 → 81 % | 90 → 80 % |
| decidable (p_true > 0.95) | 85 → 83 % | 50 → 40 % | 57 → 70 % | 65 → 60 % |
| confidently wrong | 0 → 0 % | 0 → 0 % | 0 → 0 % | 0 → 0 % |
| median posterior shift of log X | −0.00 dex | −0.10 dex | +0.01 dex | −0.11 dex |

Of the 33 planets with CH₄ present and decidable on the clean spectrum, the modelled retrieval recovers
all 33 (median p_true 1.000) and 94 % stay decidable; the evidence is unchanged (median log Z −13.4 →
−13.5). So the posterior says what the Δχ² test said: the haze removes no information the labels need
(H₂O loses a little decidability, CO₂ gains some, as in the Δχ² table above), and the clear-model
retrieval's zero recall and 38–41 % confident errors on the same spectra are entirely the model's. This is
the posterior-based form of B4, met for all four molecules on the sixty. The full-population modelled
run (894 planets, 8 workers) started 2026-10-24 18:25 and reports on 2026-10-26.

### F7. Muting census: six of the fourteen real JWST spectra are more muted than any atmosphere the simulator population can produce at their bulk parameters, and the distance rule cannot see it because it normalises amplitude away

For each real planet with a converted spectrum, the dispersion of its measured Tier-3 bins (standard
deviation of the depths, with the per-bin errors removed in quadrature) against the same dispersion of
its clear H₂/He twin at the published abundances over the same bins; the twin under grey decks from the
population's lowest allowed cloud top (5 × 10² Pa) down to 1 Pa; the twin at the population's coldest
allowed temperature (0.7 T_eq, the floor of its prior) with the 5 × 10² Pa deck, which is the most muted
atmosphere the population can draw at those bulk parameters; and the cloud-only muting of a random
whole of POP-I (955 planets; the first pass used a sample of 150), their rendered spectrum against their own clear twin
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
percentile 1.08, 99th 2.27 and maximum 5.56 of 955.

Every warm giant with a published abundance set (HAT-P-26 b, WASP-107 b, WASP-39 b, GJ 3470 b, WASP-166 b)
is muted by a factor of 3 to 5 against its clear twin, and by 1.5 to 2.2 beyond the coldest, cloudiest
atmosphere the population allows at its mass, radius and irradiation. For five planets no grey deck at any
pressure in the model reaches the real amplitude, because their photospheres sit above the population's
cloud floor. Within the population, clouds mute the median planet by 1 %, the 90th percentile by 23 %,
the 99th by a factor of 2.5, and the most muted of 955 by 6.3 (TOI-836.01, a deck at 649 Pa, just above the prior's
5 × 10² Pa floor, on a small cool planet). Six planets of 955 (0.6 %) are muted as much as WASP-166 b, three (0.3 %) as
much as WASP-39 b, and one as much as WASP-107 b or HAT-P-26 b; so the population can reach the giants' muting, but
only at its cloud floor and in fewer than one planet in a hundred, whereas five of five real giants sit there. The three sub-Neptunes with retrieved methane-rich atmospheres (K2-18 b,
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

### F8. Retraining on a departure-aware population recovers the real methane planets whose spectra are muted beyond the simulator's prior, and not the two that are not; it costs a point on clean spectra and does not beat the AIRS-only routing

The editor's E4 asked for the two error classes on real data. The test (`alfnoor_retrain.py`,
2026-10-20): a third of the faithful POP-III training planets re-rendered with a Lee–Mie haze of
density log-uniform in 10⁶–3 × 10⁸ m⁻³ ("haze"), and in a second set a further sixth with a grey deck
log-uniform in 5–500 Pa, two decades above the population prior's 5 × 10² Pa top ("hazedeck");
1,954 of 1,996 renders succeeded (the rest keep their clean spectrum). The four learners retrained
with the same binning, noise draw and normalisation as the faithful run, for the three input
variants, and scored on POP-I and on the real panel exactly as before
(`results/alfnoor_retrain_pop1.{csv,txt}`, `results/alfnoor_retrain_panel.{csv,txt}`,
`results/jwst_panel_{haze,hazedeck}/`). The reviewer called this "familiar domain randomization";
it is used here as the test of the error class, not as the contribution.

On the simulated population, all inputs, accuracy / recall of present, mean over the four learners:

| | clean | haze 3 × 10⁷ | cloud 10² Pa | spots 20 % | noise × 3 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| CH₄ faithful | 84.9 / 72 | 66.3 / 11 | 75.4 / 39 | 74.6 / 35 | 70.3 / 28 |
| CH₄ haze | 83.6 / 72 | 78.9 / 49 | 77.9 / 46 | 80.7 / 60 | 70.6 / 31 |
| CH₄ hazedeck | 83.9 / 74 | 79.7 / 52 | 79.5 / 52 | 81.2 / 61 | 71.1 / 34 |
| H₂O faithful | 74.9 / 46 | 62.2 / 1 | 67.7 / 22 | 67.5 / 19 | 65.6 / 20 |
| H₂O hazedeck | 72.4 / 43 | 66.7 / 22 | 67.9 / 25 | 69.9 / 40 | 64.1 / 20 |
| CO₂ faithful | 79.5 / 65 | 63.5 / 11 | 73.2 / 42 | 70.2 / 31 | 66.3 / 23 |
| CO₂ hazedeck | 79.2 / 66 | 75.4 / 46 | 76.5 / 53 | 77.3 / 58 | 66.6 / 26 |
| NH₃ faithful | 84.5 / 68 | 61.7 / 2 | 70.3 / 27 | 69.2 / 22 | 68.4 / 25 |
| NH₃ hazedeck | 82.6 / 66 | 71.9 / 32 | 73.0 / 37 | 74.8 / 41 | 68.3 / 30 |

The retrained classifier gives back two thirds of the CH₄ haze loss (66 → 79 against 85 clean), half
of the CO₂ and NH₃ loss, a quarter of the H₂O loss, and most of the spots loss it was never trained
on; the deep decks add a point or two more under the deep-deck departure; the noise × 3 column does
not move, which is the irrecoverable class again. The cost is 1–2 points on clean spectra
(CH₄ −1.0 to −1.3, H₂O −2.4, CO₂ −0.3 to −0.8, NH₃ −1.7 to −1.9), which the paper must state. Under
haze the retrained all-inputs classifier (78.9 / 49) is level with the frozen AIRS-only one
(78.9 / 54); the AIRS-only variant itself is unchanged by retraining (80.4 clean, 79.7 haze), as it
should be, since the haze enters through the points it does not use.

On the real panel (threshold 10⁻⁴, verdict = majority of the four learners), scored labels correct:

| inputs | faithful | haze | hazedeck |
| :-- | :-- | :-- | :-- |
| all | 22 / 35 | 26 / 36 | 27 / 36 |
| no optical | 25 / 36 | 27 / 38 | 26 / 36 |
| AIRS only | 29 / 37 | 29 / 36 | 29 / 38 |

The CH₄-present reductions with all inputs, against the muting census of F7 (the deck the real
spectrum needs beyond the population's 5 × 10² Pa floor to match its muting):

| reduction | deck needed (F7) | faithful | haze | hazedeck | AIRS-only (any training) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| K2-18 b alt | 238 Pa | absent (1 of 4) | split (2) | present (4) | present |
| K2-18 b hu | 117 Pa | absent (1) | present (3) | split (2) | present |
| TOI-1231 b | 47 Pa | split (2) | present (3) | present (4) | present |
| K2-18 b | none | absent (0) | absent (0) | split (2) | present |
| TOI-270 d | none | absent (1) | absent (1) | absent (1) | split → absent |
| TOI-270 d hm | within prior | absent (0) | absent (1) | absent (1) | present |

The three reductions whose muting lies beyond the simulator's aerosol prior are the three the
retrained classifier recovers; the three within the prior it does not (K2-18 b's primary reduction
reaches a split under the deep decks). So on the real planets the error classes split the same way
as on the population: where the failure is the training prior, extending the prior repairs it, and
the paper can say which planets those are and why. Where it is not (TOI-270 d in both reductions,
K2-18 b primary, with Tier-1 detection frequencies of 7–30 %), neither retraining nor the optical
diagnosis helps with all inputs, and only the AIRS-only variant finds K2-18 b. The retrained
classifier also moves the CO₂ and H₂O verdicts of the warm giants (WASP-166 b H₂O and CO₂, GJ 3470 b
and HAT-P-26 b CO₂ to present, correctly; HD 189733 b CO₂ to present, wrongly), consistent with the
same mechanism: the muted giants were being read as featureless.

For the paper: the retraining is one paragraph of the E4 section, as the test that the real-planet
errors of F2 are of the recoverable class for the muted planets and not for the others, with the
clean-accuracy cost and the fact that it recovers fewer CH₄ reductions (2–3 of 6) than the AIRS-only
routing (5 of 6) at no retraining. It is not the contribution; the error-class result is.

**Four independent reviews of the manuscript, and what they changed (2026-10-28).** Four reviewers (a
referee in the field, a methods and reporting referee, a numbers auditor and an editorial reader) read the
28 October draft independently. Two found, and the code confirms, that every retrieval in the paper runs on
the noise-free spectrum with the Tier-1 σ only in the likelihood (`identifiability_calibrate.py` adds a noise
draw only under `--noisy`), while the classifier is scored on one noisy realisation; the paper did not say so.
They also found that the retrieval's abundance prior (10⁻⁹–10⁻², POP-III's) is wider than POP-I's
(10⁻⁷–10⁻²), so an uninformative spectrum already gives p_true = 5/7 = 0.71 for an absent label and 0.29 for a
present one: an absent label reaches 0.95 on a Bayes factor of 8, a present one needs 47, and the retrieval's
"ceiling" (92–94 %) is 97–98 % on absent labels and 82–88 % on present ones. Other verified items: the five
decline rules' thresholds are quantiles of the clean POP-I *test* scores (`np.quantile(s0[valid], 1 -
DECLINE)` on `S["clean"]`), not held out, so "flags 10 % of clean" is by construction; Table 8's random row is
one draw; the 0.84 level, the introduced/shared split and the E3 expectations (set after the 124-planet pilot
of the same test) are post hoc relative to the 16 October file; Table 10 omitted several committed
expectations (A2's CO₂ clause, A4, A7, P4, P5, C4, C5, and B5–B7, never run); Jaeger et al. (ref. 16) is cited
for a claim its paper argues against; and twenty numbers disagreed with the result files (the auditor's list:
14 not 15 of 16 pairs, 7–17 % not 7–16 %, −17.5 not −17.8, TOI-270 d's AIRS-only retrained verdict, the
H₂O labels of the three AIRS-only planets counted in 22/35, filled-bin ranges, retraining shares, muting 3.1–5.3
and 1.7–2.2, 0.5 not 0.3 points, a third not a sixth of H₂O labels, H₂O omitted from two ranges, NH₃ 19 % at
0.84, 98 % against ≥ 99 %, AUROC 0.47–0.63, 0.44–0.52, CO₂ shared 53 %). All are corrected in the rewrite.

**The retrieval on the classifier's own noisy realisation, clean spectra (2026-10-28;
`identifiability_calibrate.py --noisy --random-sample --n 200 --jobs 9`, 7.4 h; `noisy_vs_noisefree.py`).**
A fixed random 200 of the 955 planets (SEED + 11), retrieved on the same noise draw the classifier reads
(SEED + 405, the training draw consumed first). Against the noise-free posteriors of the same planets:
decidable fractions at 0.95 move by at most 5 points (CH₄ 70.5 → 65.0 %, H₂O 44.0 → 48.5, CO₂ 55.0 → 55.5,
NH₃ 66.5 → 65.5), but the labels move under them: 5–11 % of labels lose decidability under the draw and 5–12 %
gain it, the correlation of p_true between the two runs is 0.67–0.79. The retrieval's accuracy on the
realisation is 3–6 points lower (CH₄ 95.5 → 92.5, H₂O 90.0 → 84.0, CO₂ 95.0 → 89.5, NH₃ 92.5 → 91.0 %), and
labels with p_true < 0.05 appear: 0 of 800 noise-free, 8 of 800 (1.0 %) on the realisation; the "one misled
label in 3,820" of the noise-free population is a property of noise-free data. Of the four learners' 590
errors on these 200 realisations, 18.5 % are "recoverable" by the noise-free posterior (CH₄ 29.6, H₂O 16.1, CO₂
14.1, NH₃ 17.0 %), 14.9 % are decidable by the noisy posterior, and the noisy retrieval gets 60 % of them
right at p ≥ 0.5 (noise-free: 73 %), most of that from the prior's absent default. So on clean spectra the
recoverable share defined by the expected posterior is an upper bound that overstates by about a fifth of
itself, the population prevalence of undecidable labels is robust to the definition, and the retrieval's
confident-error rate on data is about 1 % rather than 0.03 %. The hazed-spectrum runs (free-haze and
clear-model on the same 200 realisations) follow.

**Cheap checks from data in hand (2026-10-28).** Table 8's random row over 200 draws (`triage_spread.py`):
CH₄ 16.2 ± 1.4, 12.3 ± 1.6, 9.5 ± 1.7, 4.4 ± 1.6, 1.7 ± 1.1 targets at f = 10–70 %, so k-NN (8 at 20 %)
and Mahalanobis (9) beat random by 2–3 standard deviations at f = 20–30 % and nothing else does; for H₂O
every rule is within noise of random or worse. Table 9 spreads over the 20 draws (q = 0.2, f = 20 %): optical
12.6 ± 2.1, oracle 12.4 ± 1.9, Mahalanobis 12.6 ± 1.7, k-NN 13.5 ± 2.0, PCA 14.4 ± 1.8, random 20.1 ± 3.6,
margin 19.6 ± 3.5, optical → clear-model 71.0 ± 7.7; hazed recall optical 80 ± 4, oracle 81 ± 4, Mahalanobis
64 ± 4 %. The optical score, the oracle and the Mahalanobis rule are within one standard deviation of each
other in targets lost; the score's advantage over the distance rules is recall and selectivity, not targets at
this budget. Decidability by Bayes factor > 19 against the retrieval prior's odds instead of p_true > 0.95:
undecidable 39/62/50/35 % (CH₄/H₂O/CO₂/NH₃) against 32/54/44/32 %; by class at 0.95, present labels are
undecidable in 38/57/58/46 % and absent in 29/52/35/23 %. Muting bootstrap (`muting_bootstrap.py`, 2,000
resamples of the real depths with their errors, correction for the doubled noise): the five giants with
published abundance sets keep 68 % intervals of 4.35–4.55 (WASP-107 b), 5.13–5.55 (HAT-P-26 b), 3.11–3.95
(GJ 3470 b), 2.66–3.86 (WASP-166 b) and 3.67–3.79 (WASP-39 b), beyond the population's maximum at their bulk
parameters in 100 % of resamples; the sub-Neptunes' intervals (0.6–2.6) sit inside the population's range.

### F8b. The free-haze repair of hazed spectra survives the noise realisation

Follow-up to the noisy-realisation check on clean spectra (above). `identifiability_calibrate.py
--departure haze3e7 --free-haze --noisy --random-sample --n 200` finished 2026-10-28 22:38 (15.5 h,
median 43 min per planet); `noisy_vs_noisefree.py --departure haze3e7 --free-haze` writes
`results/noisy_vs_noisefree_haze3e7_freehaze.{txt,csv}`. Against the noise-free free-haze retrieval of
the same 200 hazed spectra: decidable fractions 64.5 → 63.0 % (CH₄), 33.5 → 37.5 (H₂O), 62.5 → 64.0
(CO₂), 60.5 → 60.5 (NH₃); 6.5–9.5 % of labels lose decidability under the draw and 7.5–10.5 % gain it;
corr(p_true) 0.72–0.76; accuracy 96.5 → 93.0, 84.0 → 80.0, 94.5 → 88.0, 90.5 → 86.5 %; confidently
wrong (p_true < 0.05) 2 → 12 of 800. Of the four learners' 1,177 errors on the same realisation,
37.8 % are recoverable by the noise-free posterior and 38.1 % decidable by the noisy one; the noisy
retrieval gets 72.3 % of them right (noise-free 78.9 %). So Table 9's repair is of the realisation,
not only of the expectation. Added to Section 3.6 and the limitations.

### F8c. The clear-model failure on hazed spectra is noise-invariant: the misled class is a property of the model, not of the draw

`identifiability_calibrate.py --departure haze3e7 --noisy --random-sample --n 200` finished 2026-10-29
11:45 (13.1 h, median 32 min per planet; 196 completed, 2 timeouts and 2 inf/NaN failures against 18 of
954 noise-free); `noisy_vs_noisefree.py --departure haze3e7` writes `results/noisy_vs_noisefree_haze3e7.{txt,csv}`.
Against the noise-free clear-model retrieval of the same hazed spectra nothing moved: decidable
fractions identical to the tenth of a point (CH₄ 62.2 / 62.2 %, H₂O 59.7 / 59.7, CO₂ 62.2 / 62.2, NH₃
59.7 / 59.7, all of it absent labels; present labels 0 → 0 % decidable), no label lost or gained
decidability, corr(p_true) 1.00, median |Δp_true| 0.000, confidently wrong labels 74 → 74 (CH₄, CO₂)
and 79 → 79 (H₂O, NH₃) of 196, the largest p_present on any present label 6 × 10⁻⁷ → 2 × 10⁻⁴, median
abundance shift 0.03–0.04 dex, and the evidence fell a further 47 in median log Z (−259 → −306).
Reading: on a right model the noise draw moves labels by a few points (F8, F8b); on a wrong model it
moves nothing, because the posterior is pinned by the misfit, not the data. So the recoverable /
irrecoverable split survives the realisation on both sides, and a "confidently wrong" rate measured on
noise-free spectra under a wrong model is the rate on data. Added to Section 3.6 and the limitations.

### F9. Dose-response of the diagnostic: at a haze density that costs 12 points of accuracy, the optical flag sees 60 % of the hazed planets, the CH₄ selection loses nothing, and the loss moves to H₂O where the routing recovers a seventh of it

The mixed-population test (F5/F6, Table 10) hazed every planet at 3 × 10⁷ m⁻³, where the optical score
flags 98 % of hazed planets at a 1 % clean rate; the manuscript listed the dose-response at weaker
densities as not reported. On 2026-10-28 POP-I was re-rendered with the Lee–Mie haze at 2 × 10⁶ m⁻³
(the lowest density at which the stress test measured a loss: 69.0 % accuracy, −12.1 points, Table 4;
`alfnoor_trust.py render` writes the same file) and `e3_budget.py --haze haze2e6 --q 0 0.1 0.2 0.5 1.0`
was run with no fallback posteriors yet (`results/e3_budget_freehaze_haze2e6_noretrieval.{csv,txt}`;
955 planets, 20 draws). Findings:

* **Everyone hazed at 2 × 10⁶ (q = 1):** CH₄ accuracy 86.4 → 72.7 %, CH₄ recall at 0.5 75 → 27 %, yet
  CH₄ targets lost at the 60 % selection 20 → 19: the haze flattens the probabilities but keeps the
  CH₄ ranking. H₂O targets lost 51 → 86 (accuracy 78.2 → 64.1 %, recall 51 → 7 %); joint 95 → 122.
* **Routing by the optical flag to the AIRS-only variant (no retrieval):** CH₄ 19 (unchanged), hazed CH₄
  recall 27 → 54 %; H₂O 86 → 81 (5 of the 35 added targets recovered); joint worse, 122 → 139, because
  the AIRS-only variant is weaker on H₂O. At 3 × 10⁷ the same routing recovered 12 of the 18 CH₄
  targets the haze added (Table 11).
* **Detection at q = 0.2, top 20 % most suspect (hazed / clean inside):** optical 60 / 10 % (98 / 1 % at
  3 × 10⁷); PCA 56 / 11; Mahalanobis 46 / 14; k-NN 31 / 17; input disagreement 29 / 18; band metric
  14 / 21; margin 6 / 23; ensemble 9 / 23. The confidence rules again hold fewer hazed planets than
  clean ones. At f = 50 % the optical and PCA scores hold 86 % of the hazed planets.
* **Mixed population q = 0.2:** classifier alone 19.1 ± 1.1 CH₄ targets lost (28.9 ± 4.0 at 3 × 10⁷),
  routed 20.0 ± 1.6; H₂O 62.1 ± 2.3 vs routed 60.8 ± 1.8.

Reading: the departure the flag detects fully is also the one that moves CH₄ targets out of the
selection; the one it detects by half costs accuracy and H₂O targets but almost no CH₄ targets, and the
no-retrieval remedy recovers little there. What the budgeted free-haze retrieval buys at 2 × 10⁶ is
queued on the 200-planet random subsample (`identifiability_calibrate.py --departure haze2e6 --free-haze
--random-sample --n 200`, then the clear model on the same planets for a Table 9 row), behind the
noisy-realisation chain. Expectation written before that run: the free-haze retrieval recovers most of
the H₂O loss on the planets it reaches, but with the flag at 60 % sensitivity a 20 % budget routed by it
will sit between random and the oracle rather than on the oracle (at 3 × 10⁷ it matched the oracle).
Added to the manuscript in Sections 2.7, 3.6 and the limitations; the response letter's E3 paragraph
states it.

### F10. The reproduction's deviations from the published accuracies are sampling scatter with a sub-point systematic offset; the two non-monotone pairs move with the seed

Asked why "two of sixteen pairs differ" was called unfixable: it was mis-stated. The 14 of 16 is the
monotone-with-threshold pattern, broken for CO₂ at 10⁻⁵ by the forest (79.6 vs 77.7 at 10⁻⁴) and the
support vector (81.3 vs 80.9), deviations of 2–4 points from the published cells. `alfnoor_faithful.py
--fit --noise radiometric --seed N` (new option; outputs tagged `_seedN`) refit the design with five other
noise draws and learner seeds (2026-10-28, 2.5 min each); `reproduction_seeds.py` compares
(`results/reproduction_seeds.{txt,csv}`): seed sd per cell median 0.71, max 1.77 points; the published
value lies inside the seed range in 32 of 48 cells, so seed noise alone does not explain the deviations;
adding the binomial scatter of one 954-planet test set (the published run is one population draw) and
the integer rounding gives 1.5 points per cell, against which 5 of 48 deviations exceed 2σ (2.3 expected),
none 3σ, χ² 68 for 48 cells; mean signed offset by threshold +0.56, −0.15, +0.91 and by molecule +0.07 to
+0.70. The monotone pairs are 12–14 of 16 across the six fits. Manuscript 3.1 and Table A1 now state the
scatter and the sub-point residual.

### F11. Cheap rigour checks: the muting score in simulation is a null result, decline thresholds held out, Table 11 over noise draws

* **Muting score** (`alfnoor_diagnostic.Scorer.muting`, new rule "muting"): the simulated form of the
  census (F7), the noise-corrected dispersion of the AIRS bins against the 10th percentile of the 50
  training spectra nearest in (modulation_5H, T_eq, T_host, R_p). Over all 104 bins the Lee–Mie haze
  *raises* dispersion (its optical slope), so the score was restricted to the AIRS bins; there it flags
  10–14 % of hazed, 8 % of spotted and 14 % of deep-deck spectra at a 10 % clean rate (AUROC 0.43–0.58),
  and its selective credit is as negative as every other rule's (−9 to −19). The simulated departures
  stay inside the amplitude range the training decks (5 × 10²–10⁶ Pa) already span; the real giants
  (1.7–2.2× beyond) do not. Excess white noise, by contrast, it separates almost perfectly in the other
  direction (AUROC 0.04/0.02 for ×2/×3), i.e. a variance-excess check. Reported as one sentence in the
  Discussion; not a router candidate in the budget test. The full diagnostic was rerun with the rule;
  every other rule's row is unchanged.
* **Held-out thresholds** (`threshold_holdout.py`, `results/threshold_holdout.{txt,csv}`): thresholds set on a random
  half of POP-I and applied to the other half, 20 splits. Every rule's decline rate under every departure is
  reproduced within 0.5 points of the in-sample value (e.g. optical under haze 99.8 vs 99.8 %, Mahalanobis 79.4
  vs 79.3 %, k-NN under spots 45.6 vs 45.4 %); the clean rates are 7.6-10.7 % (margin 7.6 % because of ties,
  as in-sample). The "thresholds not held out" limitation is closed; stated in the limitations.
* **Table 11 over noise draws** (`table11_spread.py`, `test_cases(noise_seed=d)`, `results/table11_spread.{txt,csv}`;
  ten draws, learners fixed): CH₄ targets lost, classifier alone 16.0 ± 3.1 clean / 35.8 ± 4.8 haze / 32.6 ± 4.4
  spots / 37.4 ± 3.2 noise ×2; routed by the optical flag 17.1 ± 3.1 / 26.8 ± 4.8 / 27.9 ± 3.4 / 37.6 ± 2.9;
  AIRS-only everywhere 25.8 / 27.6 / 29.0 / 48.4. The committed draw (20 / 38 / 29 / 37; routed 19 / 26 / 28 / 37)
  is a high clean draw, so "recovers 12 of the 18 targets haze adds" becomes 9 of 20 on average (half, not two
  thirds), and the spot benefit (1 in the committed draw) is 5 of 17 on average. Declined-to-the-back rules 22-108;
  band metric alone 107-112; random 142-147. Manuscript 3.6, Table 11 caption and limitations updated.

### F12. The irrecoverable class is recognised in part by the classifier's own confidence, no cheap flag does better, and perfect recognition is worth four targets in 359

Asked why the paper has no cheap, novel diagnostic. The paper's "nothing cheap recognises the
irrecoverable class" rested on the bulk-feature regressor (AUROC 0.47-0.63), which never saw the
spectrum. `decidability_cheap.py` (2026-10-29, minutes; `results/decidability_cheap.{txt,csv}`) tests
on the clean POP-I spectra, with the retrieval of every planet as truth (undecidable = p_true ≤ 0.95):

* **Existing scores, AUROC for undecidability:** margin 0.73 / 0.61 / 0.71 / 0.74 and ensemble spread
  0.74 / 0.59 / 0.72 / 0.77 (CH₄ / H₂O / CO₂ / NH₃); Mahalanobis and k-NN 0.60-0.69 (0.68 for H₂O,
  the best there); optical 0.39-0.58; disagree 0.49-0.62; band metric 0.53-0.69; muting 0.56-0.62;
  bulk regressor 0.61-0.66. So the classifier's own confidence does recognise the class, imperfectly.
* **A learned flag** (random forest, 5-fold out of fold, spectrum + bulk features, the retrieval's
  labels as target): 0.73 / 0.74 / 0.73 / 0.69; spectrum only 0.72 / 0.73 / 0.71 / 0.68; bulk only
  0.61-0.68. It beats the margin only for H₂O (0.74 vs 0.61) and ties it elsewhere. Declining the
  10/20/30 % most undecidable by the flag raises CH₄ accuracy 86.4 → 88.4 / 89.3 / 91.0 %; by the
  margin 90.0 / 91.9 / 94.6 % (the margin ranks errors, not only undecidables).
* **Retrieval budget** (CH₄ targets lost at 60 %, f = 10/20/30/50 %): random 16/12/10/4; learned flag
  13/10/7/4; margin 17/14/11/5; k-NN 13/8/5/3; Mahalanobis 13/9/7/3; measured Δχ² (an oracle that
  knows decidability) 13/8/4/2; the retrieval's own p_true 7/7/4/1. For H₂O nothing short of p_true
  beats random until f = 50 %. The ceiling of any decidability recogniser is the Δχ² row: four
  targets of 359 at f = 20 %, because the undecidable planets are the ones the retrieval cannot help
  either.

Reading: a cheap recogniser of the irrecoverable class exists and is not novel (the classifier's
confidence); a novel one cannot beat it by much (the spectrum-trained flag ties it); and the value of
recognising the class is bounded and small for selection. Its value is as a withheld verdict, which is
the paper's recommendation. Abstract, 3.6, Discussion, Conclusion, Table C1 (C2), the response letter
(E3, E4) and the cover letter now say this instead of "nothing cheap recognises".

### F13. Routed by either flag, the classifier recovers methane on four of the six real reductions and 27 of 35 labels, at 4.6 points of clean accuracy in simulation

The paper had said the optical flag "did not fire on the real misses" while noting that input
disagreement flagged four of six. `router_real.py` (2026-10-29; `results/jwst_panel/router_real.{txt,csv}`)
applies the routing of F5 to the 13 panel reductions cell by cell: a (molecule, learner) cell takes the
AIRS-only learner's verdict when the optical-excess score or the learner's input disagreement exceeds
its clean 10 % threshold, the all-input verdict otherwise; the planet's verdict is the majority of four.

| router | scored labels correct | CH₄ on the six methane reductions (present / split / absent) | cells routed |
| none (all inputs) | 22/35 | 0 / 1 / 5 | 0/208 |
| optical | 24/35 | 2 / 1 / 3 | 48/208 |
| disagree | 26/35 | 3 / 1 / 2 | 30/208 |
| either | 27/35 | 4 / 1 / 1 | 72/208 |
| AIRS-only everywhere | 29/37 | 5 / 1 / 0 | 208/208 |

Per reduction (either): K2-18b absent → present (4/4 routed, optical), K2-18b-alt absent → present
(optical), TOI-270d-hm absent → present (4/4, disagreement), TOI-1231b split → present (3/4,
disagreement), K2-18b-hu absent → split (1/4), TOI-270d (muting 0.7, in prior) absent → absent (0/4).
Other changes: GJ3470b CO₂ split → present [present], WASP-166b CO₂ absent → present [present],
HD189733b H₂O absent → present [ambiguous], HD189733b CO₂ absent → split [absent] (the one loss).

Simulated cost (diagnostic.txt remedy block, rerun with "either"): flagged on clean 9 % (disagree),
10 % (optical), 17 % (either); mean accuracy on clean 81.2 → 77.1 (disagree), 79.7 (optical), 76.6
(either); under haze 63.5 → 72.4 / 73.3 / 73.3; spots 70.4 → 73.5 / 73.8 / 73.9; noise ×2 71.9 →
70.3 / 71.5 / 70.2. CH₄ under haze: either 66.6 → 78.3 %, recall 11.6 → 53.7 % (same as optical).
Budget test: "either" ordering = optical's (12.6 at f = 20 %, 3e7). The disagreement flag on clean
spectra routes exactly the planets where the weaker AIRS-only learner differs from the all-input one,
so it costs 4 points there; on the real panel the differing answer is the right one. Reading: the
router's value is set by the prevalence of the departure in the population it meets; the paper now
states both numbers (abstract, 3.6, Discussion, Conclusion, Table 8 last column, letters). Table 11
over ten draws with the combined router (CH₄ targets lost; clean / haze / spots / deck / compound / noise ×2):
classifier 16.0 / 35.8 / 32.6 / 28.6 / 69.5 / 37.4; optical 17.1 / 26.8 / 27.9 / 28.7 / 28.7 / 37.6; either 19.0 / 26.8 / 27.8 / 29.0 / 28.7 / 39.2;
disagree alone 18.8 / 33.3 / 31.4 / 29.2 / 60.8 / 39.0. The combined router costs 2-3 targets on clean and noisy spectra and
matches the optical flag everywhere else.

### F14. Nested-input disagreement locates the departure but a three-level router ties the two-level one

Script `nested_router.py` → `results/nested_router.{txt,csv}`. The classifier's three input sets are nested
(all ⊃ no-optical ⊃ AIRS-only), so the disagreement splits into D₁ = |p_all − p_no-optical| (sensitive to
the three optical points) and D₂ = |p_no-optical − p_AIRS-only| (sensitive to the NIRSpec range), each at the
90th percentile of its clean distribution. Firing rates (mean over molecules and learners), D₁ / D₂:
clean 9.0 / 9.1 %; haze 3e7 34.5 / 16.0; haze 2e6 25.4 / 9.0; spots 21.5 / 12.4; cloud deck 4.7 / 8.1;
compound 34.8 / 23.3; noise ×2 6.0 / 7.4. So D₂ stays at its clean rate at the weaker haze (the slope is
confined to the optical points) and rises at 3e7 and under the compound departure for CO₂ and NH₃
(21 and 21 %; compound 33 and 31 %), whose bands the slope reaches in NIRSpec. The localisation works.
As a router (accuracy, mean over molecules and learners; clean / haze3e7 / haze2e6 / spots / deck / compound / noise×2):
classifier 81.2 / 63.5 / 69.2 / 70.4 / 71.4 / 62.3 / 71.9; combined flag → AIRS 76.6 / 73.3 / 73.2 / 73.9 / 71.8 / 73.5 / 70.2;
nested three-level 76.1 / 72.6 / 74.2 / 73.7 / 71.9 / 72.7 / 69.8; D₁ only → no-optical 78.8 / 71.3 / 76.2 / 73.4 / 71.9 / 69.2 / 70.6.
Reading: the three-level router ties the combined flag everywhere (within a point); the cheaper fallback
(no-optical) halves the clean cost and wins at the weaker haze but loses 4 points under the compound
departure, so which input set to fall back on depends on how far the departure reaches, which is what the
two levels measure. Paper-worthy as a sentence group in 3.6 and the Methods, not as a new method: it is
the principle "route to the largest agreeing input set", and here it buys localisation, not accuracy.

### F15. The panel extended from 13 to 22 reductions: the routing result holds on planets it had not seen

Nine planets added (data via `fetch_jwst_data.sh`; labels fixed from the free-chemistry retrievals before any
verdict was computed, 2026-10-29): GJ 9827 d, WASP-52 b, L 98-59 d, GJ 1214 b, TOI-732 c (LTT 3780 c),
WASP-80 b, HD 209458 b, WASP-96 b, WASP-17 b; WASP-39 b is scored on the router too. Not obtainable: LTT 9779 b,
WASP-69 b (transmission unpublished or MAST-only), WASP-52 b NIRISS (MAST-only; the G395H reduction of Pan et al.
2026 is used instead). Scores (`results/jwst_panel/panel_summary.txt`): all inputs 35/51 (first batch 22/35),
no-optical 40/53, AIRS-only 45/56; on the second batch alone 13/16, 15/18, 15/19, the same ordering.
Router (`router_real.txt`, 23 reductions with WASP-39 b): none 37/53, optical 39/53, disagree 43/54, either 44/54,
AIRS-only everywhere 47/58. Methane, eight reductions of five planets: all-input absent on 7, split on 1;
either-router present on 5, split 1, absent 2 (TOI-270 d Felix; GJ 1214 b, whose label is a 3σ "suggestive" CH₄
in a CO₂-dominated atmosphere). On the second batch the disagreement flag turned TOI-732 c's CH₄ absent → present
and WASP-96 b's H₂O split → present, and changed no second-batch label for the worse; WASP-52 b's NH₃ goes
absent → present under routing where the paper reports no evidence but no limit (unscored). Twin controls for
TOI-732 c: real kNN 11.30 (declined), twin masked 5.91, twin CH₄ found 67 % (77 % with real errors), real 22 %;
muting 0.97 (0.76-1.47), inside the prior; the retrained sets do not move it, consistent with the within-prior rule.
Muting census now ten giants: WASP-17 b 3.99 (3.84-4.18) vs prior max 1.96, WASP-80 b 2.89 (2.48-3.71) vs 1.66,
HD 209458 b 2.35 (2.28-2.44) vs 1.33, all beyond in every resample; WASP-96 b 1.49 (1.39-1.61) vs 1.94 and
WASP-52 b 2.26 (2.17-2.36) vs 2.39 inside. So eight of ten giants beyond the prior, the two inside being the two
least muted; "five of five" is replaced throughout. GJ 9827 d: no dispersion above its errors in 63 % of resamples
(muting undefined); its H₂O is missed by every variant although the twin is found in 99 %, the data's precision.
Retrained design on the extended panel: haze 41/54, haze + deck 43/54 (was 26/36, 27/36). Distance rule: five of
the eight methane reductions declined (GJ 1214 b 11.32, TOI-732 c 11.30 join the three), the giants not.
Reading: the real-panel routing result was a count of six on three planets; it is now a count of eight on five,
with two of the three added methane-bearing spectra recovered by the flags and the one not recovered the weakest
label of the set. The giants' muting generalises with two informative exceptions. Written into the abstract, 2.6
(Table 3), 3.5 (Table 8, Figure 4), 3.6, Discussion, Table C1 and both letters; refs 70-81 added.

### F16. The muting census as a router: each real failure class to its own remedy gives 47 of 56 panel labels (2026-10-29; `muting_router.py`)

Zero-compute test from existing outputs (`results/jwst_panel/{muting_census,muting_bootstrap,router_real}.csv`, the
base and retrained panel verdicts). A planet whose muting exceeds the prior's most muted twin at its bulk parameters
(point estimate; 9 reductions, 8 with P > 0.95) is sent to the design retrained with hazed + deep-deck training planets;
every other planet keeps the published design or the either-flag routing. Results (scored labels; giants = ten warm giants
with abundances):

| router | scored | on muting-flagged | on giants | CH₄ P/split/A |
| :-- | --: | --: | --: | --: |
| none | 37/53 | 16/21 | 17/22 | 0/1/7 |
| either → AIRS-only | 44/54 | 18/22 | 20/24 | 5/1/2 |
| muting → retrained hazedeck | 42/56 | 21/24 | 22/25 | 0/1/7 |
| retrained hazedeck everywhere | 45/56 | 21/24 | 23/26 | 2/2/4 |
| **muting → retrained, else either → AIRS-only** | **47/56** | 21/24 | 23/26 | 5/1/2 |
| retrained everywhere, either → its AIRS-only (no census) | 46/58 | 19/23 | 21/25 | 6/0/2 |

The muting route changes five labels, all on giants and all to the right answer (HAT-P-26 b CO₂, GJ 3470 b CO₂,
WASP-166 b H₂O and CO₂, WASP-80 b H₂O: split/absent → present) and nothing else; "either, then muting" (either flag
first at the cell level) loses the gain because the disagreement flag fires on at least one learner on those cells, so the
order is census first. The no-census version finds methane on six of eight (K2-18 b-hu becomes present) but pays two
labels (HD 189733 b CO₂ absent → present, GJ 1214 b CO₂ split → absent).

Reading: the real misses are two classes with two remedies. The methane sub-Neptunes are a shape failure (the optical
points), repaired by another input; the giants are an amplitude failure (muting beyond the prior), repaired by extending
the prior. A two-axis diagnostic, amplitude against the prior and shape across the nested inputs, each routed to its own
remedy, is the pipeline the panel supports. Caveat stated in the paper: the census twin needs a composition, so the
amplitude flag is computable only after a retrieval (or from the literature, as here); the version without it is the
retrained design everywhere plus the either flag. Added to Sections 3.5, 3.6, the Discussion, the abstract and the
Conclusion.

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
irrecoverable, the errors haze and spots add are 57–87 % recoverable, a deep deck's 51–82 % and
threefold noise's 10–56 %, the retrieval of every
planet is the Tier-2 selection oracle on clean spectra, and on clean spectra no diagnostic routes a
partial retrieval budget better than chance, the confidence rules worse. On a mixed population with a
fifth of the planets hazed, the optical score routed to a retrieval with the haze density free turns a
20 % budget into the oracle's selection (CH₄ hazed-target recall 3 → 80 %, targets lost 29 → 13 against
the oracle's 12), the distance rules come within 16–26 points, the confidence rules find none of the hazed
planets, and the same budget sent to the existing clear-model retrieval is worse than no retrieval; the
free-haze retrieval on the hazed population recovers the density on every planet and 222 of the 224 CH₄
labels decidable on the clean spectrum, so E3 and E4 are both met. The muting census (F7) turns the one-planet WASP-39 b
result into a population statement: five of five warm giants with retrieved abundances are muted beyond
anything the simulator population can draw at their bulk parameters, which no normalised-input rule can
see. The remaining pieces are the free-haze retrieval of the clean population (running, due 31 October; it
replaces the clean retrieval's posteriors on the clean side of the E3 table) and the manuscript trim.

(Note, 2026-10-29: the manuscript tables were renumbered in the light editorial pass; old Table 1 is now Table A1 and old Tables 2-11 are Tables 1-10. Table numbers in the findings above are the pre-pass numbers.)
