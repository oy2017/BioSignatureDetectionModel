# Can a simulator-trained classifier for Ariel Tier-1 spectra be trusted?

Owen Yang<sup>a</sup>

<sup>a</sup> Henry M. Gunn High School, 780 Arastradero Rd, Palo Alto, CA 94306, USA. Tel.: +1 650 686 7813. E-mail: owenhyang@gmail.com (corresponding author)

---

## Abstract

Machine-learning classifiers have been proposed for sorting the roughly 1,000 planets of Ariel's Tier-1 reconnaissance survey by composition, but no Ariel data exist, so such classifiers have been trained and validated only on spectra from the simulator that produced them. This technical note asked whether one such classifier could be trusted when real atmospheres differ from the simulator. The molecular classifiers proposed in the Ariel consortium's Tier-1 population study, whose code was not released, were re-implemented from the published description; the re-implementation reproduced all 48 published accuracies within 3.7 points. The classifier was then tested against fifteen physically motivated departures from its simulator. For each departure the test measured the accuracy lost, the part of that loss a retrained classifier could recover, whether training on a randomized grid absorbed it, and whether confidence scores, distance-based novelty scores or conformal prediction flagged the errors. The classifier was robust to an alternative opacity database, an alternative radiative-transfer code and two omitted absorbers (losses of 0.1-3.5 points). It was not robust to haze: at a haze strength comparable to the weakest retrieved for observed hot Jupiters, it reported methane and water absent on nearly every planet that contained them, with rising confidence and no warning from its own ensemble, although 87 % of the loss was recoverable. Removing the three optical photometric points from its input removed most of this failure, which identified those points as the cause. A second classifier, built for a different Ariel science question, showed a nearly opposite pattern at full resolution, so the trust profile of one classifier did not carry over to another. Each classifier therefore needs its own stress test before its outputs are used to select planets.

**Keywords:** exoplanet atmospheres; transmission spectroscopy; Ariel; Tier-1 survey; machine learning; classification; reproducibility; re-implementation; robustness; domain shift; hazes; stellar contamination; out-of-distribution detection; conformal prediction

## 1. Introduction

When a planet passes in front of its star, a small fraction of the starlight passes through the planet's atmosphere on its way to the observer. Molecules in the atmosphere absorb that light at their own characteristic wavelengths, so the apparent size of the planet changes with wavelength. The record of that change, the transmission spectrum, shows which molecules are present. Ariel, a European Space Agency mission scheduled for launch in 2029, will measure transmission spectra of about 1,000 transiting exoplanets between 0.5 and 7.8 µm, using three broadband photometers below 1.1 µm and three spectrometers above (1). Telescope time will not allow every planet to be studied deeply. The survey is therefore organised in tiers: every target is observed first at low signal-to-noise in a Tier-1 reconnaissance pass, and a substantial subset is then re-observed at higher precision in Tier 2 (1-4). Tier-1 data are expected to reveal the presence of molecular features and the degree of cloudiness (4), so they are the natural basis for deciding which planets receive the deeper observations. A planet judged uninteresting at Tier 1 may never be observed again.

Making that decision for about 1,000 planets is itself a problem. The standard way to read a spectrum, an atmospheric retrieval, fits a physical model with sampling algorithms that need between 10⁵ and 10⁸ model evaluations per planet (5), and its assumptions and priors must be chosen and checked by a specialist (6). Machine-learning classifiers are an attractive alternative: once trained, they return an answer for every planet in seconds. The Ariel consortium's population study of the Tier-1 survey proposed two retrieval-free ways of classifying planets: a band-based metric and, as a preliminary assessment, four scikit-learn classifiers trained to flag the presence of CH₄, H₂O, CO₂ and NH₃ (6). The classifiers reached 64-89 % accuracy on a held-out simulated population. Their code and trained models were not released.

Because Ariel has not yet flown, these classifiers, like the classifiers and retrieval networks developed for the Ariel Data Challenges (5) and the wider machine-learning retrieval literature (7, 8), were trained and tested on spectra from the same simulator. That test shows that a classifier has learned its simulator. It cannot show how the classifier behaves when real atmospheres differ from the simulator, and there are well-documented ways in which they will. Hazes and clouds are common in observed hot Jupiters and mute or reshape their spectra across a continuum from clear to strongly obscured (9, 10). Unocculted spots on the host star imprint false spectral features, most strongly for cool stars (11). Opacity data compiled from different line lists give different absorption cross sections (12, 13). Training grids include only a handful of absorbers: the database of the first Ariel Data Challenge contains five (5), while carbon-rich atmospheres, for example, are expected to carry HCN and C₂H₂ in abundance (14). The authors of the consortium study acknowledged the limitation, describing their spectra as "transmission spectral shapes" to test methods against (6). The concern is practical. A classifier that fails in these situations does not announce it; it returns confident answers, and the planets it wrongly dismisses do not receive the observations that would have revealed them.

Deliberately altering the test data and measuring the consequences is established practice in machine learning (15, 16) and in simulation-based inference (17, 18). In exoplanet atmospheres, the nearest study tested a retrieval network on spectra with an added absorber, a removed absorber and unocculted star spots (19), and a recent flow-matching retrieval left model mismatch such as unmodelled clouds to future work (8). No test of this kind had been reported for a classifier proposed for Ariel's Tier-1 survey, and because the consortium's code was not released, any such test first requires an independent re-implementation that demonstrably matches the published results.

This technical note asked whether the consortium's Tier-1 molecular classifiers could be trusted when real atmospheres differ from their simulator, and whether the answer carries over to other classifiers. It re-implemented the classifiers from their published description and verified the re-implementation against all 48 published accuracies (Section 3.1). It then subjected them to a stress test of fifteen physically motivated departures from the simulator, measuring for each how much accuracy was lost, how much of the loss any retraining could recover, whether training on a randomized grid absorbed it, and whether the classifiers' confidence, distance-based novelty scores or conformal prediction flagged the errors (Sections 3.2-3.4). To find out whether the resulting trust profile belonged to this classifier or to Ariel classifiers in general, the same test was applied to a second classifier, built for a different Ariel science question, the carbon-to-oxygen ratio; a single classifier with a different profile is enough to show that one classifier's stress test cannot stand in for another's (Section 3.5).

The test showed that the published design was robust to a different opacity database, a different radiative-transfer code and omitted absorbers, but was confidently blind to haze of an observed strength: it reported methane and water absent while their infrared bands were intact, because it relied on the three optical photometric points. The second classifier had a nearly opposite profile. These results are useful in three ways. Anyone applying or extending the consortium design gains an openly released re-implementation that reproduces the published results, together with a specific weakness and two remedies. Anyone proposing a classifier for Ariel gains a stress-test procedure, applicable to any classifier trained on simulated spectra, with released scripts for the two classifiers studied here. And selecting Ariel's deeper-tier targets with any simulator-trained classifier gains a concrete requirement: the classifier must be tested on its own, against the ways its simulator can be wrong, before its outputs are used. All code, result tables and the expectations recorded before the runs are in a public repository directory (see Data and Code Availability).

## 2. Materials and Methods

### 2.1 The published classifier and its re-implementation

The re-implementation followed Sections 2.2 and 2.5 of the consortium study (6) element by element (Table 1). Isothermal atmospheres with random temperatures, abundances of CH₄, H₂O, CO₂ and NH₃, and grey cloud decks were generated for every planet of the Ariel candidate list, four times for training (POP-III) and once for testing (POP-I). Spectra were binned at Tier-3 resolution, scattered with Tier-1 noise and normalised per spectrum. A molecule was labelled present if its abundance exceeded 10⁻⁵, 10⁻⁴ or 10⁻³, and four scikit-learn classifiers at default settings (20) were trained for each molecule and threshold; these defaults have not changed since version 0.22, the version the study cites.

**Table 1.** Elements of the published classifier design, the statement in the consortium study (6) (page of the arXiv version), and the re-implementation.

| Element | Published statement | Re-implementation |
| :-- | :-- | :-- |
| Planet list | Ariel candidate list, 1,000 planets including predicted TESS discoveries (p. 4) | 965 known planets of the 2026 Mission Candidate Sample |
| Training and test populations | POP-III: each planet repeated 4 times; POP-I: each planet once (pp. 5-6) | Same |
| Temperature | Between 0.7 and 1.05 of the equilibrium temperature, isothermal (p. 4) | Same |
| Atmosphere grid | 100 layers, 10⁻⁴ to 10⁶ Pa; H₂ and He with He/H₂ = 0.17 (p. 4) | Same |
| Abundances | CH₄, H₂O, CO₂, NH₃ log-uniform, 10⁻⁷-10⁻² (POP-I) and 10⁻⁹-10⁻² (POP-III) (pp. 4, 6) | Same |
| Clouds | Grey opaque deck, 5 × 10² to 10⁶ Pa (p. 4) | Same |
| Opacities | ExoMol line lists; H₂-H₂ and H₂-He collision-induced absorption (p. 5, Table 2) | Exo-Transmit tables (12); no collision-induced absorption (Section 2.2) |
| Binning | Tier 3, R = 20, 100, 30 (pp. 2, 5) | Same, with the three photometric points |
| Noise | "The noise estimated with ArielRad at each spectral bin … a re-scaled version of the Tier 3 noise, obtained by combining the number of transit observations needed to match the Tier 1 required SNR" (p. 5) | Payload noise model at each target's Tier-1 transit count (Section 2.3) |
| Normalisation | "Each example spectrum is normalised to zero mean and unit dispersion" (p. 11) | Same |
| Classifiers | scikit-learn defaults: KNN with k = 5, MLP with 100 hidden units, RFC, SVC one-vs-one (p. 12) | Same (scikit-learn 1.7) |
| Evaluation | Trained on POP-III, tested on POP-I, accuracy at three thresholds (p. 18) | Same |

### 2.2 Interpretation and unavoidable deviations

The study did not state whether the classifier input included the three optical photometric points. It stated that the classifiers learn "from their spectral shape over the whole wavelength range sampled by Ariel" and "gather information from all the spectral data points" (6), and its figures of observed Tier-1 spectra show the photometric points, so they were included.

Three deviations could not be avoided. (i) Opacities came from Exo-Transmit's tables (12, 21); substituting ExoMol cross sections (13) for H₂O, CH₄, CO₂ and CO cost the classifier 2.9 points (Section 3.2), which bounds this choice. (ii) Collision-induced absorption was omitted; adding the H₂-H₂ and H₂-He contributions changed test spectra by at most 3 ppm, against feature amplitudes of 270-1,400 ppm. (iii) Noise came from a payload model instead of ArielRad, which is not public (Section 2.3). The planet list also differed (Table 1).

An earlier reading of the publication, with spectra binned to the seven Tier-1 points, the atmosphere truncated at 1 Pa and the noise set exactly at the Tier-1 requirement, missed the published accuracies by 10 points on average and by up to 38. The comparison with the published values therefore discriminates between readings of the recipe.

### 2.3 Forward model and noise

Spectra were computed with TauREx 3 (22) through MultiREx (23) at R ≈ 1,000 and binned by integration. TauREx 3.3.2 reads the pressure grid of Exo-Transmit opacity tables as bar, whereas Exo-Transmit treats it as Pa; the grid was divided by 10⁵ at load, and the inconsistency was reported to the TauREx developers (24). With the correction, TauREx spectra of 1,804 test atmospheres agreed better with Exo-Transmit's own code (median shape correlation 0.9986, against 0.986).

Per-bin noise came from a payload model built with ExoSim 2 (25): the one-hour noise-to-signal ratio for the host's effective temperature, scaled to the host's distance and radius and to the transit duration, and divided by the square root of the number of transits. Its single free level was calibrated so that a signal-to-noise ratio of 7 on the five-scale-height atmospheric signal reproduced the candidate sample's Tier-3 transit counts, which the model then predicted within 0.07 dex (17 %). For the three example planets in Figure 1 of the consortium study (HD 209458 b, GJ 1214 b and WASP-79 b), its AIRS noise of about 75, 430 and 230 ppm per bin matched the published noise bands in size.

### 2.4 A second classifier

The second classifier addressed the carbon-to-oxygen ratio, a stated Ariel objective (1): an atmosphere was labelled carbon-rich if C/O > 1, the chemical boundary above which oxygen is locked in CO and water is depleted (14, 26). Its grid held 18,040 training and 9,016 test atmospheres in five test sets, with independently drawn planet radius (1-26 R⊕), mass (1-300 M⊕), host temperature (2,500-7,500 K), atmospheric temperature (500-2,500 K), C/O (uniform, 0.2-1.8) and metallicity (uniform, −1 to 1.5 dex). Abundances of H₂O, CH₄, CO, CO₂, NH₃ and O₃ came from chemical equilibrium computed with FastChem (27) at 10⁻² bar. Spectra were binned to the Tier-3 layout of the mission's radiometric model (R = 15, 100 and 30; 102 points) (28) and to the seven-point Tier-1 layout, with noise at a signal-to-noise ratio of 15 on each spectrum's amplitude, or from the payload model of Section 2.3 for the known Ariel targets. Gradient-boosted trees (29) on per-spectrum-normalised bins, tuned by cross-validation, reached 96.1 % on the test sets (95.9 ± 0.1 % across ten training resamples).

Because the two classifiers differ in label, grid, model and noise, their comparison can show whether their trust profiles agree, not why they differ; causes were tested separately (Sections 3.3 and 3.5).

### 2.5 The stress test

Each departure (Table 2) re-rendered or modified the same test planets, so every loss was a within-planet difference under the same noise draw. Haze followed the Lee et al. prescription (30) with 0.1 µm particles. Retrievals of ten observed hot Jupiters expressed haze opacity at 0.35 µm as a multiple a of the H₂ Rayleigh opacity and found log₁₀ a = 2.1-5.7 (10); at 1 mbar, the densities 3 × 10⁷, 2.4 × 10⁸ and 10¹⁰ m⁻³ used here correspond to log₁₀ a ≈ 2.4-2.7, 3.3-3.6 and 4.9-5.2, placing 3 × 10⁷ m⁻³ at the weak end of the observed range. Spots used PHOENIX spectra (31) at a spot-to-photosphere temperature ratio of 0.85 (11).

**Table 2.** Departures from the simulator used in the stress test.

| Departure | Levels | What it represents |
| :-- | :-- | :-- |
| Grey cloud deck at a fixed pressure | 10⁴, 10³ (inside the published cloud prior), 10², 10 Pa | High, opaque clouds |
| Haze (Lee et al. Mie, 0.1 µm) | 2 × 10⁵ to 10¹⁰ m⁻³; also 0.03 and 0.5 µm particles | Photochemical hazes, absent from the training grid |
| Unocculted star spots, faculae | 2-20 % spot coverage; 10 % faculae | Stellar contamination |
| Spots and haze together | 20 % and 3 × 10⁷ m⁻³ | Compound departure |
| Extra white or correlated noise | 1.5, 2 and 3 × the per-bin σ | Underestimated or correlated noise |
| Gain ramp | 1 and 2 noise levels across the spectrum | Instrumental systematic |
| ExoMol opacity database | H₂O, CH₄, CO₂, CO | Opacity data |
| Exo-Transmit's own code | Same atmospheres, deck and He fill | Radiative-transfer code |
| HCN and C₂H₂ added | 10⁻⁷-10⁻⁴ (consortium); equilibrium abundances (second classifier) | Absorbers omitted from the training grid |

Unless stated, results are for the 10⁻⁴ threshold, averaged over the four classifiers and four molecules. Five quantities were measured. (i) Loss: clean accuracy minus accuracy under the departure. (ii) Irreducible loss: clean accuracy minus the accuracy of the design retrained with the departure present, so that recovery is measured against what retraining can achieve. (iii) Absorption: the loss of the design retrained on a randomized grid in which each planet independently carried haze (60 % of planets; log-uniform, 10⁵-3 × 10⁸ m⁻³), spots (70 %; 0-20 %) and noise 1-3 times nominal (32); three variants each omitted one ingredient, testing transfer to an unseen departure (15). (iv) Detection: five decline rules (probability margin; spread of the four classifiers' probabilities (33); Mahalanobis distance (34); k-nearest-neighbour distance (35); principal-component reconstruction error), each with a threshold declining 10 % of clean planets and credited only with accepted-planet accuracy above the clean selective baseline at equal coverage (16, 36). (v) Calibration: expected calibration error and the coverage of split-conformal sets calibrated to 90 % on half the clean planets (37, 38), for the three classifiers that output probabilities. Expectations for the consortium classifier were committed to the repository before the runs (Section 3.6).

## 3. Results and Discussion

### 3.1 The re-implementation reproduced the published accuracies

![Figure 1](figures/note_fig1_reproduction.png)

**Figure 1.** The re-implementation against the published accuracies for the 48 combinations of classifier, molecule and abundance threshold. (a) Re-implemented against published accuracy; dashed line, equality. (b) Difference between the two for each combination, grouped by molecule, with marker shape giving the abundance threshold. Grey bands: ±5 points.

Across the 48 published accuracies, the re-implementation differed by +0.4 points on average, 1.4 in absolute value and at most 3.7 (Figure 1; Appendix A), with a correlation of 0.96. It also reproduced the published patterns: accuracy rose with the abundance threshold for 15 of 16 classifier-molecule pairs (16 of 16 published), and water was the hardest molecule at 10⁻⁵ and 10⁻⁴ in both. At 10⁻⁴ the mean accuracies were 84.9 % (CH₄), 74.9 % (H₂O), 79.5 % (CO₂) and 84.4 % (NH₃), against 84.0, 74.8, 80.5 and 85.0 % published. Since a mistaken reading of the recipe missed by 10 points (Section 2.2), this agreement indicates that the re-implementation followed the published design.

### 3.2 Where the classifier could and could not be trusted

![Figure 2](figures/note_fig2_consortium_map.png){width=5.6}

**Figure 2.** The consortium classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Open circles: loss of the classifier as published; filled circles: loss after randomized training; black ticks: irreducible loss. Error bars: standard deviation across the four molecules.

**Table 3.** The consortium classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Losses are in points below clean accuracy. The margin rule declines 10 % of clean planets; margin AUROC (error) measures how well the margin ranks errors, and distance AUROC (shift) how well the Mahalanobis distance separates altered from clean spectra (0.5 is chance). Conformal sets were calibrated to 90 % on clean spectra.

<!-- table:consortium -->
| Mismatch | Accuracy (%) | Loss | Irreducible | Loss after randomized training | Margin rule keeps (%) | Margin AUROC (error) | Distance AUROC (shift) | Conformal coverage (%) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| None (clean) | 81.1 | — | — | 1.8 | 93 | 0.76 | — | 93 |
| Grey cloud deck, 10³ Pa | 78.2 | 2.9 | — | 2.7 | 93 | 0.74 | 0.56 | 91 |
| Grey cloud deck, 10² Pa | 71.3 | 9.8 | 2.6 | 6.1 | 94 | 0.72 | 0.66 | 85 |
| Haze, 2 × 10⁶ m⁻³ | 69.0 | 12.1 | — | 3.9 | 95 | 0.73 | 0.77 | 82 |
| Haze, 3 × 10⁷ m⁻³ | 63.4 | 17.7 | 2.3 | 6.8 | 99 | 0.70 | 0.93 | 74 |
| Haze, 2.4 × 10⁸ m⁻³ | 62.3 | 18.7 | — | 10.5 | 99 | 0.65 | 0.88 | 76 |
| Unocculted spots, 10 % | 74.3 | 6.8 | 2.3 | 3.5 | 95 | 0.69 | 0.62 | 87 |
| Unocculted spots, 20 % | 70.3 | 10.8 | 3.7 | 5.0 | 96 | 0.67 | 0.71 | 82 |
| Spots 20 % + haze 3 × 10⁷ m⁻³ | 62.2 | 18.8 | 3.7 | 7.9 | 100 | 0.64 | 0.97 | 71 |
| White noise, 2 × σ | 71.8 | 9.3 | 7.3 | 8.3 | 93 | 0.70 | 0.74 | 87 |
| White noise, 3 × σ | 67.5 | 13.5 | 11.8 | 12.6 | 95 | 0.67 | 0.82 | 83 |
| Correlated noise, 2 × σ | 68.7 | 12.4 | 10.2 | 13.2 | 93 | 0.63 | 0.49 | 82 |
| Gain ramp, 2 noise levels | 71.2 | 9.8 | 1.6 | 11.7 | 92 | 0.71 | 0.45 | 86 |
| ExoMol opacity database | 78.2 | 2.9 | 1.0 | 4.1 | 91 | 0.74 | 0.58 | 93 |
| Exo-Transmit radiative transfer | 77.6 | 3.5 | 2.8 | 4.4 | 91 | 0.73 | 0.64 | 91 |
| HCN and C₂H₂ added | 81.0 | 0.1 | -0.4 | 2.0 | 93 | 0.76 | 0.50 | 93 |
<!-- /table:consortium -->

The classifier lost little to a different opacity database (2.9 points), a different radiative-transfer code (3.5), added HCN and C₂H₂ (0.1) or a cloud deck at 10³ Pa, inside its training prior (2.9) (Table 3, Figure 2). It lost far more to haze (12.1 points at 2 × 10⁶ m⁻³, 17.7 at 3 × 10⁷ m⁻³), to spots (10.8 at 20 % coverage), to a cloud deck at 10² Pa (9.8), to spots and haze together (18.8) and to extra noise (9.3-13.5).

Most aerosol and spot losses were recoverable. Retrained with haze, the design reached 78.8 %, so 87 % of the haze loss was reducible and the molecular information survived in the hazy spectra; 65-80 % was reducible for spots, the high deck and the compound. Noise losses were mostly irreducible (13-21 % reducible). The randomized grid cost 1.8 points on clean spectra and recovered 71 % of the reducible haze loss and 75-82 % of the spot loss; with haze omitted from the grid, randomizing spots and noise still recovered 62 % of the improvement on hazy spectra. It did not help with the opacity database, the code or the gain ramp, none of which it contained.

### 3.3 Why haze blinded the classifier

![Figure 3](figures/note_fig3_haze.png)

**Figure 3.** (a) A clear-atmosphere test planet with CH₄ above 10⁻⁴ at Tier-3 binning, without and with haze at 3 × 10⁷ m⁻³ (noise-free); shading marks the three photometric points. (b) Detection rate, the share of planets containing a molecule for which it was reported, against haze density for the published input, the input without the photometric points and the AIRS bins only (mean over four classifiers and four molecules).

Under haze at 3 × 10⁷ m⁻³, the multilayer-perceptron, random-forest and support-vector classifiers reported CH₄ on 0-5 % of planets (27-33 % on clean spectra; 38 % truly present) and detected it on 0-13 % of the planets containing it; their H₂O detection rate fell to 0 %. The multilayer perceptron's mean confidence rose from 0.93 to 1.00. Yet the molecular bands were intact: the haze added a median 1,094 ppm to the three photometric points, 140 ppm across NIRSpec and 5 ppm across AIRS, leaving the AIRS band amplitudes unchanged (ratio 1.00; Figure 3a).

Four versions of the design, differing only in input, identified the cause (Table 4). Normalising with statistics from the bins above 1.1 µm, while keeping every bin, left the failure unchanged, so normalisation was not the cause. Removing the photometric points cut the loss to haze at 3 × 10⁷ m⁻³ from 17.7 to 5.6 points and to 20 % spots from 10.8 to 3.2; using only the AIRS bins cut them to 0.5 and to none. Transplanting only the three hazy photometric points into clean spectra reproduced the failure (CH₄ detection 1.6 %), and restoring them in hazy spectra removed it (72 %). The sensitivity to haze and spots therefore came through the photometric inputs.

**Table 4.** The consortium design with four inputs, each retrained on the same spectra (threshold 10⁻⁴; mean over four classifiers and four molecules). Cells: accuracy (%), with the detection rate (%) in parentheses. Haze densities in m⁻³.

<!-- table:variants -->
| Classifier input | Clean | Haze 2 × 10⁶ | Haze 3 × 10⁷ | Haze 2.4 × 10⁸ | Spots 20 % | Spots + haze |
| :-- | --: | --: | --: | --: | --: | --: |
| All 104 bins (published) | 81.1 (63) | 69.0 (23) | 63.4 (6) | 62.3 (3) | 70.3 (26) | 62.2 (3) |
| All bins, normalised with the bins above 1.1 µm | 80.9 (61) | 68.6 (21) | 63.3 (6) | 62.3 (3) | 69.9 (25) | 62.5 (3) |
| Photometric points removed | 77.3 (53) | 76.8 (51) | 71.7 (33) | 65.1 (14) | 74.1 (44) | 69.5 (27) |
| AIRS bins only (above 1.95 µm) | 73.7 (44) | 73.6 (44) | 73.2 (44) | 70.8 (41) | 73.9 (45) | 73.4 (45) |
<!-- /table:variants -->

The mechanism follows from the design: the classifiers use every point including the photometric ones, their training atmospheres have grey clouds but no haze, and a small-particle haze raises the optical points while barely changing the infrared bands. Any implementation of the design is therefore exposed; only the size of the effect depends on implementation choices. Tripling the photometric noise reduced the haze loss to 15.6 points and a tenfold increase to 10.2; 0.03 µm particles at the same optical extinction gave 17.5 points. Particles of 0.5 µm also altered the infrared (AIRS amplitude ratio 1.94), and every input then failed (13.4-19.9 points), a different failure.

The photometric points do carry information: removing them cost 3.8 points on clean spectra, and using AIRS alone 7.4. With haze at 3 × 10⁷ m⁻³, the input without them became the more accurate once about a third of the targets carried such a haze (about 40 % for AIRS alone). Two remedies follow: training with chromatic hazes (Section 3.2), or excluding or down-weighting the optical points.

### 3.4 Whether the classifier could tell

No decline rule restored clean-level accuracy under a serious departure, and every rule's credit against the clean selective baseline was negative under haze (−17.5 to −25.8 points), spots and the high cloud deck (Table 3). Confidence gave no warning of haze: the margin rule declined 7.4 % of clean planets but 0.9 % of hazy ones, and its error ranking fell from 0.76 to 0.70. Distance scores separated hazy from clean spectra (AUROC 0.90-0.94) and declined 71-82 % of hazy planets, but did not rank errors within them (0.44-0.51), acting as a population alarm rather than a filter. Conformal coverage fell to 73.7 % under haze and 82.3 % under spots, and expected calibration error rose from 0.088 to 0.231 under haze. Randomizing haze and spots into training lowered the Mahalanobis separation of hazy spectra from 0.94 to 0.38 and of spotted spectra from 0.72 to 0.45: the more robust design could no longer flag the departures it had absorbed.

### 3.5 The trust profile did not carry over to a second classifier

![Figure 4](figures/note_fig4_carbonrich.png)

**Figure 4.** (a) Accuracy on carbon-rich planets without and with HCN and C₂H₂ at equilibrium abundances, on the simulated test sets (Tier 3) and on the 965 known Ariel targets with payload noise (Tiers 3, 2 and 1); dashed line, 50 %. (b) Accuracy lost to haze at 3 × 10⁷ m⁻³ with all inputs and without the three photometric points, for the consortium classifier and the carbon-rich classifier at Tier-3 and Tier-1 binning; numbers give the clean accuracy lost by removing the points.

The carbon-rich classifier responded almost oppositely to the two departures that divided trust from distrust for the consortium classifier (Figure 4). HCN and C₂H₂, abundant in carbon-rich atmospheres above about 800 K (14, 26), are absent from its training grid, as from the Ariel Data Challenge grid (5). Added at equilibrium abundances, they lowered accuracy on carbon-rich planets from 97.6 to 48.9 % on the simulated test sets, leaving oxygen-rich planets unaffected (94.5 and 95.7 %). On the known Ariel targets with payload noise, carbon-rich accuracy fell from 93.0 to 65.0 % at Tier-3 binning, from 91.4 to 61.9 % at Tier 2 and from 84.0 to 58.4 % at Tier 1. The classifier stayed confident: expected calibration error rose from 0.009 to 0.240 and the mean probability assigned to true carbon-rich planets fell from 0.96 to 0.47. Retraining with the absorbers restored 96.1 %, while a randomized grid of haze, clouds, spots, noise and disequilibrium chemistry did not help (75.6 %). Its errors, unlike the consortium classifier's errors under haze, were ranked well by distance scores (AUROC 0.90-0.91), though not by confidence (0.70). The contrast has a chemical reading: the added absorbers carry no information on whether CH₄ or H₂O exceeds 10⁻⁴, but are tied to C/O.

Haze at 3 × 10⁷ m⁻³ cost the carbon-rich classifier 5.1 points at Tier-3 binning, and removing its photometric points reduced this only to 2.8 (Figure 4b). At Tier-1 binning, where three of the seven points are photometric, haze cost 29.1 points; removing the points cut the loss to 7.4 but cost 18.6 points of clean accuracy, so it was no remedy.

At full resolution, one classifier was robust to omitted absorbers and fragile to haze, and the other the reverse; at Tier 1, both were fragile to haze through the photometric points. The stress test of one Ariel classifier did not predict the other's.

### 3.6 Expected and surprising results

**Table 5.** Expectations for the consortium classifier committed before the stress test, and outcomes.

| Expectation | Outcome |
| :-- | :-- |
| Haze and high-cloud losses mostly irreducible (< 50 % recoverable) | Not met: 87 % (haze) and 73 % (deck) recoverable |
| Spot losses mostly recoverable (> 70 %) | Not met: 65-66 % |
| Randomized grid costs ≤ 2 points on clean spectra | Met: 1.8 points |
| Randomized grid recovers ≥ 50 % of the spot loss and < 50 % of the haze loss | Spots met (75-82 %); haze not met (71 %) |
| Ingredients left out of the randomized grid transfer little (< 30 %) | Not met: 62 % (haze), 85 % (spots) |
| ExoMol opacities cost ≥ 5 points; Exo-Transmit's code ≤ 2 points | Neither met: 2.9 and 3.5 points |
| Spots and haze together cost less than the sum of their losses | Met: 9.6 points less |
| Correlated noise costs ≥ 2 points more than white noise at the same variance | Met: 3.1 points more at twice the noise |
| Distance scores separate hazy from clean spectra (AUROC ≥ 0.9) but do not rank errors; no rule has positive credit | Met |
| Expected calibration error rises ≥ 3 times under haze; conformal coverage < 85 % | Coverage met (73.7 %); calibration error rose 2.6 times |
| Spot loss ≥ 1.5 times larger for M-dwarf hosts | Not met: 7.6 points for M-dwarf hosts, 12.3 for F-type hosts |
| Randomizing haze and spots lowers their Mahalanobis separation by ≥ 0.2 | Met |

Table 5 compares the expectations with the outcomes. The largest surprise was not anticipated: the haze failure came from the photometric inputs while the molecular bands were intact, with confidence rising as accuracy fell. Four expectations about recoverability erred in the same direction, since aerosol losses proved largely recoverable. The reversed host dependence of the spot loss, contrary to the stronger chromatic contamination of M dwarfs (11), remains unexplained and is not used in any conclusion.

### 3.7 Limitations

All spectra, including those of the known Ariel targets, were simulated, so the test probed departures within one simulator family rather than real observations. The consortium classifiers were a preliminary proposal (6), and the results concern the published design as re-implemented, with the deviations of Section 2.2, not the original trained models, which were unavailable. Haze strengths were anchored to retrieved hazes through an order-of-magnitude conversion that depends on the assumed pressure level. The carbon-rich classifier, not a published proposal, served as a counterexample; its independently drawn parameters and an atmosphere truncated at 1-10 Pa are documented in the repository. No machine-learning retrieval trained on the Ariel Data Challenge database was tested.

## 4. Conclusion

The molecular classifiers proposed for Ariel's Tier-1 survey were re-implemented from their published description and reproduced the published accuracies within 3.7 points. A stress test against fifteen departures from the simulator showed that the classifiers could be trusted under changes of opacity database, radiative-transfer code and omitted absorbers, and could not be trusted under haze, star spots and high clouds. Under haze at the weak end of the range retrieved for observed hot Jupiters, the classifiers reported methane and water absent while the infrared bands were intact, became more confident, and gave no warning; the failure came from the three optical photometric inputs, and removing those inputs or training with hazes largely removed it. A second classifier, built for Ariel's carbon-to-oxygen objective, had a nearly opposite trust profile at full resolution, so a stress test of one classifier does not stand in for another. Before any simulator-trained classifier is used to select planets for Ariel's deeper tiers, it should be subjected to its own stress test. The procedure applies to any classifier trained on simulated spectra whose forward model can be re-run; the scripts released with this note implement it for the two classifiers studied and can be adapted to others.

## Acknowledgements

Computations used open-source software (TauREx 3, MultiREx, Exo-Transmit, FastChem, ExoSim 2, scikit-learn, XGBoost) and public data (the ExoMolOP tables, the PHOENIX stellar atlas and the Ariel Mission Candidate Sample). No financial support was received.

The author thanks Rachael Kaci, instructor of the Advanced Authentic Research (AAR) program at Henry M. Gunn High School, and research mentor Victoria Lloyd, for their guidance and feedback throughout this project.

In the interest of transparency, the author discloses the following assistance: Anthropic's Claude Code was used to help implement the Python analysis code and to manage the project's GitHub repository, and Google Gemini was used for polishing and proofreading the manuscript. These tools were used under the author's direction; the author designed the study, directed the analysis, interpreted the results, and is responsible for all content, having reviewed and validated all AI-assisted code and text.

## Data and Code Availability

The re-implemented classifier, the stress-test scripts, every result table and figure, the expectations committed before the runs, and the corrections made to the forward model are in the directory `ariel_tier1_trust` of https://github.com/oy2017/BioSignatureDetectionModel, which serves as the supplementary file of this note.

## Appendix A. Published and re-implemented accuracies

**Table A1.** Accuracy (%) of the re-implemented classifiers and the published values (6) for each classifier, molecule and abundance threshold (POP-III training, POP-I test).

<!-- table:repro -->
| Classifier | Molecule | > 10⁻⁵: this work | > 10⁻⁵: published | > 10⁻⁴: this work | > 10⁻⁴: published | > 10⁻³: this work | > 10⁻³: published |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| KNN | CH₄ | 80.0 | 79 | 84.6 | 83 | 85.0 | 85 |
| KNN | H₂O | 60.9 | 64 | 68.8 | 71 | 83.9 | 82 |
| KNN | CO₂ | 77.8 | 77 | 78.7 | 79 | 82.2 | 82 |
| KNN | NH₃ | 73.2 | 75 | 80.7 | 82 | 86.2 | 84 |
| MLP | CH₄ | 80.8 | 78 | 86.4 | 85 | 87.4 | 87 |
| MLP | H₂O | 70.8 | 70 | 77.6 | 76 | 87.3 | 84 |
| MLP | CO₂ | 80.7 | 77 | 80.8 | 81 | 82.5 | 83 |
| MLP | NH₃ | 80.4 | 80 | 86.0 | 86 | 88.6 | 87 |
| RFC | CH₄ | 77.3 | 77 | 82.6 | 82 | 84.0 | 87 |
| RFC | H₂O | 67.4 | 69 | 75.5 | 74 | 85.3 | 82 |
| RFC | CO₂ | 79.6 | 76 | 77.7 | 79 | 83.1 | 83 |
| RFC | NH₃ | 79.0 | 78 | 84.9 | 85 | 87.8 | 87 |
| SVC | CH₄ | 81.0 | 79 | 86.0 | 86 | 87.2 | 89 |
| SVC | H₂O | 68.7 | 69 | 77.7 | 78 | 86.1 | 84 |
| SVC | CO₂ | 81.3 | 79 | 80.9 | 83 | 86.1 | 84 |
| SVC | NH₃ | 78.2 | 81 | 86.0 | 87 | 88.9 | 87 |
<!-- /table:repro -->

## References

1. Tinetti G, Drossart P, Eccleston P, et al. A chemical survey of exoplanets with ARIEL. Exp Astron, 46: 135-209, 2018. https://doi.org/10.1007/s10686-018-9598-x
2. Edwards B, Mugnai L, Tinetti G, Pascale E, Sarkar S. An updated study of potential targets for Ariel. Astron J, 157: 242, 2019. https://doi.org/10.3847/1538-3881/ab1cb9
3. Edwards B, Tinetti G. The Ariel target list: the impact of TESS and the potential for characterizing multiple planets within a system. Astron J, 164: 15, 2022. https://doi.org/10.3847/1538-3881/ac6bf9
4. Radica M, Cowan NB, Cloutier R, Wang LY. On the information content of Ariel transmission spectra: reassessing the tier system. arXiv, 2604.07598, 2026. https://doi.org/10.48550/arXiv.2604.07598
5. Changeat Q, Yip KH. ESA-Ariel Data Challenge NeurIPS 2022: introduction to exo-atmospheric studies and presentation of the Atmospheric Big Challenge (ABC) database. RAS Tech Instrum, 2: 45-61, 2023. https://doi.org/10.1093/rasti/rzad001
6. Mugnai LV, Al-Refaie A, Bocchieri A, Changeat Q, Pascale E, Tinetti G. Alfnoor: assessing the information content of Ariel's low-resolution spectra with planetary population studies. Astron J, 162: 288, 2021. https://doi.org/10.3847/1538-3881/ac2e92
7. Nixon MC, Madhusudhan N. Assessment of supervised machine learning for atmospheric retrieval of exoplanets. Mon Not R Astron Soc, 496: 269-281, 2020. https://doi.org/10.1093/mnras/staa1150
8. Gebhard TD, Wildberger J, Dax M, et al. Flow matching for atmospheric retrieval of exoplanets: where reliability meets adaptive noise levels. Astron Astrophys, 693: A42, 2025. https://doi.org/10.1051/0004-6361/202451861
9. Sing DK, Fortney JJ, Nikolov N, et al. A continuum from clear to cloudy hot-Jupiter exoplanets without primordial water depletion. Nature, 529: 59-62, 2016. https://doi.org/10.1038/nature16068
10. Pinhas A, Madhusudhan N, Gandhi S, MacDonald R. H2O abundances and cloud properties in ten hot giant exoplanets. Mon Not R Astron Soc, 482: 1485-1498, 2019. https://doi.org/10.1093/mnras/sty2544
11. Rackham BV, Apai D, Giampapa MS. The transit light source effect: false spectral features and incorrect densities for M-dwarf transiting planets. Astrophys J, 853: 122, 2018. https://doi.org/10.3847/1538-4357/aaa08c
12. Kempton EM-R, Lupu R, Owusu-Asare A, Slough P, Cale B. Exo-Transmit: an open-source code for calculating transmission spectra for exoplanet atmospheres of varied composition. Publ Astron Soc Pac, 129: 044402, 2017. https://doi.org/10.1088/1538-3873/aa61ef
13. Chubb KL, Rocchetto M, Yurchenko SN, et al. The ExoMolOP database: cross sections and k-tables for molecules of interest in high-temperature exoplanet atmospheres. Astron Astrophys, 646: A21, 2021. https://doi.org/10.1051/0004-6361/202038350
14. Madhusudhan N. C/O ratio as a dimension for characterizing exoplanetary atmospheres. Astrophys J, 758: 36, 2012. https://doi.org/10.1088/0004-637X/758/1/36
15. Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and perturbations. Int Conf Learn Represent, 2019. https://doi.org/10.48550/arXiv.1903.12261
16. Jaeger PF, Lüth CT, Klein L, Bungert TJ. A call to reflect on evaluation practices for failure detection in image classification. Int Conf Learn Represent, 2023. https://doi.org/10.48550/arXiv.2211.15259
17. Cannon P, Ward D, Schmon SM. Investigating the impact of model misspecification in neural simulation-based inference. arXiv, 2209.01845, 2022. https://doi.org/10.48550/arXiv.2209.01845
18. Schmitt M, Bürkner P-C, Köthe U, Radev ST. Detecting model misspecification in amortized Bayesian inference with neural networks. In: Pattern Recognition (DAGM GCPR 2023), Lecture Notes in Computer Science 14264: 541-557, 2024. https://doi.org/10.1007/978-3-031-54605-1_35
19. Ardévol Martínez F, Min M, Kamp I, Palmer PI. Convolutional neural networks as an alternative to Bayesian retrievals for interpreting exoplanet transmission spectra. Astron Astrophys, 662: A108, 2022. https://doi.org/10.1051/0004-6361/202142976
20. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res, 12: 2825-2830, 2011. https://doi.org/10.48550/arXiv.1201.0490
21. Freedman RS, Lustig-Yaeger J, Fortney JJ, Lupu RE, Marley MS, Lodders K. Gaseous mean opacities for giant planet and ultracool dwarf atmospheres over a range of metallicities and temperatures. Astrophys J Suppl Ser, 214: 25, 2014. https://doi.org/10.1088/0067-0049/214/2/25
22. Al-Refaie AF, Changeat Q, Waldmann IP, Tinetti G. TauREx 3: a fast, dynamic, and extendable framework for retrievals. Astrophys J, 917: 37, 2021. https://doi.org/10.3847/1538-4357/ac0252
23. Duque-Castaño DS, Zuluaga JI, Flor-Torres L. Machine-assisted classification of potential biosignatures in Earth-like exoplanets using low signal-to-noise ratio transmission spectra. Mon Not R Astron Soc, 539: 1528-1552, 2025. https://doi.org/10.1093/mnras/staf563
24. Yang O. ExoTransmitOpacity reads Exo-Transmit table pressures as bar (x1e5), but the tables are in Pa. TauREx 3 issue 172, GitHub, 2026. https://github.com/ucl-exoplanets/taurex3/issues/172
25. Mugnai LV, Bocchieri A, Pascale E, Lorenzani A, Papageorgiou A. ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission. Exp Astron, 59: 9, 2025. https://doi.org/10.1007/s10686-024-09976-2
26. Moses JI, Madhusudhan N, Visscher C, Freedman RS. Chemical consequences of the C/O ratio on hot Jupiters: examples from WASP-12b, CoRoT-2b, XO-1b, and HD 189733b. Astrophys J, 763: 25, 2013. https://doi.org/10.1088/0004-637X/763/1/25
27. Stock JW, Kitzmann D, Patzer ABC, Sedlmayr E. FastChem: a computer program for efficient complex chemical equilibrium calculations in the neutral/ionized gas phase with applications to stellar and planetary atmospheres. Mon Not R Astron Soc, 479: 865-874, 2018. https://doi.org/10.1093/mnras/sty1531
28. Mugnai LV, Pascale E, Edwards B, Papageorgiou A, Sarkar S. ArielRad: the Ariel radiometric model. Exp Astron, 50: 303-328, 2020. https://doi.org/10.1007/s10686-020-09676-7
29. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min, 785-794, 2016. https://doi.org/10.1145/2939672.2939785
30. Lee J-M, Heng K, Irwin PGJ. Atmospheric retrieval analysis of the directly imaged exoplanet HR 8799b. Astrophys J, 778: 97, 2013. https://doi.org/10.1088/0004-637X/778/2/97
31. Husser T-O, Wende-von Berg S, Dreizler S, et al. A new extensive library of PHOENIX stellar atmospheres and synthetic spectra. Astron Astrophys, 553: A6, 2013. https://doi.org/10.1051/0004-6361/201219058
32. Tobin J, Fong R, Ray A, Schneider J, Zaremba W, Abbeel P. Domain randomization for transferring deep neural networks from simulation to the real world. Proc IEEE/RSJ Int Conf Intell Robots Syst, 23-30, 2017. https://doi.org/10.1109/IROS.2017.8202133
33. Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv Neural Inf Process Syst, 30: 6402-6413, 2017. https://doi.org/10.48550/arXiv.1612.01474
34. Lee K, Lee K, Lee H, Shin J. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Adv Neural Inf Process Syst, 31: 7167-7177, 2018. https://doi.org/10.48550/arXiv.1807.03888
35. Sun Y, Ming Y, Zhu X, Li Y. Out-of-distribution detection with deep nearest neighbors. Proc Int Conf Mach Learn, 162: 20827-20840, 2022. https://doi.org/10.48550/arXiv.2204.06507
36. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. Adv Neural Inf Process Syst, 30: 4878-4887, 2017. https://doi.org/10.48550/arXiv.1705.08500
37. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv, 2107.07511, 2021. https://doi.org/10.48550/arXiv.2107.07511
38. Tibshirani RJ, Barber RF, Candès EJ, Ramdas A. Conformal prediction under covariate shift. Adv Neural Inf Process Syst, 32: 2530-2540, 2019. https://doi.org/10.48550/arXiv.1904.06019
