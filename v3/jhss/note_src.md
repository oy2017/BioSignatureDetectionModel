# Can a simulator-trained classifier for Ariel Tier-1 spectra be trusted?

Owen Yang<sup>a</sup>

<sup>a</sup> Henry M. Gunn High School, 780 Arastradero Rd, Palo Alto, CA 94306, USA. Tel.: +1 650 686 7813. E-mail: owenhyang@gmail.com (corresponding author)

---

## Abstract

Machine-learning classifiers have been proposed for sorting the roughly 1,000 planets of Ariel's Tier-1 reconnaissance survey by composition, but no Ariel data exist, so such classifiers have been trained and validated only on spectra from the simulator that produced them. This technical note asked whether one such proposal could be trusted when real atmospheres differ from the simulator. The molecular classifiers proposed in the Ariel consortium's Tier-1 population study, whose code was not released, were re-implemented from the published description; the re-implementation reproduced all 48 published accuracies within 3.7 points. The re-implemented classifiers were then tested against fifteen physically motivated departures from their simulator. For each departure the test measured the accuracy lost, the part of that loss a retrained classifier could recover, whether training on a randomized grid absorbed it, and whether confidence scores, distance-based novelty scores or conformal prediction flagged the errors. The classifiers were robust to an alternative opacity database, an alternative radiative-transfer code and two omitted absorbers (losses of 0.1-3.5 points). They were not robust to haze: at a haze strength comparable to the weakest retrieved for observed hot Jupiters, they reported methane and water absent on nearly every planet that contained them, with rising confidence and no warning from their own ensemble, although 87 % of the loss was recoverable. Removing the three optical photometric points from their input eliminated most of this failure, which identified those points as the cause. A second classifier, built for a different Ariel science question, showed a nearly opposite pattern at full resolution, so the trust profile of one classifier did not carry over to another. Each classifier therefore needs its own stress test before its outputs are used to select planets.

**Keywords:** exoplanet atmospheres; transmission spectroscopy; Ariel; Tier-1 survey; machine learning; classification; reproducibility; re-implementation; robustness; domain shift; hazes; stellar contamination; out-of-distribution detection; conformal prediction

## 1. Introduction

When a planet transits its star, part of the starlight passes through the planet's atmosphere, and molecules in the atmosphere absorb that light at their characteristic wavelengths. The resulting transmission spectrum shows which molecules are present. Ariel, a European Space Agency mission scheduled for launch in 2029, will measure transmission spectra of about 1,000 exoplanets between 0.5 and 7.8 µm, using three broadband photometers below 1.1 µm and three spectrometer channels (NIRSpec, AIRS-CH0 and AIRS-CH1) at longer wavelengths {tinetti2018}. Because not all planets can be studied deeply, each target is observed first at low signal-to-noise in a Tier-1 reconnaissance survey, and a subset is then re-observed at higher precision in Tier 2 {tinetti2018, edwards2019, edwards2022, radica2026}. Tier-1 spectra, which are expected to reveal molecular features and cloudiness {radica2026}, are thus the basis for choosing which planets receive deeper observations; a planet dismissed at Tier 1 may never be observed again.

Reading a spectrum with an atmospheric retrieval requires 10⁵-10⁸ model evaluations per planet {changeat2023} and specialist choices of assumptions and priors {mugnai2021}, which makes retrievals costly as a first pass over 1,000 planets. Machine-learning classifiers, which answer in seconds once trained, are an attractive alternative. The Ariel consortium's population study of the Tier-1 survey proposed, as a preliminary assessment, four scikit-learn classifiers that flag the presence of CH₄, H₂O, CO₂ and NH₃, reaching 64-89 % accuracy on simulated planets {mugnai2021}. Their code and trained models were not released.

Like the models developed for the Ariel Data Challenges {changeat2023} and in the wider machine-learning retrieval literature {nixon2020, gebhard2025}, these classifiers were trained and tested on spectra from the same simulator. Such a test shows that a classifier has learned its simulator, but not how the classifier behaves when real atmospheres differ from the simulator. Real atmospheres will differ in documented ways: hazes and clouds are common and mute or reshape spectra {sing2016, pinhas2019}; unocculted star spots imprint false features {rackham2018}; opacity data built from different line lists disagree {kempton2017, chubb2021}; and training grids omit absorbers that real atmospheres contain. The grid of the first Ariel Data Challenge, for example, contains only five absorbers {changeat2023}, whereas carbon-rich atmospheres are expected to carry HCN and C₂H₂ as well {madhusudhan2012}. A classifier that fails in these situations does not announce it. It returns confident answers, and the planets it wrongly dismisses never receive the observations that would have revealed them.

Measuring how models respond to deliberately altered data is established practice in machine learning {hendrycks2019, jaeger2023} and in simulation-based inference {cannon2022, schmitt2023}. In exoplanet atmospheres, one study tested a retrieval network against an added absorber, a removed absorber and star spots {ardevol2022}, and a recent retrieval left unmodeled clouds to future work {gebhard2025}. No such test had been reported for a classifier proposed for Ariel's Tier-1 survey, and testing the consortium classifiers first requires an independent re-implementation that reproduces the published results.

This technical note asked whether the consortium classifiers could be trusted when real atmospheres differ from their simulator, and whether this answer, the classifier's trust profile, carries over to other classifiers. The classifiers were re-implemented and checked against all 48 published accuracies (Section 3.1), then stress-tested against fifteen physically motivated departures from the simulator (Sections 3.2-3.4). The same test was applied to a second classifier, built for a different Ariel science question, the carbon-to-oxygen (C/O) ratio; a single classifier with a different trust profile suffices to show that one classifier's test cannot stand in for another's (Section 3.5).

The published design proved robust to a different opacity database, a different radiative-transfer code and omitted absorbers, but confidently blind to haze of an observed strength because of its reliance on three optical photometric inputs; the second classifier had a nearly opposite profile. The note provides an openly released re-implementation of the consortium classifiers, with a documented weakness and two remedies; a stress-test procedure for any classifier trained on simulated spectra, with scripts for the two classifiers studied; and a requirement for Ariel target selection: each simulator-trained classifier must be tested on its own before its outputs are used. Code, results and the expectations recorded before the runs are publicly available (see Data and Code Availability).

## 2. Materials and Methods

### 2.1 The published classifier and its re-implementation

The re-implementation followed Sections 2.2 and 2.5 of the consortium study {mugnai2021} element by element (Table 1). Isothermal atmospheres with random temperatures, abundances of CH₄, H₂O, CO₂ and NH₃, and gray cloud decks were generated for every planet of the Ariel candidate list, four times for training (POP-III) and once for testing (POP-I). Spectra were binned at Tier-3 resolution, scattered with Tier-1 noise and normalized per spectrum. A molecule was labeled present if its abundance exceeded 10⁻⁵, 10⁻⁴ or 10⁻³, and four scikit-learn classifiers at default settings {pedregosa2011} were trained for each molecule and threshold; these defaults have not changed since version 0.22, the version the study cites.

**Table 1.** Elements of the published classifier design, the statement in the consortium study {mugnai2021} (page of the arXiv version), and the re-implementation.

| Element | Published statement | Re-implementation |
| :-- | :-- | :-- |
| Planet list | Ariel candidate list, 1,000 planets including predicted TESS discoveries (p. 4) | 965 known planets of the 2026 Mission Candidate Sample |
| Training and test populations | POP-III: each planet repeated 4 times; POP-I: each planet once (pp. 5-6) | Same |
| Temperature | Between 0.7 and 1.05 of the equilibrium temperature, isothermal (p. 4) | Same |
| Atmosphere grid | 100 layers, 10⁻⁴ to 10⁶ Pa; H₂ and He with He/H₂ = 0.17 (p. 4) | Same |
| Abundances | CH₄, H₂O, CO₂, NH₃ log-uniform, 10⁻⁷-10⁻² (POP-I) and 10⁻⁹-10⁻² (POP-III) (pp. 4, 6) | Same |
| Clouds | Gray opaque deck, 5 × 10² to 10⁶ Pa (p. 4) | Same |
| Opacities | ExoMol line lists; H₂-H₂ and H₂-He collision-induced absorption (p. 5, Table 2) | Exo-Transmit tables {kempton2017}; no collision-induced absorption (Section 2.2) |
| Binning | Tier 3, R = 20, 100, 30 (pp. 2, 5) | Same, with the three photometric points |
| Noise | "The noise estimated with ArielRad at each spectral bin … a re-scaled version of the Tier 3 noise, obtained by combining the number of transit observations needed to match the Tier 1 required SNR" (p. 5) | Payload noise model at each target's Tier-1 transit count (Section 2.3) |
| Normalization | "Each example spectrum is normalised to zero mean and unit dispersion" (p. 11) | Same |
| Classifiers | scikit-learn defaults: KNN with k = 5, MLP with 100 hidden units, RFC, SVC one-vs-one (p. 12) | Same (scikit-learn 1.7) |
| Evaluation | Trained on POP-III, tested on POP-I, accuracy at three thresholds (p. 18) | Same |

### 2.2 Interpretation and unavoidable deviations

The study did not state whether the classifier input included the three optical photometric points. It stated that the classifiers learn "from their spectral shape over the whole wavelength range sampled by Ariel" and "gather information from all the spectral data points" {mugnai2021}, and its figures of observed Tier-1 spectra show the photometric points, so they were included.

Three deviations could not be avoided. (i) Opacities came from Exo-Transmit's tables {kempton2017, freedman2014}; substituting ExoMol cross sections {chubb2021} for H₂O, CH₄, CO₂ and CO cost the classifier 2.9 points (Section 3.2), which bounds the effect of this choice. (ii) Collision-induced absorption was omitted; adding the H₂-H₂ and H₂-He contributions changed test spectra by at most 3 ppm, against feature amplitudes of 270-1,400 ppm. (iii) Noise came from a payload model instead of ArielRad, which is not public (Section 2.3). The planet list also differed (Table 1).

An earlier reading of the publication, with spectra binned to the seven Tier-1 points, the atmosphere truncated at 1 Pa and the noise set exactly at the Tier-1 requirement, missed the published accuracies by 10 points on average and by up to 38. The comparison with the published values therefore discriminates between readings of the recipe.

### 2.3 Forward model and noise

Spectra were computed with TauREx 3 {alrefaie2021} through MultiREx {duque2025} at R ≈ 1,000 and binned by integration. TauREx 3.3.2 reads the pressure grid of Exo-Transmit opacity tables as bar, whereas Exo-Transmit treats it as Pa; the pressure grid was therefore divided by 10⁵ when the tables were loaded, and the inconsistency was reported to the TauREx developers {taurex172}. With the correction, TauREx spectra of 1,804 test atmospheres agreed better with Exo-Transmit's own code (median shape correlation 0.9986, against 0.986).

Per-bin noise came from a payload model built with ExoSim 2 {mugnai2025}: the one-hour noise-to-signal ratio for the host's effective temperature, scaled to the host's distance and radius and to the transit duration, and divided by the square root of the number of transits. Its one free parameter, the overall noise level, was calibrated so that a signal-to-noise ratio of 7 on the five-scale-height atmospheric signal reproduced the candidate sample's Tier-3 transit counts, which the model then predicted within 0.07 dex (17 %). For the three example planets in Figure 1 of the consortium study (HD 209458 b, GJ 1214 b and WASP-79 b), its AIRS noise of about 75, 430 and 230 ppm per bin matched the published noise bands in size.

### 2.4 A second classifier

The second classifier addressed the carbon-to-oxygen ratio, a stated Ariel objective {tinetti2018}: an atmosphere was labeled carbon-rich if C/O > 1, the chemical boundary above which oxygen is locked in CO and water is depleted {madhusudhan2012, moses2013}. Its grid held 18,040 training and 9,016 test atmospheres in five test sets, with independently drawn planet radius (1-26 R⊕), mass (1-300 M⊕), host temperature (2,500-7,500 K), atmospheric temperature (500-2,500 K), C/O (uniform, 0.2-1.8) and metallicity (uniform, −1 to 1.5 dex). Abundances of H₂O, CH₄, CO, CO₂, NH₃ and O₃ came from chemical equilibrium computed with FastChem {stock2018} at 10⁻² bar. Spectra were binned to the Tier-3 layout of the mission's radiometric model (R = 15, 100 and 30; 102 points) {mugnai2020} and to the seven-point Tier-1 layout. Noise was set to a signal-to-noise ratio of 15 on each spectrum's amplitude for the simulated test sets, and taken from the payload model of Section 2.3 for the known Ariel targets. Gradient-boosted trees {chen2016} on per-spectrum-normalized bins, tuned by cross-validation, reached 96.1 % on the test sets (95.9 ± 0.1 % across ten training resamples).

Because the two classifiers differ in label, grid, model and noise, their comparison can show whether their trust profiles agree, not why they differ; causes were tested separately (Sections 3.3 and 3.5).

### 2.5 The stress test

For each departure (Table 2), the same test planets were re-rendered or modified, so every loss was a within-planet difference under the same noise draw. Haze followed the Lee et al. prescription {lee2013} with particles of 0.1 µm radius. Retrievals of ten observed hot Jupiters expressed haze opacity at 0.35 µm as a multiple, a, of the H₂ Rayleigh opacity and found log₁₀ a = 2.1-5.7 {pinhas2019}; at 1 mbar, the densities 3 × 10⁷, 2.4 × 10⁸ and 10¹⁰ m⁻³ used here correspond to log₁₀ a ≈ 2.4-2.7, 3.3-3.6 and 4.9-5.2, placing 3 × 10⁷ m⁻³ at the weak end of the observed range. Spots used PHOENIX spectra {husser2013} at a spot-to-photosphere temperature ratio of 0.85 {rackham2018}.

**Table 2.** Departures from the simulator used in the stress test.

| Departure | Levels | What it represents |
| :-- | :-- | :-- |
| Gray cloud deck at a fixed pressure | 10⁴, 10³ (inside the published cloud prior), 10², 10 Pa | High, opaque clouds |
| Haze (Lee et al. Mie, 0.1 µm) | 2 × 10⁵ to 10¹⁰ m⁻³; also 0.03 and 0.5 µm particles | Photochemical hazes, absent from the training grid |
| Unocculted star spots, faculae | 2-20 % spot coverage; 10 % faculae | Stellar contamination |
| Spots and haze together | 20 % and 3 × 10⁷ m⁻³ | Compound departure |
| Extra white or correlated noise | 1.5, 2 and 3 × the per-bin σ | Underestimated or correlated noise |
| Gain ramp | 1 and 2 noise levels across the spectrum | Instrumental systematic |
| ExoMol opacity database | H₂O, CH₄, CO₂, CO | Opacity data |
| Exo-Transmit's own code | Same atmospheres, deck and He fill | Radiative-transfer code |
| HCN and C₂H₂ added | 10⁻⁷-10⁻⁴ (consortium); equilibrium abundances (second classifier) | Absorbers omitted from the training grid |

Unless stated, results are for the 10⁻⁴ threshold, averaged over the four classifiers and four molecules; "the consortium classifier" refers to this average. Five quantities were measured. (i) Loss: clean accuracy minus accuracy under the departure. (ii) Irreducible loss: clean accuracy minus the accuracy of the design retrained with the departure present, so that recovery is measured against what retraining can achieve. (iii) Absorption: the loss of the design retrained on a randomized grid in which each planet independently carried haze (60 % of planets; log-uniform, 10⁵-3 × 10⁸ m⁻³), spots (70 %; 0-20 %) and noise 1-3 times the nominal level {tobin2017}; three variants each omitted one ingredient, testing transfer to an unseen departure {hendrycks2019}. (iv) Detection: five decline rules (probability margin; spread of the four classifiers' probabilities {lakshminarayanan2017}; Mahalanobis distance {lee2018}; k-nearest-neighbor distance {sun2022}; principal-component reconstruction error), each with a threshold declining 10 % of clean planets and credited only with accepted-planet accuracy above the clean selective baseline at equal coverage {geifman2017, jaeger2023}. Rankings were summarized by the area under the receiver operating characteristic curve (AUROC), for which 0.5 is chance. (v) Calibration: expected calibration error and the coverage of split-conformal sets calibrated to 90 % on half the clean planets {angelopoulos2021, tibshirani2019}, for the three classifiers that output probabilities. Expectations for the consortium classifier were committed to the repository before the runs (Section 3.6).

## 3. Results and Discussion

### 3.1 The re-implementation reproduced the published accuracies

![Figure 1](figures/note_fig1_reproduction.png)

**Figure 1.** The re-implementation against the published accuracies for the 48 combinations of classifier, molecule and abundance threshold. (a) Re-implemented against published accuracy; dashed line, equality. (b) Difference between the two for each combination, grouped by molecule, with marker shape giving the abundance threshold. Gray bands: ±5 points.

Across the 48 published accuracies, the re-implementation differed by +0.4 points on average, 1.4 in absolute value and at most 3.7 (Figure 1; Appendix A), with a correlation of 0.96. It also reproduced the published patterns: accuracy rose with the abundance threshold for 15 of 16 classifier-molecule pairs (16 of 16 published), and water was the hardest molecule at 10⁻⁵ and 10⁻⁴ in both. At 10⁻⁴ the mean accuracies were 84.9 % (CH₄), 74.9 % (H₂O), 79.5 % (CO₂) and 84.4 % (NH₃), against 84.0, 74.8, 80.5 and 85.0 % published. Since a mistaken reading of the recipe missed by 10 points (Section 2.2), this agreement indicates that the re-implementation followed the published design.

### 3.2 Where the classifier could and could not be trusted

![Figure 2](figures/note_fig2_consortium_map.png){width=5.6}

**Figure 2.** The consortium classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Open circles: loss of the classifier as published; filled circles: loss after randomized training; black ticks: irreducible loss. Error bars: standard deviation across the four molecules.

**Table 3.** The consortium classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Losses are in points below clean accuracy. The margin rule declines 10 % of clean planets; margin AUROC (error) measures how well the margin ranks errors, and distance AUROC (shift) how well the Mahalanobis distance separates altered from clean spectra (0.5 is chance). Conformal sets were calibrated to 90 % on clean spectra.

<!-- table:consortium -->
| Mismatch | Accuracy (%) | Loss | Irreducible | Loss after randomized training | Margin rule keeps (%) | Margin AUROC (error) | Distance AUROC (shift) | Conformal coverage (%) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| None (clean) | 81.1 | — | — | 1.8 | 93 | 0.76 | — | 93 |
| Gray cloud deck, 10³ Pa | 78.2 | 2.9 | — | 2.7 | 93 | 0.74 | 0.56 | 91 |
| Gray cloud deck, 10² Pa | 71.3 | 9.8 | 2.6 | 6.1 | 94 | 0.72 | 0.66 | 85 |
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

The classifier lost little to a different opacity database (2.9 points), a different radiative-transfer code (3.5), added HCN and C₂H₂ (0.1) or a cloud deck at 10³ Pa inside its training prior (2.9 points; Table 3, Figure 2). It lost far more to haze (12.1 points at 2 × 10⁶ m⁻³, 17.7 at 3 × 10⁷ m⁻³), to spots (10.8 at 20 % coverage), to a cloud deck at 10² Pa (9.8), to spots and haze together (18.8) and to extra noise (9.3-13.5).

Most aerosol and spot losses were recoverable. Retrained with haze, the design reached 78.8 %, so 87 % of the haze loss was reducible and the molecular information survived in the hazy spectra; 65-80 % was reducible for spots, the high cloud deck and the compound departure. Noise losses were mostly irreducible (13-21 % reducible). The randomized grid cost 1.8 points on clean spectra and recovered 71 % of the reducible haze loss and 75-82 % of the spot loss; with haze omitted from the grid, randomizing spots and noise still recovered 62 % of the improvement on hazy spectra. It did not help with the opacity database, the code or the gain ramp, none of which it contained.

### 3.3 Why haze blinded the classifier

![Figure 3](figures/note_fig3_haze.png)

**Figure 3.** (a) A clear-atmosphere test planet with CH₄ above 10⁻⁴ at Tier-3 binning, without and with haze at 3 × 10⁷ m⁻³ (noise-free); shading marks the three photometric points. (b) Detection rate, the share of planets containing a molecule for which it was reported, against haze density for the published input, the input without the photometric points and the AIRS bins only (mean over four classifiers and four molecules).

Under haze at 3 × 10⁷ m⁻³, the multilayer-perceptron, random-forest and support-vector classifiers reported CH₄ on 0-5 % of planets (27-33 % on clean spectra; 38 % truly present) and detected it on 0-13 % of the planets containing it; their H₂O detection rate fell to 0 %. The multilayer perceptron's mean confidence rose from 0.93 to 1.00. Yet the molecular bands were intact: the haze added a median 1,094 ppm to the three photometric points, 140 ppm across NIRSpec and 5 ppm across AIRS, leaving the AIRS band amplitudes unchanged (ratio 1.00; Figure 3a).

Comparing four versions of the design that differed only in their input identified the cause (Table 4). Normalizing with statistics from the bins above 1.1 µm, while keeping every bin, left the failure unchanged, so normalization was not the cause. Removing the photometric points cut the loss to haze at 3 × 10⁷ m⁻³ from 17.7 to 5.6 points and the loss to 20 % spots from 10.8 to 3.2; using only the AIRS bins cut the haze loss to 0.5 points and removed the spot loss. Transplanting only the three hazy photometric points into clean spectra reproduced the failure (CH₄ detection 1.6 %), and restoring them in hazy spectra removed it (72 %). The sensitivity to haze and spots therefore came through the photometric inputs.

**Table 4.** The consortium design with four inputs, each retrained on the same spectra (threshold 10⁻⁴; mean over four classifiers and four molecules). Cells: accuracy (%), with the detection rate (%) in parentheses. Haze densities in m⁻³.

<!-- table:variants -->
| Classifier input | Clean | Haze 2 × 10⁶ | Haze 3 × 10⁷ | Haze 2.4 × 10⁸ | Spots 20 % | Spots + haze |
| :-- | --: | --: | --: | --: | --: | --: |
| All 104 bins (published) | 81.1 (63) | 69.0 (23) | 63.4 (6) | 62.3 (3) | 70.3 (26) | 62.2 (3) |
| All bins, normalized with the bins above 1.1 µm | 80.9 (61) | 68.6 (21) | 63.3 (6) | 62.3 (3) | 69.9 (25) | 62.5 (3) |
| Photometric points removed | 77.3 (53) | 76.8 (51) | 71.7 (33) | 65.1 (14) | 74.1 (44) | 69.5 (27) |
| AIRS bins only (above 1.95 µm) | 73.7 (44) | 73.6 (44) | 73.2 (44) | 70.8 (41) | 73.9 (45) | 73.4 (45) |
<!-- /table:variants -->

The mechanism follows from the design: the classifiers use every point including the photometric ones, their training atmospheres have gray clouds but no haze, and a small-particle haze raises the optical points while barely changing the infrared bands. Any implementation of the design is therefore susceptible; only the size of the effect depends on implementation choices. Tripling the photometric noise reduced the haze loss to 15.6 points and a tenfold increase to 10.2; 0.03 µm particles at the same optical extinction gave 17.5 points. Particles of 0.5 µm also altered the infrared (AIRS amplitude ratio 1.94), and every input then failed (13.4-19.9 points), a different failure.

The photometric points do carry information: removing them cost 3.8 points on clean spectra, and using AIRS alone 7.4. With haze at 3 × 10⁷ m⁻³, the input without them became the more accurate once about a third of the targets carried such a haze (about 40 % for AIRS alone). Two remedies follow: training with chromatic hazes (Section 3.2), or excluding or down-weighting the optical points.

### 3.4 Whether the classifier could tell

Under every departure that cost more than 5 points, the planets each decline rule accepted were less accurate than those it accepted on clean spectra. Every rule's credit against the clean selective baseline was negative under haze (−17.5 to −25.8 points), spots and the high cloud deck (Table 3). Confidence gave no warning of haze: the margin rule declined 7.4 % of clean planets but 0.9 % of hazy ones, and its AUROC for ranking errors fell from 0.76 to 0.70. Distance scores separated hazy from clean spectra (AUROC 0.90-0.94) and declined 71-82 % of hazy planets, but did not rank errors within them (AUROC 0.44-0.51), acting as a population alarm rather than a filter. Conformal coverage fell to 73.7 % under haze and 82.3 % under spots, and expected calibration error rose from 0.088 to 0.231 under haze. Randomizing haze and spots into training lowered the Mahalanobis separation of hazy spectra from 0.94 to 0.38 and of spotted spectra from 0.72 to 0.45: the more robust design could no longer flag the departures it had absorbed.

### 3.5 The trust profile did not carry over to a second classifier

![Figure 4](figures/note_fig4_carbonrich.png)

**Figure 4.** (a) Accuracy on carbon-rich planets without and with HCN and C₂H₂ at equilibrium abundances, on the simulated test sets (Tier 3) and on the 965 known Ariel targets with payload noise (Tiers 3, 2 and 1); dashed line, 50 %. (b) Accuracy lost to haze at 3 × 10⁷ m⁻³ with all inputs and without the three photometric points, for the consortium classifier and the carbon-rich classifier at Tier-3 and Tier-1 binning; numbers give the clean accuracy lost by removing the points.

The carbon-rich classifier responded almost oppositely to omitted absorbers and haze, the two departures that most clearly separated where the consortium classifier could and could not be trusted (Figure 4). HCN and C₂H₂, abundant in carbon-rich atmospheres above about 800 K {madhusudhan2012, moses2013}, are absent from its training grid, as from the Ariel Data Challenge grid {changeat2023}. Added at equilibrium abundances, they lowered accuracy on carbon-rich planets from 97.6 to 48.9 % on the simulated test sets, leaving oxygen-rich planets unaffected (94.5 and 95.7 %). On the known Ariel targets with payload noise, carbon-rich accuracy fell from 93.0 to 65.0 % at Tier-3 binning, from 91.4 to 61.9 % at Tier 2 and from 84.0 to 58.4 % at Tier 1. The classifier stayed confident: expected calibration error rose from 0.009 to 0.240 and the mean probability assigned to true carbon-rich planets fell from 0.96 to 0.47. Retraining with the absorbers restored 96.1 %, while a randomized grid of haze, clouds, spots, noise and disequilibrium chemistry did not help (75.6 %). Its errors, unlike the consortium classifier's errors under haze, were ranked well by distance scores (AUROC 0.90-0.91), though not by confidence (0.70). The contrast has a chemical reading: the added absorbers carry no information on whether CH₄ or H₂O exceeds 10⁻⁴, but are tied to C/O.

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

Table 5 compares the expectations with the outcomes. The largest surprise was not among the expectations: the haze failure came from the photometric inputs while the molecular bands were intact, with confidence rising as accuracy fell. Three expectations underestimated how much of the aerosol loss retraining or randomization could recover, and one overestimated the recovery of spot losses. The reversed host dependence of the spot loss, contrary to the stronger chromatic contamination of M dwarfs {rackham2018}, remains unexplained and is not used in any conclusion.

### 3.7 Limitations

All spectra, including those of the known Ariel targets, were simulated, so the test probed departures within one simulator family rather than real observations. The consortium classifiers were a preliminary proposal {mugnai2021}, and the results concern the published design as re-implemented, with the deviations of Section 2.2, not the original trained models, which were unavailable. Haze strengths were anchored to retrieved hazes through an order-of-magnitude conversion that depends on the assumed pressure level. The carbon-rich classifier, not a published proposal, served as a counterexample; its independently drawn parameters and an atmosphere truncated at 1-10 Pa are documented in the repository. No machine-learning retrieval trained on the Ariel Data Challenge database was tested.

## 4. Conclusion

The molecular classifiers proposed for Ariel's Tier-1 survey were re-implemented from their published description and reproduced the published accuracies within 3.7 points. A stress test against fifteen departures from the simulator showed that the classifiers could be trusted when the opacity database or radiative-transfer code changed or absorbers were omitted, but not under haze, star spots or high clouds. Under haze at the weak end of the range retrieved for observed hot Jupiters, the classifiers reported methane and water absent while the infrared bands were intact, became more confident, and gave no warning; the failure came from the three optical photometric inputs, and excluding those inputs or training with hazes largely eliminated it. A second classifier, built for Ariel's carbon-to-oxygen objective, had a nearly opposite trust profile at full resolution, so a stress test of one classifier does not stand in for another. Before any simulator-trained classifier is used to select planets for Ariel's deeper tiers, it should be subjected to its own stress test. The procedure applies to any classifier trained on simulated spectra whose forward model can be re-run; the scripts released with this note implement it for the two classifiers studied and can be adapted to others.

## Acknowledgements

Computations used open-source software (TauREx 3, MultiREx, Exo-Transmit, FastChem, ExoSim 2, scikit-learn, XGBoost) and public data (the ExoMolOP tables, the PHOENIX stellar atlas and the Ariel Mission Candidate Sample). No financial support was received.

The author thanks Rachael Kaci, instructor of the Advanced Authentic Research (AAR) program at Henry M. Gunn High School, and research mentor Victoria Lloyd, for their guidance and feedback throughout this project.

In the interest of transparency, the author discloses the following assistance: Anthropic's Claude Code was used to help implement the Python analysis code and to manage the project's GitHub repository, and Google Gemini was used for polishing and proofreading the manuscript. These tools were used under the author's direction; the author designed the study, directed the analysis, interpreted the results, and is responsible for all content, having reviewed and validated all AI-assisted code and text.

## Data and Code Availability

The re-implemented classifier, the stress-test scripts, every result table and figure, the expectations committed before the runs, and the corrections made to the forward model are in the directory `ariel_tier1_trust` of https://github.com/oy2017/BioSignatureDetectionModel, which serves as the supplementary file of this note.

## Appendix A. Published and re-implemented accuracies

**Table A1.** Accuracy (%) of the re-implemented classifiers and the published values {mugnai2021} for each classifier, molecule and abundance threshold (POP-III training, POP-I test).

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

<!-- references -->
