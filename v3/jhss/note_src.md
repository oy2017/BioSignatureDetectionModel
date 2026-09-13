# Can a simulator-trained classifier for Ariel Tier-1 spectra be trusted?

Owen Yang<sup>a</sup>

<sup>a</sup> Henry M. Gunn High School, 780 Arastradero Rd, Palo Alto, CA 94306, USA. Tel.: +1 650 686 7813. E-mail: owenhyang@gmail.com (corresponding author)

---

## Abstract

Machine-learning classifiers have been proposed for sorting the roughly 1,000 planets of Ariel's Tier-1 reconnaissance survey by composition, but no Ariel data exist, so such classifiers have been trained and validated only on spectra from the simulator that produced them. This technical note asked whether one such classifier could be trusted when real atmospheres differ from the simulator. The molecular classifiers proposed in the Ariel consortium's Tier-1 population study, whose code was not released, were re-implemented from the published description; the re-implementation reproduced all 48 published accuracies within 3.7 points. The classifier was then tested against fifteen physically motivated departures from its simulator. For each departure the test measured the accuracy lost, the part of that loss a retrained classifier could recover, whether training on a randomized grid absorbed it, and whether confidence scores, distance-based novelty scores or conformal prediction flagged the errors. The classifier was robust to an alternative opacity database, an alternative radiative-transfer code and two omitted absorbers (losses of 0.1-3.5 points). It was not robust to haze: at a haze strength comparable to the weakest retrieved for observed hot Jupiters, it reported methane and water absent on nearly every planet that contained them, with rising confidence and no warning from its own ensemble, although 87 % of the loss was recoverable. Removing the three optical photometric points from its input removed most of this failure, which identified those points as the cause. A second classifier, built for a different Ariel science question, showed the opposite pattern, so the trust profile of one classifier did not carry over to another. Each classifier therefore needs its own stress test before its outputs are used to select planets.

**Keywords:** exoplanet atmospheres; transmission spectroscopy; Ariel; Tier-1 survey; machine learning; classification; reproducibility; re-implementation; robustness; domain shift; hazes; stellar contamination; out-of-distribution detection; conformal prediction

## 1. Introduction

When a planet passes in front of its star, a small fraction of the starlight passes through the planet's atmosphere on its way to the observer. Molecules in the atmosphere absorb that light at their own characteristic wavelengths, so the apparent size of the planet changes with wavelength. The record of that change, the transmission spectrum, shows which molecules are present. Ariel, a European Space Agency mission scheduled for launch in 2029, will measure transmission spectra of about 1,000 transiting exoplanets between 0.5 and 7.8 µm, using three broadband photometers below 1.1 µm and three spectrometers above {tinetti2018}. Telescope time will not allow every planet to be studied deeply. The survey is therefore organised in tiers: every target is observed first at low signal-to-noise in a Tier-1 reconnaissance pass, and a substantial subset is then re-observed at higher precision in Tier 2 {tinetti2018, edwards2019, edwards2022, radica2026}. Tier-1 data are expected to reveal the presence of molecular features and the degree of cloudiness {radica2026}, so they are the natural basis for deciding which planets receive the deeper observations. A planet judged uninteresting at Tier 1 may never be observed again.

Making that decision for about 1,000 planets is itself a problem. The standard way to read a spectrum, an atmospheric retrieval, fits a physical model with sampling algorithms that need between 10⁵ and 10⁸ model evaluations per planet {changeat2023}, and its assumptions and priors must be chosen and checked by a specialist {mugnai2021}. Machine-learning classifiers are an attractive alternative: once trained, they return an answer for every planet in seconds. The Ariel consortium's population study of the Tier-1 survey proposed two retrieval-free ways of classifying planets: a band-based metric and, as a preliminary assessment, four scikit-learn classifiers trained to flag the presence of CH₄, H₂O, CO₂ and NH₃ {mugnai2021}. The classifiers reached 64-89 % accuracy on a held-out simulated population. Their code and trained models were not released.

Because Ariel has not yet flown, these classifiers, like the classifiers and retrieval networks developed for the Ariel Data Challenges {changeat2023} and the wider machine-learning retrieval literature {nixon2020, gebhard2025}, were trained and tested on spectra from the same simulator. That test shows that a classifier has learned its simulator. It cannot show how the classifier behaves when real atmospheres differ from the simulator, and there are well-documented ways in which they will. Hazes and clouds are common in observed hot Jupiters and mute or reshape their spectra across a continuum from clear to strongly obscured {sing2016, pinhas2019}. Unocculted spots on the host star imprint false spectral features, most strongly for cool stars {rackham2018}. Opacity data compiled from different line lists give different absorption cross sections {kempton2017, chubb2021}. Training grids include only a handful of absorbers: the database of the first Ariel Data Challenge contains five {changeat2023}, while carbon-rich atmospheres, for example, are expected to carry HCN and C₂H₂ in abundance {madhusudhan2012}. The authors of the consortium study acknowledged the limitation, describing their spectra as "transmission spectral shapes" to test methods against {mugnai2021}. The concern is practical. A classifier that fails in these situations does not announce it; it returns confident answers, and the planets it wrongly dismisses do not receive the observations that would have revealed them.

Deliberately altering the test data and measuring the consequences is established practice in machine learning {hendrycks2019, jaeger2023} and in simulation-based inference {cannon2022, schmitt2023}. In exoplanet atmospheres, the nearest study tested a retrieval network on spectra with an added absorber, a removed absorber and unocculted star spots {ardevol2022}, and a recent flow-matching retrieval left model mismatch such as unmodelled clouds to future work {gebhard2025}. No test of this kind had been reported for a classifier proposed for Ariel's Tier-1 survey, and because the consortium's code was not released, any such test first requires an independent re-implementation that demonstrably matches the published results.

This technical note asked whether the consortium's Tier-1 molecular classifiers could be trusted when real atmospheres differ from their simulator, and whether the answer carries over to other classifiers. It re-implemented the classifiers from their published description and verified the re-implementation against all 48 published accuracies (Section 3.1). It then subjected them to a stress test of fifteen physically motivated departures from the simulator, measuring for each how much accuracy was lost, how much of the loss any retraining could recover, whether training on a randomized grid absorbed it, and whether the classifiers' confidence, distance-based novelty scores or conformal prediction flagged the errors (Sections 3.2-3.4). To find out whether the resulting trust profile belonged to this classifier or to Ariel classifiers in general, the same test was applied to a second classifier, built for a different Ariel science question, the carbon-to-oxygen ratio; a single classifier with a different profile is enough to show that one classifier's stress test cannot stand in for another's (Section 3.5).

The test showed that the published design was robust to a different opacity database, a different radiative-transfer code and omitted absorbers, but was confidently blind to haze of an observed strength: it reported methane and water absent while their infrared bands were intact, because it relied on the three optical photometric points. The second classifier had a nearly opposite profile. These results are useful in three ways. Anyone applying or extending the consortium design gains a specific weakness and two remedies. Anyone proposing a classifier for Ariel gains a stress-test procedure, applicable to any classifier trained on simulated spectra, with released scripts for the two classifiers studied here. And selecting Ariel's deeper-tier targets with any simulator-trained classifier gains a concrete requirement: the classifier must be tested on its own, against the ways its simulator can be wrong, before its outputs are used. All code, result tables and the expectations recorded before the runs are in a public repository directory (see Data and Code Availability).

## 2. Materials and Methods

### 2.1 The published classifier and its re-implementation

The re-implementation followed Sections 2.2 and 2.5 of the consortium study {mugnai2021} element by element (Table 1). Planet populations were built on the Ariel candidate list. The training population (POP-III) repeated each planet four times with independently drawn atmospheres, and the test population (POP-I) used each planet once. Each atmosphere was isothermal at a temperature drawn between 0.7 and 1.05 times the planet's equilibrium temperature, with 100 layers spanning 10⁻⁴ to 10⁶ Pa, an H₂/He fill with He/H₂ = 0.17, constant-with-altitude abundances of CH₄, H₂O, CO₂ and NH₃ drawn log-uniformly (10⁻⁹-10⁻² for training, 10⁻⁷-10⁻² for testing), and a grey opaque cloud deck at a pressure drawn log-uniformly between 5 × 10² and 10⁶ Pa. Each spectrum was binned at the study's Tier-3 resolution (R = 20, 100 and 30 in NIRSpec, AIRS-CH0 and AIRS-CH1, plus the three photometric channels; 104 points) and scattered with Gaussian noise equal to the Tier-3 noise of each bin rescaled to the number of transits the planet needs for Tier 1. Each spectrum was normalised to zero mean and unit dispersion. A molecule was labelled present if its abundance exceeded 10⁻⁵, 10⁻⁴ or 10⁻³. The classifiers were scikit-learn's k-nearest-neighbour (k = 5), multilayer-perceptron (one hidden layer of 100 units), random-forest and support-vector classifiers at default settings {pedregosa2011}. These defaults have not changed since version 0.22, the version the study cites.

**Table 1.** Elements of the published classifier design, the statement in the consortium study {mugnai2021} (page of the arXiv version), and the re-implementation.

| Element | Published statement | Re-implementation |
| :-- | :-- | :-- |
| Planet list | Ariel candidate list, 1,000 planets including predicted TESS discoveries (p. 4) | 965 known planets of the 2026 Mission Candidate Sample |
| Training and test populations | POP-III: each planet repeated 4 times; POP-I: each planet once (pp. 5-6) | Same |
| Temperature | Between 0.7 and 1.05 of the equilibrium temperature, isothermal (p. 4) | Same |
| Atmosphere grid | 100 layers, 10⁻⁴ to 10⁶ Pa; H₂ and He with He/H₂ = 0.17 (p. 4) | Same |
| Abundances | CH₄, H₂O, CO₂, NH₃ log-uniform, 10⁻⁷-10⁻² (POP-I) and 10⁻⁹-10⁻² (POP-III) (pp. 4, 6) | Same |
| Clouds | Grey opaque deck, 5 × 10² to 10⁶ Pa (p. 4) | Same |
| Opacities | ExoMol line lists; H₂-H₂ and H₂-He collision-induced absorption (p. 5, Table 2) | Exo-Transmit tables {kempton2017}; no collision-induced absorption (Section 2.2) |
| Binning | Tier 3, R = 20, 100, 30 (pp. 2, 5) | Same, with the three photometric points |
| Noise | "The noise estimated with ArielRad at each spectral bin … a re-scaled version of the Tier 3 noise, obtained by combining the number of transit observations needed to match the Tier 1 required SNR" (p. 5) | Payload noise model at each target's Tier-1 transit count (Section 2.3) |
| Normalisation | "Each example spectrum is normalised to zero mean and unit dispersion" (p. 11) | Same |
| Classifiers | scikit-learn defaults: KNN with k = 5, MLP with 100 hidden units, RFC, SVC one-vs-one (p. 12) | Same (scikit-learn 1.7) |
| Evaluation | Trained on POP-III, tested on POP-I, accuracy at three thresholds (p. 18) | Same |

### 2.2 Readings of the publication and unavoidable deviations

Two details were not stated explicitly, and each was resolved from the text and checked against the published results. First, the study did not state explicitly that the classifier input included the three optical photometric points. It stated that the classifiers learn "from their spectral shape over the whole wavelength range sampled by Ariel" and "gather information from all the spectral data points" {mugnai2021}, and its figures of the observed Tier-1 spectra show the photometric points. The re-implementation therefore included them. Second, the noise of each Tier-3 bin was computed from a payload noise model rather than from ArielRad, which is not public (Section 2.3).

Three deviations could not be avoided. (i) The opacities were Exo-Transmit's tables {kempton2017, freedman2014} rather than ExoMol cross sections. Replacing the tables with ExoMol cross sections {chubb2021} for H₂O, CH₄, CO₂ and CO cost the re-implemented classifier 2.9 points (Section 3.2), which bounds the effect of this choice. (ii) Collision-induced absorption was not included. Adding the H₂-H₂ and H₂-He contributions changed test spectra by at most 3 ppm, against feature amplitudes of 270-1,400 ppm. (iii) The planet list was the 965 known planets of the current candidate sample {edwards2022} instead of the 1,000 planets of the 2019 list {edwards2019}, which included predicted TESS discoveries.

An earlier reading of the publication, which binned the spectra to the seven Tier-1 points, truncated the atmosphere at 1 Pa and set the noise exactly at the Tier-1 requirement, missed the published accuracies by 10 points on average and by up to 38 points. The comparison with the published table therefore discriminated between readings: it did not pass an implementation that departed from the recipe.

### 2.3 Forward model and noise

Spectra were computed with TauREx 3 {alrefaie2021} through MultiREx {duque2025} at a native resolution of R ≈ 1,000 and binned by integration. TauREx 3.3.2 reads the pressure grid of Exo-Transmit opacity tables as bar, whereas Exo-Transmit's own reader and manual treat it as Pa; the pressure grid was therefore divided by 10⁵ at load. The inconsistency was reported to the TauREx developers {taurex172}. With the correction, TauREx spectra of 1,804 test atmospheres matched Exo-Transmit's own code more closely (median shape correlation 0.9986 against 0.986 without it).

The noise of each bin came from a payload noise model built with ExoSim 2 {mugnai2025}: the one-hour noise-to-signal ratio of each bin for the host's effective temperature, scaled to the host's distance and radius and to the transit duration, divided by the square root of the number of transits. Its single free level was calibrated so that a signal-to-noise ratio of 7 on the five-scale-height atmospheric signal reproduced the candidate sample's Tier-3 transit counts; the calibrated model predicted those counts with a scatter of 0.07 dex (17 %). For the three example planets of the study's Figure 1 (HD 209458 b, GJ 1214 b and WASP-79 b), the model's per-bin noise in the AIRS channels (about 75, 430 and 230 ppm) was of the same size as the published noise bands.

### 2.4 A second classifier, for a different Ariel science question

A second classifier was built to test whether the results for the consortium classifier were specific to it. It addressed the atmospheric carbon-to-oxygen ratio, a stated Ariel science objective {tinetti2018}, and labelled an atmosphere carbon-rich if C/O > 1, the chemical boundary above which oxygen is locked in CO and water is depleted {madhusudhan2012, moses2013}. The grid held 18,040 training and 9,016 test atmospheres in five independent test sets, with planet radius 1-26 R⊕, mass 1-300 M⊕, host temperature 2,500-7,500 K, atmospheric temperature 500-2,500 K, C/O uniform in 0.2-1.8 and metallicity uniform in −1 to 1.5 dex, each drawn independently. Abundances of H₂O, CH₄, CO, CO₂, NH₃ and O₃ followed from chemical equilibrium computed with FastChem {stock2018} at 10⁻² bar. Spectra were binned to the Ariel Tier-3 layout of the mission's radiometric model (R = 15, 100 and 30, 102 points) {mugnai2020} and at the seven-point Tier-1 layout. The classifier was gradient-boosted trees {chen2016} on per-spectrum-normalised bins, tuned by cross-validation, with noise at a signal-to-noise ratio of 15 on each spectrum's amplitude; on the known Ariel targets, noise came from the payload model of Section 2.3. It reached 96.1 % on the simulated test sets (95.9 ± 0.1 % across ten training resamples).

The second classifier differed from the first in label, grid, model and noise convention. The comparison was therefore not designed to show why the two classifiers respond differently. It was designed to show whether they respond alike. If a classifier built with standard practice for a stated Ariel goal had a different trust profile, then no single classifier's stress test could be taken as representative, and each classifier would need its own. Reasons for any difference were tested separately (Sections 3.3 and 3.5).

### 2.5 The stress test

Each departure from the simulator re-rendered or modified the same test planets, so every loss was a within-planet difference against the same noise draw. Table 2 lists the departures. Hazes and clouds are common among observed hot Jupiters, whose spectra span a continuum from clear to strongly muted {sing2016}. Haze followed the Lee et al. prescription {lee2013} with 0.1 µm particles. On the Rayleigh-enhancement scale used in retrievals of observed hot Jupiters {pinhas2019}, where the haze opacity at 0.35 µm is expressed as a multiple a of the H₂ Rayleigh opacity, the densities 3 × 10⁷, 2.4 × 10⁸ and 10¹⁰ m⁻³ correspond to log₁₀ a ≈ 2.4-2.7, 3.3-3.6 and 4.9-5.2 at 1 mbar. The ten hot Jupiters of that retrieval study spanned log₁₀ a = 2.1-5.7 {pinhas2019}, so 3 × 10⁷ m⁻³ lies at the weak end of the observed range. Unocculted spots used PHOENIX spectra {husser2013} at a spot-to-photosphere temperature ratio of 0.85 {rackham2018}.

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

Five quantities were measured for each departure, at the 10⁻⁴ threshold and averaged over the four classifiers and four molecules unless stated. (i) The loss: clean accuracy minus accuracy under the departure. (ii) The irreducible loss: clean accuracy minus the accuracy of the same design retrained on training spectra with the same departure, so that the reducible part of a loss was measured against what retraining could achieve. (iii) Absorption: the loss of the design retrained on a randomized training grid, in which each planet carried a haze (60 % of planets, log-uniform 10⁵-3 × 10⁸ m⁻³), spots (70 %, 0-20 %) and a noise level 1-3 times the nominal one, drawn independently {tobin2017}; three further variants each left one ingredient out, to measure whether robustness transferred to a departure never seen {hendrycks2019}. (iv) Detection: five rules for declining a planet, namely the probability margin, the spread of the four classifiers' probabilities {lakshminarayanan2017}, the Mahalanobis distance {lee2018}, the k-nearest-neighbour distance {sun2022} and a principal-component reconstruction error. Each rule's threshold was fixed to decline 10 % of clean test planets, and each rule was credited only with the accuracy of accepted planets above the clean selective baseline at the same coverage {geifman2017, jaeger2023}. (v) Calibration: expected calibration error and the coverage of split-conformal prediction sets calibrated to 90 % on half of the clean planets {angelopoulos2021, tibshirani2019}, for the three classifiers that output probabilities.

Expectations for the consortium classifier were written down and committed to the repository before these runs, and are compared with the outcomes in Section 3.6.

## 3. Results and Discussion

### 3.1 The re-implementation reproduced the published accuracies

![Figure 1](figures/note_fig1_reproduction.png)

**Figure 1.** Accuracy of the re-implemented classifiers against the published values for the 48 combinations of classifier, molecule and abundance threshold. The dashed line marks equality and the grey band ±5 points.

Across all 48 published accuracies, the re-implementation differed by +0.4 points on average, by 1.4 points in absolute value, and by at most 3.7 points (Figure 1; Appendix A). The correlation between published and re-implemented values was 0.96. The re-implementation also reproduced the published patterns. Accuracy rose with the abundance threshold for 15 of the 16 classifier-molecule pairs, against 16 of 16 in the publication, and water was the hardest molecule at the 10⁻⁵ and 10⁻⁴ thresholds in both. At the 10⁻⁴ threshold, the mean accuracies were 84.9 % for CH₄, 74.9 % for H₂O, 79.5 % for CO₂ and 84.4 % for NH₃, against 84.0, 74.8, 80.5 and 85.0 % published. Because a mistaken reading of the recipe had missed by 10 points on average (Section 2.2), this agreement was taken as evidence that the re-implementation followed the published design.

### 3.2 Where the classifier could and could not be trusted

![Figure 2](figures/note_fig2_consortium_map.png)

**Figure 2.** The re-implemented consortium classifier under each departure from its simulator (abundance threshold 10⁻⁴, mean over the four classifiers and four molecules). Open circles: accuracy lost by the classifier as published; filled circles: loss of the same design retrained on a randomized grid; black ticks: irreducible loss, what a design retrained at the test condition still lost. Error bars: standard deviation of the loss across the four molecules.

**Table 3.** The re-implemented consortium classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Loss, irreducible loss and randomized loss are in accuracy points against the clean accuracy. The margin rule's threshold declines 10 % of clean planets; AUROC (error) measures how well the margin ranks the classifier's errors, and distance AUROC (shift) how well the Mahalanobis distance separates altered from clean spectra (0.5 is chance). Conformal coverage is for sets calibrated to 90 % on clean spectra.

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

The classifier could be trusted under three departures that are often thought to be serious (Table 3, Figure 2). Replacing the opacity tables with the ExoMol database cost 2.9 points, running the same atmospheres through Exo-Transmit's own code cost 3.5 points, and adding HCN and C₂H₂ cost 0.1 points. A grey cloud deck at 10³ Pa, inside the published cloud prior, cost 2.9 points.

It could not be trusted under haze, star spots, high cloud decks or extra noise. A haze of 2 × 10⁶ m⁻³ already cost 12.1 points; at 3 × 10⁷ m⁻³, the weak end of observed hot-Jupiter hazes, the loss was 17.7 points, and spots and haze together cost 18.8 points. Spots covering 20 % of the stellar disk cost 10.8 points and a cloud deck at 10² Pa 9.8 points.

The losses differed in whether they could be recovered. The same design retrained with haze reached 78.8 %, so 87 % of the haze loss was reducible and the information needed to detect the molecules survived in the hazy spectra. For spots, the high cloud deck and the compound departure, 65-80 % was reducible. Noise losses were mostly irreducible (13-21 % reducible), as expected when information is removed from the data. The randomized training grid cost 1.8 points on clean spectra and recovered 71 % of the reducible haze loss and 75-82 % of the spot loss; even with haze left out of the grid, randomizing spots and noise recovered 62 % of the improvement on hazy spectra. It did not help with the opacity database, the radiative-transfer code or the gain ramp, which the grid did not contain.

### 3.3 Why haze blinded the classifier

![Figure 3](figures/note_fig3_haze.png)

**Figure 3.** Why haze blinded the consortium classifier. (a) A test planet with CH₄ above 10⁻⁴ and a clear atmosphere, at the classifier's Tier-3 binning, without haze and with haze at 3 × 10⁷ m⁻³ (noise-free); the shaded region contains the three photometric points. (b) Share of planets that contain each molecule for which the classifier reported the molecule, against haze density, for the published input, the input without the three photometric points and the AIRS bins only (mean over the four classifiers and four molecules).

Under haze at 3 × 10⁷ m⁻³, the multilayer-perceptron, random-forest and support-vector classifiers reported CH₄ on 0-5 % of test planets, against 27-33 % on clean spectra and 38 % truly present, and found it on 0-13 % of the planets that contained it. For H₂O their detection rate fell to 0 %. The multilayer perceptron's mean confidence rose from 0.93 to 1.00. The haze did not hide the molecular bands: it added a median 1,094 ppm to the three optical photometric points, 140 ppm across NIRSpec and 5 ppm across AIRS, and left the peak-to-peak amplitude of the AIRS bands unchanged (ratio 1.00; Figure 3a).

Four versions of the design, identical except for the input, identified the cause (Table 4). Normalising each spectrum with the statistics of the bins above 1.1 µm, while keeping every bin as input, left the failure unchanged, so the normalisation was not the cause. Removing the three photometric points reduced the loss at 3 × 10⁷ m⁻³ from 17.7 to 5.6 points and the loss to 20 % spots from 10.8 to 3.2 points; using only the AIRS bins reduced them to 0.5 points and to no measurable loss. Transplanting only the three hazy photometric points into otherwise clean spectra reproduced the failure (CH₄ detection 1.6 %), and restoring them in hazy spectra removed it (72 %). The published design's sensitivity to haze and spots therefore came through its optical photometric inputs.

**Table 4.** The consortium design with four different inputs, each retrained on the same training spectra (threshold 10⁻⁴; mean over four classifiers and four molecules). Cells give accuracy in percent, with the percentage of planets containing the molecule for which it was reported in parentheses. Haze densities in m⁻³.

<!-- table:variants -->
| Classifier input | Clean | Haze 2 × 10⁶ | Haze 3 × 10⁷ | Haze 2.4 × 10⁸ | Spots 20 % | Spots + haze |
| :-- | --: | --: | --: | --: | --: | --: |
| All 104 bins (published) | 81.1 (63) | 69.0 (23) | 63.4 (6) | 62.3 (3) | 70.3 (26) | 62.2 (3) |
| All bins, normalised with the bins above 1.1 µm | 80.9 (61) | 68.6 (21) | 63.3 (6) | 62.3 (3) | 69.9 (25) | 62.5 (3) |
| Photometric points removed | 77.3 (53) | 76.8 (51) | 71.7 (33) | 65.1 (14) | 74.1 (44) | 69.5 (27) |
| AIRS bins only (above 1.95 µm) | 73.7 (44) | 73.6 (44) | 73.2 (44) | 70.8 (41) | 73.9 (45) | 73.4 (45) |
<!-- /table:variants -->

The result followed from the design and did not depend on details of the re-implementation within the ranges tested. The classifiers used every point of the spectrum including the photometric points, their training atmospheres had grey clouds but no haze, and a small-particle haze raises the optical points while leaving the infrared bands almost unchanged; any implementation of that design can therefore be misled by haze while the molecular information remains in the spectrum. The size of the effect did depend on implementation choices. With the photometric noise tripled, the haze loss was 15.6 points, and with it multiplied by ten, 10.2 points. With 0.03 µm particles at the same optical extinction the loss was 17.5 points. Particles of 0.5 µm also changed the infrared (AIRS amplitude ratio 1.94), and every input choice then failed (losses of 13.4-19.9 points), a different failure from the one described here.

The photometric points were not useless: removing them cost 3.8 points on clean spectra, and using only the AIRS bins cost 7.4 points. With haze at 3 × 10⁷ m⁻³, the version without the photometric points was the more accurate one once about a third or more of the targets carried such a haze, and the AIRS-only version once about 40 % did. Two remedies follow: training with chromatic hazes, which recovered most of the loss (Section 3.2), or excluding or down-weighting the optical points.

### 3.4 Whether the classifier could tell

None of the five decline rules restored clean-level accuracy under any serious departure, and every rule's credit against the clean selective baseline was negative under haze (−17.5 to −25.8 points), spots and the high cloud deck (Table 3). Confidence did not warn of haze: the margin rule, which declined 7.4 % of clean planets, declined 0.9 % of hazy ones, and its ability to rank the classifier's errors fell from 0.76 to 0.70. Distance scores did separate hazy spectra from clean ones (AUROC 0.90-0.94) and declined 71-82 % of them, but did not rank errors within them (AUROC 0.44-0.51), so they acted as an alarm for the whole population rather than as a way to keep the correct planets. The conformal prediction sets, calibrated to 90 % coverage on clean spectra, covered 73.7 % under haze and 82.3 % under spots, and expected calibration error rose from 0.088 to 0.231 under haze. Randomizing haze and spots into training reduced the Mahalanobis separation of hazy spectra from 0.94 to 0.38 and of spotted spectra from 0.72 to 0.45: the more robust design was also blind to the departure it had absorbed.

### 3.5 The trust profile did not carry over to a second classifier

![Figure 4](figures/note_fig4_carbonrich.png)

**Figure 4.** The trust profile of the second classifier. (a) Accuracy on carbon-rich planets without and with HCN and C₂H₂ at their equilibrium abundances: on the simulated test sets at Tier-3 binning and on the 965 known Ariel targets under the payload noise model at Tier-3, Tier-2 and Tier-1 binning. The dashed line marks 50 %. (b) Accuracy lost to haze at 3 × 10⁷ m⁻³ with all inputs and with the three photometric points removed, for the consortium classifier and for the carbon-rich classifier at Tier-3 and Tier-1 binning; the clean accuracy given up by removing the points is printed above each pair.

The carbon-rich classifier responded almost oppositely to the two departures that separated trust from distrust for the consortium classifier (Figure 4). HCN and C₂H₂ become major constituents of carbon-rich atmospheres above about 800 K {madhusudhan2012, moses2013} but are absent from the five-gas grid of the Ariel Data Challenge database {changeat2023} and from the second classifier's training grid. Adding them at their equilibrium abundances lowered accuracy on carbon-rich planets from 97.6 % to 48.9 % on the simulated test sets, while accuracy on oxygen-rich planets was unchanged (94.5 and 95.7 %). On the known Ariel targets under the payload noise model, the corresponding accuracy fell from 93.0 to 65.0 % at Tier-3 binning, from 91.4 to 61.9 % at Tier-2 binning and from 84.0 to 58.4 % at Tier-1 binning. The classifier remained confident: its expected calibration error rose from 0.009 to 0.240 and the mean probability it assigned to a true carbon-rich planet fell from 0.96 to 0.47. Retraining with the two absorbers in the grid restored 96.1 %, and a randomized grid containing haze, clouds, spots, noise and disequilibrium chemistry did not help (75.6 %). Unlike the consortium classifier, this classifier's errors were flagged by distance scores (AUROC 0.90-0.91 for ranking errors) and not by its confidence (0.70). Two absorbers that carry no information about whether CH₄ or H₂O exceeds 10⁻⁴ cost the consortium classifier nothing; two absorbers tied by chemistry to C/O cost the carbon-rich classifier 26-30 points on carbon-rich targets.

Haze, the departure that most affected the consortium classifier, cost the carbon-rich classifier 5.1 points at Tier-3 binning, and removing its photometric points changed that loss only to 2.8 points (Figure 4b). At Tier-1 binning, where three of the seven points are photometric, haze cost it 29.1 points, removing the points reduced the loss to 7.4 points, and the points carried 18.6 points of clean accuracy, so removing them was no remedy there.

The two classifiers were both built with standard practice on simulated Ariel spectra, and their trust profiles at full resolution were close to opposite: robust to omitted absorbers and fragile to haze for one, fragile to omitted absorbers and robust to haze for the other. At Tier-1 binning both were fragile to haze through the photometric points. Because the classifiers differ in several respects at once, the comparison does not isolate why; the explanations offered above rest on the separate tests of Section 3.3 and on the chemistry of the absorbers. What the comparison does establish is that the answer to "can this classifier be trusted?" obtained for one Ariel classifier did not transfer to the other.

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

The main surprise was not among the expectations: the classifier's failure under haze came from its optical photometric inputs while the molecular bands were intact, and it became more confident as it failed. Four expectations about recoverability were wrong in the same direction, since the losses to aerosols turned out to be largely recoverable. The reversed dependence of the spot loss on host type, which is contrary to the stronger chromatic contamination of M dwarfs {rackham2018}, was not explained and is not used in any conclusion.

### 3.7 Limitations

All spectra, including those of the known Ariel targets, were simulated, so the test measured behaviour under departures from one simulator family rather than on real observations. The consortium classifiers were described as a preliminary assessment {mugnai2021}, not as an adopted mission pipeline, and the results apply to the published design as re-implemented here; the trained models of the original study were not available. The re-implementation used Exo-Transmit opacities and a payload noise model in place of ExoMol and ArielRad, with the measured consequences given in Section 2.2. The haze prescription was one parameterisation, anchored to retrieved hot-Jupiter hazes through an order-of-magnitude conversion that depends on the assumed pressure level. The carbon-rich classifier is not a published proposal; it served as a counterexample, and its grid of independently drawn parameters and its atmosphere truncated at 1-10 Pa are assumptions documented in the repository. No machine-learning retrieval trained on the Ariel Data Challenge database was tested.

## 4. Conclusion

The molecular classifiers proposed for Ariel's Tier-1 survey were re-implemented from their published description and reproduced the published accuracies within 3.7 points. A stress test against fifteen departures from the simulator showed that the classifiers could be trusted under changes of opacity database, radiative-transfer code and omitted absorbers, and could not be trusted under haze, star spots and high clouds. Under haze at the weak end of the range retrieved for observed hot Jupiters, the classifiers reported methane and water absent while the infrared bands were intact, became more confident, and gave no warning; the failure came from the three optical photometric inputs, and removing those inputs or training with hazes largely removed it. A second classifier, built for Ariel's carbon-to-oxygen objective, had a nearly opposite trust profile, so a stress test of one classifier does not stand in for another. Before any simulator-trained classifier is used to select planets for Ariel's deeper tiers, it should be subjected to its own stress test. The procedure applies to any classifier trained on simulated spectra whose forward model can be re-run; the scripts released with this note implement it for the two classifiers studied and can be adapted to others.

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
