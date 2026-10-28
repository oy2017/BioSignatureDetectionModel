**Original article**

# Recoverable and irrecoverable errors of a simulator-trained classifier for Ariel Tier-1 spectra, with an external test on JWST observations

Owen Yang<sup>a</sup>

<sup>a</sup> Henry M. Gunn High School, 780 Arastradero Rd, Palo Alto, CA 94306, USA. Tel.: +1 650 686 7813. E-mail: owenhyang@gmail.com (corresponding author)

---

## Abstract

Machine-learning classifiers have been proposed for sorting the roughly 1,000 planets of Ariel's Tier-1 reconnaissance survey by composition, and a planet they dismiss may never be observed again. Such classifiers are trained and validated on spectra from one simulator, so their errors on real spectra have not been measured, and it is not known which of those errors could be fixed. This study separates a classifier's errors into two classes: recoverable, where the spectrum decides the label at its precision and the classifier missed it, and irrecoverable, where the spectrum does not decide the label, so no method can. The molecular classifiers of the Ariel consortium's Tier-1 population study were re-implemented from the published description and reproduced all 48 published accuracies within 3.7 points. A nested-sampling retrieval of every one of the 955 test planets then measured which labels each spectrum decides at Tier-1 precision. Under the consortium's own test prior, a third to a half of the four molecular labels are undecidable, three quarters of the classifier's errors on clean spectra fall on those labels, and the retrieval itself, at 365 core-hours, selects Tier-2 targets as well as an oracle. Under a photochemical haze the labels stay decidable and four fifths of the errors the haze introduces are recoverable; under threefold noise the undecidable fraction doubles and only 10 to 56 % of the added errors are recoverable. The same two classes appear on real data. On the JWST spectra of eleven planets converted to the Ariel grid by a validated procedure, the classifier reproduced the published composition of WASP-39 b, but called methane absent on five of six reductions of the three methane-bearing sub-Neptunes while the same design restricted to its AIRS spectrometer inputs found it on five of six, and five of five warm giants are muted three to five times beyond their clear twins, further than the simulator population reaches at their bulk parameters in 99 % of its planets. A retrieval with the training forward model fails on hazy spectra more confidently than the classifier, so the existing method is not a fallback for the recoverable class. A one-number score on the optical inputs finds the affected planets in the simulated population: routed to the AIRS-only variant of the design, it recovers a third of the methane targets haze removes from the Tier-2 queue at no retraining; routed to a retrieval with the haze density free, it allocates a 20 % retrieval budget as well as an oracle that knows which planets are hazed, recovering 80 % of the hazed methane targets against 17 % for a random budget and 3 % for confidence-based rules, whereas the same budget sent to the existing retrieval loses more targets than no retrieval. Every expectation was committed before its run, and the outcomes, including those not met, are reported.

**Keywords:** exoplanet atmospheres; transmission spectroscopy; Ariel; JWST; Tier-1 survey; machine learning; classification; identifiability; nested sampling; atmospheric retrieval; reproducibility; domain shift; hazes; target selection; failure detection

## 1. Introduction

When a planet transits its star, part of the starlight passes through the planet's atmosphere, and molecules in the atmosphere absorb that light at their characteristic wavelengths. The resulting transmission spectrum shows which molecules are present. Ariel, a European Space Agency mission scheduled for launch in 2029, will measure transmission spectra of about 1,000 exoplanets between 0.5 and 7.8 µm, using three broadband photometers below 1.1 µm and three spectrometer channels (NIRSpec, AIRS-CH0 and AIRS-CH1) at longer wavelengths (1). Because not all planets can be studied deeply, each target is observed first at low signal-to-noise in a Tier-1 reconnaissance survey, and a subset is then re-observed at higher precision in Tier 2 (1-4). Tier-1 spectra are the basis for choosing which planets receive deeper observations; a planet dismissed at Tier 1 may never be observed again.

Reading a spectrum with an atmospheric retrieval requires 10⁵-10⁸ model evaluations per planet (5) and specialist choices of assumptions and priors (6), which makes retrievals costly as a first pass over 1,000 planets. Machine-learning classifiers, which answer in seconds once trained, are an attractive alternative. The Ariel consortium's population study of the Tier-1 survey proposed, as a preliminary assessment, four scikit-learn classifiers that flag the presence of CH₄, H₂O, CO₂ and NH₃, reaching 64-89 % accuracy on simulated planets (6). Their code and trained models were not released.

Like the models developed for the Ariel Data Challenges (5) and in the wider machine-learning retrieval literature (7, 8, 19), these classifiers were trained and tested on spectra from the same simulator. Such a test shows that a classifier has learned its simulator, but not how it behaves when real atmospheres differ from the simulator, and real atmospheres do: hazes and clouds are common and mute or reshape spectra (9, 10), unocculted star spots imprint false features (11), opacity data built from different line lists disagree (12, 13), and training grids omit absorbers that real atmospheres contain; misspecification of this kind is known to bias simulation-based inference in general (17, 18). JWST has now measured transmission spectra of warm giants and sub-Neptunes at a precision beyond Ariel's, and those spectra carry SO₂, hazes, patchy clouds and metallicities of tens to hundreds of times solar (39-45, 47-60), none of which a classifier trained on clear hydrogen-helium atmospheres with grey clouds has seen. WASP-39 b is the spectrum on which simulator-trained retrieval models have so far been checked against real data (63, 64), and on which the retrieved abundances were shown to move by more than an order of magnitude with the data reduction alone (46).

A classifier's error on a spectrum is one of two things. Either the information was in the spectrum and the classifier missed it, which is a property of the classifier and can be fixed by retraining, by a different input or by a fallback method; or the information was not there at that precision, which is a property of the instrument and the population, and no method can fix it, so the mission must not act on those spectra as if they were decided. This study calls the two classes recoverable and irrecoverable. Which class an error belongs to is a question about the spectrum, not the classifier, and it is answered by asking whether an atmosphere of the opposite class fits the spectrum within its noise. The consortium study asked it for one planet, a retrieval that gave a 48 % probability of CH₄ above 10⁻⁴ at Tier-1 precision (6). Bocchieri et al. (65) then ran retrievals on about 300 simulated Tier-1 targets and defined a P-statistic, the posterior probability that a molecule is present, which is the quantity this study calls decidability, and Radica et al. (4) measured it for three benchmark planets across the tiers. Neither measured it on the test population of a classifier, against that classifier's errors, or under departures from the simulator, and no classifier of this kind had been checked against a real spectrum. The check matters because confidence-based failure detection does not survive a distribution shift, a result known from image classification (16) that this study's stress test of the consortium design reproduced under haze (Section 3.2).

This study measures both classes for the consortium design, under its own simulator and under each of its departures, and tests the picture on real spectra. The classifiers were re-implemented and checked against all 48 published accuracies (Section 3.1) and stress-tested against fifteen departures from the simulator, which located a confident failure under haze in the three optical photometric inputs (Section 3.2). A nested-sampling retrieval of every test planet then measured which labels the spectra decide at Tier-1 precision, and so which of the classifier's errors are recoverable (Sections 3.3 and 3.4). The JWST spectra of WASP-39 b and ten other planets, converted to the Ariel grid by a procedure validated on a spectrum whose answer is known, gave the external test (Section 3.5). Finally, diagnostics that could recognise the two classes at deployment were scored by the Tier-2 targets they save against a retrieval of every planet, the method the classifier is meant to replace (Section 3.6). Expectations for every result were recorded in the repository before the runs (Section 3.7).

## 2. Materials and Methods

### 2.1 The published classifier and its re-implementation

The re-implementation followed Sections 2.2 and 2.5 of the consortium study (6) element by element (Table 1). Isothermal atmospheres with random temperatures, abundances of CH₄, H₂O, CO₂ and NH₃, and grey cloud decks were generated for every planet of the Ariel candidate list, four times for training (POP-III) and once for testing (POP-I). Counts differ slightly between the analyses below because a few spectra failed to render or to bin under one condition or another: 955 planets have a clean test spectrum that the retrieval of Section 2.4 could score, 954 of them a hazed one, and the selection simulation of Section 2.7 uses the 939 with a valid spectrum under every departure. Spectra were binned at Tier-3 resolution, scattered with Tier-1 noise and normalised per spectrum. A molecule was labelled present if its abundance exceeded 10⁻⁵, 10⁻⁴ or 10⁻³, and four scikit-learn classifiers at default settings (20) were trained for each molecule and threshold.

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
| Noise | "The noise estimated with ArielRad at each spectral bin … a re-scaled version of the Tier 3 noise, obtained by combining the number of transit observations needed to match the Tier 1 required SNR" (p. 5) | Payload noise model at each target's Tier-1 transit count (Section 2.2) |
| Normalisation | "Each example spectrum is normalised to zero mean and unit dispersion" (p. 11) | Same |
| Classifiers | scikit-learn defaults: KNN with k = 5, MLP with 100 hidden units, RFC, SVC one-vs-one (p. 12) | Same (scikit-learn 1.7) |
| Evaluation | Trained on POP-III, tested on POP-I, accuracy at three thresholds (p. 18) | Same |

The study did not state whether the classifier input included the three optical photometric points. It stated that the classifiers learn "from their spectral shape over the whole wavelength range sampled by Ariel" and "gather information from all the spectral data points" (6), and its figures of observed Tier-1 spectra show the photometric points, so they were included. Throughout, "the published design as re-implemented" means this reading of the recipe with the deviations of Section 2.2; the original trained models were unavailable, and no claim here concerns them.

### 2.2 Forward model, noise and unavoidable deviations

Spectra were computed with TauREx 3 (22) at R ≈ 1,000 and binned by integration: isothermal atmospheres of 100 layers with constant abundances, molecular absorption and Rayleigh scattering, a grey cloud deck and, where stated, a Mie haze. They were first rendered through the MultiREx wrapper (23); the released code calls TauREx directly and reproduces them bit for bit. TauREx 3.3.2 reads the pressure grid of Exo-Transmit opacity tables as bar, whereas Exo-Transmit treats it as Pa; the pressure grid was divided by 10⁵ when the tables were loaded. The inconsistency was reported to the TauREx developers and has since been corrected in their development branch (24). With the correction, TauREx spectra of 1,804 test atmospheres agreed better with Exo-Transmit's own code (median shape correlation 0.9986, against 0.986).

Per-bin noise came from a payload model built with ExoSim 2 (25): the one-hour noise-to-signal ratio for the host's effective temperature, scaled to the host's distance and radius and to the transit duration, and divided by the square root of the number of transits. Its one free parameter was calibrated so that a signal-to-noise ratio of 7 on the five-scale-height atmospheric signal reproduced the candidate sample's Tier-3 transit counts, which the model then predicted within 0.07 dex. For the three example planets in Figure 1 of the consortium study, its AIRS noise of about 75, 430 and 230 ppm per bin matched the published noise bands in size.

Three deviations could not be avoided. (i) Opacities came from Exo-Transmit's tables (12, 21); substituting ExoMol cross sections (13) for H₂O, CH₄, CO₂ and CO cost the classifier 2.9 points (Section 3.2), which bounds the effect of this choice. (ii) Collision-induced absorption was omitted; adding it changed test spectra by at most 3 ppm, against feature amplitudes of 270-1,400 ppm. (iii) Noise came from the payload model instead of ArielRad, which is not public. An earlier reading of the publication, with spectra binned to the seven Tier-1 points, the atmosphere truncated at 1 Pa and the noise set exactly at the Tier-1 requirement, missed the published accuracies by 10 points on average and by up to 38, so the comparison with the published values discriminates between readings of the recipe.

### 2.3 The departures

For each departure (Table 2), the same test planets were re-rendered or modified, so every loss was a within-planet difference under the same noise draw. Haze followed the Lee et al. prescription (30) with particles of 0.1 µm radius. Retrievals of ten observed hot Jupiters expressed haze opacity at 0.35 µm as a multiple of the H₂ Rayleigh opacity and found log₁₀ a = 2.1-5.7 (10); at 1 mbar, the densities 3 × 10⁷, 2.4 × 10⁸ and 10¹⁰ m⁻³ used here correspond to log₁₀ a ≈ 2.4-2.7, 3.3-3.6 and 4.9-5.2, placing 3 × 10⁷ m⁻³ at the weak end of the observed range. Spots used PHOENIX spectra (31) at a spot-to-photosphere temperature ratio of 0.85 (11).

**Table 2.** Departures from the simulator.

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
| HCN and C₂H₂ added | 10⁻⁷-10⁻⁴ | Absorbers omitted from the training grid |

Unless stated, results are for the 10⁻⁴ threshold, averaged over the four classifiers and four molecules; "the classifier" refers to this average. Three input variants of the design were trained on the same spectra: all 104 bins (the published input), the 101 bins without the three photometric points ("no-optical"), and the AIRS bins above 1.95 µm alone ("AIRS-only"). Table 2 lists every level run; Table 3 reports those that matter for what follows, and the rest are in the repository. The stress test, in the manner of a corruption benchmark (15), measured the loss (clean accuracy minus accuracy under the departure), the irreducible loss (clean accuracy minus the accuracy of the design retrained with the departure present), the loss of the design trained on a randomised grid in which each planet independently carried haze, spots and extra noise (32), and five decline rules of the kind used for failure detection: the probability margin ("margin"), the spread of the four classifiers' probabilities ("ensemble"; 33), the Mahalanobis distance (34), the mean distance to the 10 nearest training spectra ("k-NN"; 35) and the principal-component reconstruction error ("PCA"), each thresholded on held-out clean spectra to decline 10 % of them. A rule is credited only with the accuracy of the planets it accepts above the clean selective baseline at the same coverage (its "selective credit"; 16, 36), with split-conformal coverage (37, 38) alongside. These five rules are the "existing rules" of Section 3.6.

### 2.4 Identifiability: which labels a spectrum decides

For planet *i* with true parameters θ and molecule *X*, the label "X > 10⁻⁴" is decidable at precision σ if no atmosphere with the opposite label, drawn from the stated population, fits the noise-free spectrum within the noise. The cheap form of this test is

Δχ²ᵢ(X) = min over θ′ in the opposite class of Σ_b [(S_b(θ) − S_b(θ′)) / σ_b]²,

where the minimum runs over the nuisance parameters of the population (temperature over 0.7-1.05 T_eq, grey cloud top over 5 × 10² to 10⁶ Pa, the other three abundances over 10⁻⁹ to 10⁻², and an additive depth offset) with *X* held at the 10⁻⁴ boundary, which is the hardest counterfactual because band depth is monotone in abundance. The sum runs over the 104 Tier-3 bins at the planet's Tier-1 per-bin σ. The minimum was found by Nelder-Mead from the true parameters and two random restarts from the prior, keeping the smallest value (median 569 forward calls per planet-molecule; 955 planets, four molecules, 207 core-hours). Under a departure the test is rerun with the departure's physics in the counterfactual model as well, with its strength held at the true value, so that it asks whether the label is decidable given a correct model of the departure. White noise at *k* times σ needs no run, since Δχ² scales as 1/*k*².

The test represents the opposite class by its single best-fitting point, which over-credits that class, so its scale was calibrated rather than assumed. A nested-sampling retrieval (61, 62; the nestle implementation, multi-ellipsoid sampler, 100 live points, at most 20,000 likelihood calls) of every one of the 955 test planets, with the same free quantities, the four abundances over 10⁻⁹ to 10⁻² and uniform priors on the population's ranges, gave for every label the posterior probability of the true class, p_true, which is the P-statistic of Bocchieri et al. (65) evaluated for the true class (median 3,961 calls and 19.9 min per planet; 365 core-hours). A label is decidable at level *q* if p_true > *q*, with *q* = 0.95 throughout and 0.84, the one-sided one-sigma level, as a looser reference; the Δχ² at which p_true crosses 0.95 was read off per molecule and used as the test's threshold. The retrieval's own accuracy, the fraction of labels with p_true > 0.5, is the accuracy ceiling the spectra allow under the population's prior. The same retrieval was then run on the 954 hazed spectra with the training forward model (the "clear-model retrieval", the existing method applied to a departed spectrum), and with the haze added to that model as an eighth free parameter, its number density log-uniform over 10³ to 10¹⁰ m⁻³, where the lower bound is haze-free to below a part per million (the "free-haze retrieval"). The free-haze retrieval is the fallback a pipeline could run on any spectrum without knowing whether it is hazed.

The population is explicit: POP-I, the consortium's test prior, at each planet's Tier-1 transit count. The true joint distribution of atmospheric conditions among Ariel's targets is unknown, so every prevalence reported here is a prevalence under this named prior, and Section 3.5 measures how the real spectra sit against it.

### 2.5 The two error classes

Each classifier error on a spectrum was assigned a class from the same spectrum's decidability. On clean spectra the posterior is the definition: an error is recoverable if p_true > 0.95 (a retrieval of that spectrum gets the label right), undecidable if 0.05 ≤ p_true ≤ 0.95, and misled if p_true < 0.05. Under a departure the departure-aware Δχ² at the calibrated threshold is used, and the errors are further split into those the learner also makes on the clean spectrum (shared) and those the departure introduces. The prevalence of each class is the fraction of the classifier's errors in it; the expectation recorded before the runs was that haze errors would be mostly recoverable and noise errors mostly irrecoverable (Section 3.7).

### 2.6 Real JWST spectra on the Ariel grid

The primary external input is the WASP-39 b benchmark of Carter & May et al. (39), the four JWST Early Release Science modes (41-44) re-reduced uniformly with fitted inter-mode offsets, 0.52-5.3 µm, extended to 7.8 µm with the MIRI LRS spectrum of Powell et al. (40). Ten further planets with public JWST transmission spectra formed a panel: TOI-270 d (48, 49), K2-18 b in three reductions (47, 50, 51), TOI-1231 b (52), TOI-421 b (55), HAT-P-18 b (53), WASP-166 b (60), and, with coverage of the AIRS channels only, WASP-107 b (56, 57), HD 189733 b (54), HAT-P-26 b (59) and GJ 3470 b (58). Labels came from the published retrievals; where a paper's abundance sat within a decade of 10⁻⁴ or the literature disagreed, the label was marked ambiguous and not scored, which left 35-37 scored labels depending on the input variant, because a split verdict (two learners each way) is not scored.

Each step of the conversion is a stated choice with a measured consequence. JWST products are transit depths, the quantity the classifier was trained on, so no conversion of quantity was needed. Each of the 104 Tier-3 bins took the inverse-variance-weighted mean of the JWST points inside it, partial overlaps weighted by the overlap fraction. Modes were aligned through their overlaps onto NIRISS order 1; for WASP-39 b this gave PRISM −177 ppm, which is the offset Carter & May report from their joint fit. Bins with no data were filled flat at the spectrum's measured mean and flagged, and the number of filled bins is reported with every verdict (one of 104 for WASP-39 b; 5-30 for the sub-Neptunes; 48-69 for the AIRS-only planets). Per-spectrum normalisation then removed any global offset. Three noise regimes were run: JWST precision as delivered; Ariel Tier-1, the payload model's per-bin σ at the planet's Tier-1 transit count added as 500-1,000 Gaussian draws, giving a detection frequency per molecule; and Tier-2 and Tier-3 counts for context.

The procedure was validated on a spectrum whose answer is known. The benchmark's own best-fit model of WASP-39 b was binned directly onto the 104 bins, and separately sampled at the JWST wavelength grids of each mode with the per-point JWST errors and pushed through the conversion; the two differed by 22 ppm rms, 104 ppm at worst in a MIRI bin, and no verdict changed noise-free. A sensitivity matrix of 24 configurations (the individual reductions of each mode, two NIRISS pipelines, fitted and fixed limb darkening, offsets on and off, model and flat fill, with and without MIRI, PRISM alone, and the bins of Na, K, SO₂ and CO masked singly and together) measured how much the verdict depends on the reduction, which Roy-Perez et al. (46) showed must be shown rather than assumed. WASP-39 is a quiet G8V star and no stellar correction was applied.

Two population-level comparisons accompany the verdicts. For each real planet a clear H₂/He twin was rendered at the published abundances and bulk parameters; the twin's dispersion divided by the noise-corrected dispersion of the real spectrum over its measured bins is the planet's muting (1 for a spectrum as featured as its clear twin, larger when it is more muted), and the most muted atmosphere the population can draw at those bulk parameters (the coldest allowed temperature with the deepest allowed deck) says whether the real muting lies inside the simulator's prior. The same muting was computed for every planet of POP-I, rendered against its own clear twin. And the real panel was re-scored by the design retrained on a population in which a third of the training planets carried a haze of density log-uniform in 10⁶-3 × 10⁸ m⁻³, and in a second set a further sixth carried a grey deck at 5-500 Pa, up to two decades beyond the prior's 5 × 10² Pa limit; this is the test of whether a real error belongs to the recoverable class, not a proposed remedy.

### 2.7 Diagnostics, routing and the selection simulation

Four candidate diagnostics were chosen before the runs to target the two classes: (a) disagreement between the all-input, no-optical and AIRS-only variants of the design; (b) an optical-excess score, the three normalised photometric points against the normalised NIRSpec continuum as a z-score on the clean training set; (c) the consortium's own band metric (6), used as a ranking in its own right; and (d) predicted decidability, a regressor on bulk features trained on the Δχ² of Section 2.4. Each was scored by selective credit as in Section 2.3, by its AUROC for separating the two error classes, and, used as a router, by the accuracy and recall of the classifier when flagged planets are re-scored by the AIRS-only variant.

The mission-level measure is a selection simulation. The 939 POP-I planets with valid spectra receive 60 % Tier-2 slots (3), ranked by the mean present-probability of the three learners that give one (KNN, MLP and RFC; the one-vs-one SVC does not); the score is the number of planets carrying the molecule that are not selected, for CH₄, for H₂O and for either ("joint targets"). Strategies compared at the same budget: the classifier alone; the classifier with each existing rule (declined planets sent to the back of the queue); the classifier routed by each candidate; the band metric alone; and a retrieval of every planet, ranking by its present-posterior, with its cost stated. A triage variant re-scores by the retrieval only the fraction *f* of planets a rule ranks most suspect, for *f* from 10 to 70 %, and compares each rule to a random budget, to the measured Δχ² of Section 2.4 used as an ordering at its stated cost, and to the retrieval's own undecidability, which is the unattainable best ordering.

The mixed-population test applies the same budget where the retrieval shares the classifier's failure. Each of the 954 planets with a hazed spectrum is hazed with probability *q* (0.1, 0.2 or 0.5; 20 draws); the fraction *f* of planets a rule ranks most suspect (10 to 50 %) is re-scored by a fallback retrieval and the rest keep the classifier; the selection is the 60 % cut above. The fallback is the free-haze retrieval of Section 2.4, and the existing fallback, the clear-model retrieval, is run in its place for comparison. The orderings are random, the five existing rules, the optical-excess score, the input disagreement, and an oracle that knows which planets are hazed; the fixed strategies are the classifier alone, the optical score routed to the AIRS-only variant without any retrieval, and either retrieval of every planet. The metrics are targets lost at the selection, the pipeline's accuracy, and its recall on the hazed and on the clean targets. Expectations were committed before the population data (Table 10). [PENDING: the clean planets are scored by the clean retrieval of Section 2.4 until the free-haze retrieval of the clean population completes, due 31 October.]

### 2.8 Expectations committed before the runs

Expectations for every measurement were written to the repository before its run, so that surprises can be told from confirmations; those for the external test, the identifiability measurements and the diagnostics are scored in Section 3.7 (Table 10), and those for the stress test, which predate them, remain in the repository.

## 3. Results and Discussion

### 3.1 The re-implementation reproduced the published accuracies

![Figure 1](figures/note_fig1_reproduction.png)

**Figure 1.** The re-implementation against the published accuracies for the 48 combinations of classifier, molecule and abundance threshold. (a) Re-implemented against published accuracy; dashed line, equality. (b) Difference between the two for each combination, grouped by molecule, with marker shape giving the abundance threshold. Grey bands: ±5 points.

Across the 48 published accuracies, the re-implementation differed by +0.4 points on average, 1.4 in absolute value and at most 3.7 (Figure 1; Appendix A), with a correlation of 0.96. It reproduced the published patterns: accuracy rose with the abundance threshold for 15 of 16 classifier-molecule pairs (16 of 16 published), and water was the hardest molecule at 10⁻⁵ and 10⁻⁴ in both. At 10⁻⁴ the mean accuracies were 84.9 % (CH₄), 74.9 % (H₂O), 79.5 % (CO₂) and 84.4 % (NH₃), against 84.0, 74.8, 80.5 and 85.0 % published. Since a mistaken reading of the recipe missed by 10 points (Section 2.2), this agreement indicates that the re-implementation followed the published design. The reproduction was regenerated from scratch on a second machine and agreed with the committed values to 0.05 points on average.

### 3.2 The departures, and where the haze failure came from

![Figure 2](figures/note_fig2_consortium_map.png){width=5.6}

**Figure 2.** The classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Open circles: loss of the classifier as published; filled circles: loss after randomised training; black ticks: irreducible loss. Error bars: standard deviation across the four molecules.

**Table 3.** The classifier under each departure (threshold 10⁻⁴; mean over four classifiers and four molecules). Losses are in points below clean accuracy. The margin rule is thresholded to decline 10 % of held-out clean planets; margin AUROC (error) measures how well the margin ranks errors, and distance AUROC (shift) how well the Mahalanobis distance separates altered from clean spectra (0.5 is chance). Conformal sets were calibrated to 90 % on clean spectra.

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

The classifier lost little to a different opacity database (2.9 points), a different radiative-transfer code (3.5), added HCN and C₂H₂ (0.1) or a cloud deck at 10³ Pa inside its training prior (2.9; Table 3, Figure 2). It lost far more to haze (17.7 points at 3 × 10⁷ m⁻³), to spots (10.8 at 20 % coverage), to a cloud deck at 10² Pa (9.8), to spots and haze together (18.8) and to extra noise (9.3-13.5). Retrained with the departure present, the design recovered 87 % of the haze loss and 65-80 % of the spot, deck and compound losses, but only 13-21 % of the noise losses: by the retraining measure the aerosol losses were already of a different kind from the noise losses, and Section 3.4 gives that distinction a classifier-free foundation.

![Figure 3](figures/note_fig3_haze.png)

**Figure 3.** (a) A clear-atmosphere test planet with CH₄ above 10⁻⁴ at Tier-3 binning, without and with haze at 3 × 10⁷ m⁻³ (noise-free); shading marks the three photometric points. (b) Detection rate, the share of planets containing a molecule for which it was reported, against haze density for the published input, the input without the photometric points and the AIRS bins only.

Under haze at 3 × 10⁷ m⁻³, the multilayer-perceptron, random-forest and support-vector classifiers reported CH₄ on 0-5 % of planets (27-33 % on clean spectra; 38 % truly present) and their H₂O detection rate fell to 0 %, while the multilayer perceptron's mean confidence rose from 0.93 to 1.00. Yet the molecular bands were intact: the haze added a median 1,094 ppm to the three photometric points, 140 ppm across NIRSpec and 5 ppm across AIRS, leaving the AIRS band amplitudes unchanged (Figure 3a). Four versions of the design that differed only in their input located the cause (Table 4). Normalising with the bins above 1.1 µm alone changed nothing, so the failure is in the points themselves and not in the normalisation. Removing the photometric points cut the haze loss from 17.7 to 5.6 points and the spot loss from 10.8 to 3.2; using only the AIRS bins cut the haze loss to 0.5 and removed the spot loss; transplanting only the three hazy photometric points into clean spectra reproduced the failure (CH₄ detection 1.6 %), and restoring them in hazy spectra removed it (72 %). The photometric points do carry information, since removing them cost 3.8 points on clean spectra and using AIRS alone 7.4, so the AIRS-only variant is a fallback with a price, not a replacement.

**Table 4.** The design with four inputs, each retrained on the same spectra (threshold 10⁻⁴; mean over four classifiers and four molecules). Cells: accuracy (%), with the detection rate (%) in parentheses. Haze densities in m⁻³.

<!-- table:variants -->
| Classifier input | Clean | Haze 2 × 10⁶ | Haze 3 × 10⁷ | Haze 2.4 × 10⁸ | Spots 20 % | Spots + haze |
| :-- | --: | --: | --: | --: | --: | --: |
| All 104 bins (published) | 81.1 (63) | 69.0 (23) | 63.4 (6) | 62.3 (3) | 70.3 (26) | 62.2 (3) |
| All bins, normalized with the bins above 1.1 µm | 80.9 (61) | 68.6 (21) | 63.3 (6) | 62.3 (3) | 69.9 (25) | 62.5 (3) |
| Photometric points removed | 77.3 (53) | 76.8 (51) | 71.7 (33) | 65.1 (14) | 74.1 (44) | 69.5 (27) |
| AIRS bins only (above 1.95 µm) | 73.7 (44) | 73.6 (44) | 73.2 (44) | 70.8 (41) | 73.9 (45) | 73.4 (45) |
<!-- /table:variants -->

None of the five decline rules recognised the failure. Every rule's credit against the clean selective baseline was negative under haze (−17.5 to −25.8 points), spots and the high deck; the margin rule declined 7.4 % of clean planets but 0.9 % of hazy ones; distance scores separated spectra hazed at 3 × 10⁷ m⁻³ from clean ones (AUROC 0.90-0.94) but did not rank errors within them (0.44-0.51), acting as a population alarm rather than a filter; conformal coverage fell to 73.7 % under haze. This is the result of Jaeger et al. (16) in a physical setting: confidence does not detect failure under a shift, and the shift detectors that fire do not say which answers are wrong.

### 3.3 A third to a half of the labels are undecidable at Tier-1 precision

The Δχ² test and the retrieval posterior agreed in rank: Spearman correlation between Δχ² and logit p_true of 0.86 (CH₄), 0.80 (H₂O), 0.88 (CO₂) and 0.89 (NH₃) over the 955 planets. The test's scale, however, was wrong by an order of magnitude. The posterior crosses 0.95 at Δχ² of 0.45, 0.98, 0.90 and 0.56, not at the nested-model value 3.84 or the 3σ value 9, because the test represents the opposite class by its single best-fitting point while the retrieval integrates over that class's volume. The calibrated thresholds are used everywhere below, and the test is a check on the retrieval rather than a substitute: at 207 core-hours against the retrieval's 365 on this population, it is not the cheap method it was meant to be.

**Table 5.** Prevalence of undecidable labels on the 955 POP-I planets at Tier-1 precision, from the posterior of the retrieval of every planet (the one misled label of 3,820 is counted with them), with the retrieval's accuracy (the ceiling the spectra allow under the population's prior) and the classifier's clean accuracy on the same 955 spectra from the run of Section 3.6, which differs from the reproduction run of Section 3.1 in its planet set and training draw.

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | --: | --: | --: | --: |
| Undecidable, p_true < 0.95 | 32 % | 54 % | 44 % | 32 % |
| Undecidable, p_true < 0.84 | 17 % | 28 % | 27 % | 19 % |
| Retrieval accuracy (ceiling) | 93.7 | 91.6 | 93.8 | 93.9 |
| Classifier, clean | 84.3 | 75.3 | 80.7 | 84.6 |
| Gap | 9 | 16 | 13 | 9 |

A third of the CH₄ and NH₃ labels, 44 % of the CO₂ labels and 54 % of the H₂O labels are undecidable at the 0.95 level (Table 5). Only 17 % of the planets have all four labels decidable, 3 % have none, and a third have both CH₄ and H₂O decidable. Decidability is set first by the label's distance from the 10⁻⁴ boundary and second by the planet's noise: labels within half a decade of the boundary are undecidable in 89-92 % of cases, labels one to two decades away in 12-36 %, and absent labels two to three decades below the boundary in 1-8 % for CH₄, CO₂ and NH₃ but 31 % for H₂O, the planets whose noise leaves water unconstrained at any abundance. The one retrieved planet of the consortium study, CH₄ at 10⁻³·¹ with a 48 % posterior at Tier 1 (6), is a typical member of the undecidable class.

The retrieval scores 92-94 % on every molecule, is confidently wrong on one label of 3,820, and is slightly under-confident. The classifier sits 9-16 points below it, 12 on average, so on clean spectra it is well short of the information limit. Among the classifier's clean errors, only 19-25 % are on labels the spectrum decides; the remaining three quarters to four fifths are on undecidable labels, and effectively none (0-0.8 %) is on a label the spectrum decides the other way. On decidable labels the learners score 92-98 % for CH₄, CO₂ and NH₃ and 76-93 % for H₂O; on undecidable ones 58-70 %, a few points above the majority class. Recall of present labels is 82-84 % where the label is decidable and 35-49 % where it is not: the classifier's default on an uninformative spectrum is "absent", which is the prior, and Section 3.5 meets that default on real spectra.

### 3.4 The two error classes separate across the departures

Under haze at 3 × 10⁷ m⁻³, with the haze in the counterfactual model, the labels are about as decidable as on the clean spectra by the calibrated Δχ² test (Figure 4a): 74 % of CH₄ labels against 71 % clean, 38 against 44 % for H₂O, 76 against 59 % for CO₂ and 72 against 71 % for NH₃. The haze costs water a little, because its bands sit under the near-infrared slope the haze adds, and makes CO₂ more decidable, because with the haze density known its slope pins the scale height and removes part of the temperature-abundance degeneracy. The information the labels need survives the haze; what the classifier loses on hazy spectra, it loses by its model, not by the spectrum. Under 20 % unocculted spots, with the spots in the counterfactual, the decidable fractions move by 0.3 to 2.5 points and fewer than 1 % of labels in any molecule lose decidability: the contamination factor deepens the transit by a few per cent at every wavelength, and a test that knows it loses nothing. A grey deck at 10² Pa, above the prior's limit of 5 × 10² Pa, is different. With the deck at its true pressure in the counterfactual, the decidable fraction falls to 70 % for CH₄, 28 % for H₂O, 53 % for CO₂ and 63 % for NH₃, and the loss falls almost entirely on absent labels (H₂O absent labels decidable 226 → 93 of 591, present 194 → 173 of 364): the deck hides the deep atmosphere where a small abundance would have shown, so the floor on an absence rises. That is information the atmosphere removes, as noise is information the instrument removes.

**Table 6.** Error classes of the classifier under each departure (threshold 10⁻⁴; mean over the four learners; the clean column from the posterior, the others from the departure-aware Δχ² at the calibrated thresholds). "Introduced" errors are those the learner does not make on the clean spectrum; "shared" errors it makes on both.

| | CH₄ | H₂O | CO₂ | NH₃ |
| :-- | --: | --: | --: | --: |
| Clean: errors recoverable | 25 % | 20 % | 20 % | 19 % |
| Haze: accuracy, clean → haze | 84.3 → 66.6 | 75.3 → 62.1 | 80.7 → 63.6 | 84.6 → 61.6 |
| Haze: recall on decidable present labels | 83 → 14 % | 65 → 1 % | 82 → 13 % | 84 → 2 % |
| Haze: errors introduced, per learner | 218 | 170 | 218 | 250 |
| Haze: introduced errors recoverable | 87 % | 64 % | 81 % | 79 % |
| Haze: shared errors recoverable | 51 % | 29 % | 53 % | 40 % |
| Haze: all errors recoverable | 75 % | 45 % | 70 % | 67 % |
| Spots 20 %: accuracy, clean → spots | 84.3 → 74.6 | 75.3 → 67.8 | 80.7 → 70.3 | 84.6 → 68.8 |
| Spots 20 %: recall on decidable present labels | 83 → 44 % | 65 → 30 % | 82 → 45 % | 84 → 28 % |
| Spots 20 %: introduced errors recoverable | 79 % | 65 % | 57 % | 80 % |
| Spots 20 %: all errors recoverable | 66 % | 47 % | 43 % | 62 % |
| Deck 10² Pa: undecidable labels | 30 % | 72 % | 47 % | 37 % |
| Deck 10² Pa: accuracy, clean → deck | 84.3 → 75.1 | 75.3 → 67.3 | 80.7 → 73.5 | 84.6 → 69.8 |
| Deck 10² Pa: recall on decidable present labels | 83 → 45 % | 65 → 36 % | 82 → 58 % | 84 → 34 % |
| Deck 10² Pa: introduced errors recoverable | 82 % | 51 % | 52 % | 76 % |
| Deck 10² Pa: all errors recoverable | 68 % | 36 % | 36 % | 61 % |
| Noise × 2: undecidable labels | 50 % | 78 % | 72 % | 51 % |
| Noise × 2: introduced errors recoverable | 61 % | 32 % | 23 % | 53 % |
| Noise × 3: undecidable labels | 65 % | 87 % | 86 % | 67 % |
| Noise × 3: introduced errors recoverable | 56 % | 23 % | 10 % | 41 % |
| Noise × 3: all errors recoverable | 41 % | 13 % | 8 % | 28 % |

Table 6 is the two-class split measured rather than argued. The errors the haze introduces are four fifths recoverable for CH₄, CO₂ and NH₃ and two thirds for H₂O, while the errors the learners share with their clean run are half or less recoverable. The haze does not take the methane out of the spectrum; it takes the classifier's recall of decidable present labels from 83 % to 14 %. The spots are the same mechanism at lower amplitude: they cost 8-16 points of accuracy and half to two thirds of that recall while leaving the labels as decidable as before, and the errors they introduce are 57-80 % recoverable. The deck sits between haze and noise: on CH₄ and NH₃, whose bands sit high, its errors are four fifths recoverable, but on H₂O and CO₂ only half, because the deck has made undecidable a sixth of the H₂O labels and a tenth of the CO₂ labels the clean spectrum could decide, and the classifier's errors on those are the atmosphere's, not its own. Under threefold noise the undecidable fraction roughly doubles, to 65-87 %, and the added errors are mostly irrecoverable: 10-56 % recoverable, against 64-87 % for haze. A haze error is within the spectrum's reach and a noise error is not, a deep deck's errors are partly each, and the same test tells them apart label by label.

The existing method shares the recoverable failure. A nested-sampling retrieval with the training forward model, clear H₂/He with a grey deck, was run on the hazed spectra of all 954 planets that have one (500 core-hours). It failed to complete on 18 of them, 14 sampler aborts on a degenerate likelihood and 4 at a three-hour cap, against none of 955 on the clean spectra. On the 936 it completed, it called every present label absent: recall of present labels fell from 81-88 % to below 1 % for all four molecules, the posterior of the true class was below 0.05 on 99 % of present labels, and every abundance was driven 3.5-4 dex toward the prior floor, so its accuracies of 59-63 % are the absent fractions. Of the 219 such planets among the 936 with CH₄ present and decidable on the clean spectrum, it recovered two. The misfit is visible in the evidence (median log Z −12 on clean spectra, −252 on hazed), so an analyst looking at the fit would see it, but a pipeline that reads only the posterior would not. The departure is largest where the classifier's was: with the free offset removed, the haze-minus-clean difference is 18σ per bin in the three optical points (median over planets), 4.6σ over NIRSpec and 1.8σ over AIRS.

One free parameter repairs it. With the haze density added to the forward model, the same retrieval on the same 954 spectra completed on all of them (635 core-hours) and recovered the density on every planet (median log₁₀ n = 7.47 against 7.48 true; 68 % interval 0.11 dex wide, holding the true value on 95 % of planets; no posterior at the haze-free end). It recovered 222 of the 224 planets with CH₄ present and decidable on the clean spectrum (median posterior of the true class 0.996) and kept 75 % of them decidable; its accuracy stayed within 5 points of the clean retrieval on every molecule (CH₄ 93.7 → 92.3 %, H₂O 91.6 → 87.3 %, CO₂ 93.8 → 93.7 %, NH₃ 93.9 → 91.4 %), its recall of present labels fell from 88 to 82 % for CH₄ and from 82 to 69 % for H₂O, it was confidently wrong on 0.0-0.3 % of labels, and it fit with the clean evidence (median log Z −16; four planets, among them the super-puff Kepler-79 d, fit badly with the radius held fixed and still carried the right labels but one). Figure 4c shows the two retrievals side by side. The posterior says what the Δχ² test said: the haze removes nothing the labels need, and the clear-model retrieval's zero recall on the same spectra is entirely the model's.

![Figure 4](figures/fig4_identifiability.png)

**Figure 4.** The two error classes. (a) Fraction of labels decidable at Tier-1 precision by the calibrated Δχ² test, per molecule, on clean spectra and under each departure with the departure in the counterfactual model; the noise cases are the clean Δχ² divided by k². (b) Share of the errors each departure introduces that are recoverable (Table 6); the clean bar is the posterior-based share of all clean errors. (c) Posterior probability of the true CH₄ class from the clear-model and the free-haze retrievals of the same hazed spectra, against the clean-spectrum posterior, on a logit scale clipped at 0.001 and 0.999; the grey lines mark 0.05 and 0.95.

### 3.5 Real JWST spectra: the same two classes

**WASP-39 b.** At JWST precision with all 104 inputs, three of the four learners called H₂O present and all four called CH₄, CO₂ and NH₃ absent at 10⁻⁴, which matches the published composition (39, 40, 45); CO₂ at 10⁻⁵, where its published abundance of about 10⁻⁴·⁵ puts it, was present for two to three learners; NH₃ was absent in all 24 configurations of the sensitivity matrix and with each of three MIRI reductions; and the consortium's own band metric agreed with the majority. The multilayer perceptron called H₂O absent with a score of 0.00, in 23 of the 24 configurations, and on the benchmark's own best-fit model binned directly onto the 104 bins with no conversion at all, so the miss is not the conversion's. It is the shape of a real hot Saturn: the clear H₂/He twin at the published abundances is called H₂O-present by all four learners (perceptron score 1.00), as are twins with grey decks down to 10 Pa, but the real spectrum's dispersion over the measured bins is 324 ppm against 1,190 ppm for the clear twin, a muting of 3.7, whereas the most muted atmosphere the population can draw at WASP-39 b's bulk parameters reaches 2.0. The real spectrum is nevertheless inside the training distribution by every input-space rule (k-NN distance 6.2 against the decline threshold 11.25), because those rules work on normalised spectra and normalisation removes amplitude. Under Ariel Tier-1 noise at WASP-39 b's one transit, H₂O is detected in 58 % of draws with all inputs and 86 % without the three optical points, while the clear twin is detected in 100 %; the optical points decide the Tier-1 outcome on a real planet, which is Section 3.2's mechanism in a weaker form.

**The panel.** Over the 35 panel labels the literature decides, the all-input classifier was right on 22 (63 %), below the share of absent labels (22 of 35), and reproduced none of the thirteen present labels outright. Without the optical points it was right on 25 of 36, and with the AIRS channels alone on 29 of 37 (78 %). Three planets with only AIRS coverage (WASP-107 b, HAT-P-26 b, GJ 3470 b) were called H₂O-absent although H₂O is present at 6-21σ; their twins through the same coverage mask are also missed and the same twins with full coverage are found by all four learners, so this is a coverage artefact, not a classifier error, and these planets are scored only on CH₄ and CO₂; the fourth AIRS-only planet, HD 189733 b, has an ambiguous H₂O label and was called absent, correctly, on the other three. No NH₃ false positive occurred on any panel planet.

**Table 7.** The three methane-bearing sub-Neptunes in six reductions. Verdict is the majority of the four learners at 10⁻⁴; the distance rule declines above 11.25; the muting is the clear twin's dispersion over the real one; the deck needed is the grey-cloud pressure at which the twin matches the real muting, where the population's prior stops at 5 × 10² Pa; the last three columns are the all-input verdicts of the design retrained with hazed training planets and with hazed and deep-deck training planets, and the verdict of the AIRS-only variant, which is the same under every training set.

| Reduction | All inputs | No optical | AIRS only | Distance | Muting | Deck needed (Pa) | Retrained, haze | Retrained, haze + deck | AIRS only, retrained |
| :-- | :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- |
| TOI-270 d (49) | absent | absent | split | 12.70 | 0.7 | in prior | absent | absent | split |
| TOI-270 d (48) | absent | absent | present | 11.01 | 1.0 | in prior | absent | absent | present |
| K2-18 b (47) | absent | absent | present | 11.25 | 0.7 | in prior | absent | split | present |
| K2-18 b (50) | absent | split | present | 11.54 | 1.7 | 238 | split | present | present |
| K2-18 b (51) | absent | absent | present | 11.69 | 2.0 | 117 | present | split | present |
| TOI-1231 b (52) | split | absent | present | 11.21 | 1.6 | 47 | present | present | present |

**The methane planets.** With all inputs the classifier called CH₄ absent on five of the six reductions of TOI-270 d, K2-18 b and TOI-1231 b, whose methane detections are among the most secure in the JWST literature (47-52), and split on the sixth (Table 7), with Tier-1 detection frequencies of 7-16 %. With the AIRS channels alone it found CH₄ on five of six. The clear twins of these planets, through the same coverage mask and scattered with the real per-bin JWST errors, are detected in 68-100 % of draws and never flagged, so the misses are not the noise, the coverage or the classifier's in-simulator skill; they are the shape of the real spectra, and the information is in the spectrum because the AIRS-only learner finds it. All three planets have clouds or hazes in their retrievals, and the three optical points carry that shape into the learners. These are recoverable errors on real planets, in the operational sense that another learner reading the same spectrum finds the label; no retrieval of the real spectra was run, so the posterior definition of Section 2.5 is not applied to them. The distance rule flags three of six and puts the other three at the 90th percentile of clean distances.

**Muting.** Every warm giant on the panel with a published abundance set (HAT-P-26 b, WASP-107 b, WASP-39 b, GJ 3470 b, WASP-166 b) is muted by a factor of 3 to 5 against its clear twin (Figure 5), and by 1.5 to 2.2 beyond the coldest, cloudiest atmosphere the population allows at its mass, radius and irradiation; the same holds over the AIRS bins alone, so the muting is in the molecular bands and not in a haze slope. Across the 955 planets of POP-I, the population's own clouds mute the median planet by 1 %, the 90th percentile by 20 %, the 99th by a factor of 2.5 and the most muted by 6.3, at the prior's 5 × 10² Pa limit. The simulator population can therefore reach the giants' muting, but in fewer than one planet in a hundred and only at that limit, whereas five of five real giants sit there, needing what the population does not contain: metallicities of tens to hundreds of times solar, hazes or patchy clouds, which is what their JWST retrievals report. The three methane sub-Neptunes, by contrast, are muted by only 0.7-2.0, inside the range the population spans, although three of their six reductions need a deck above the prior's limit at their own bulk parameters (Table 7); their misses have the different cause above. Both results say that the real spectra carry smaller features, relative to their continua, than the clear-grid training spectra, and the classifier's operating point is tuned to the latter, which is why its default on real spectra is "absent".

![Figure 5](figures/fig5_muting.png)

**Figure 5.** Muting of the real spectra against the simulator population. (a) For each real planet, the dispersion of its clear H₂/He twin over the measured bins divided by the noise-corrected dispersion of the real spectrum (filled), and the same for the most muted atmosphere the population prior allows at the planet's bulk parameters (open); the five warm giants with published abundances are in orange, the shaded band is the 5th to 95th percentile of the population's own cloud muting and the dotted line its most muted planet. (b) Cumulative distribution of that muting over the 955 POP-I planets, each rendered with its own cloud deck against its clear twin, with the five giants marked.

**The error class of the real misses.** Retraining the design on a population in which a third of the training planets carried haze recovered two thirds of the CH₄ haze loss on the simulated population (66 → 79 % against 85 % clean), half of the CO₂ and NH₃ loss and a quarter of the H₂O loss, left the noise × 3 column unchanged, and cost 1-2 points on clean spectra. On the real panel it raised the all-input score from 22 of 35 to 26 of 36, and with the deep decks to 27 of 36. The three methane reductions whose muting lies beyond the simulator's aerosol prior (K2-18 b in both reanalyses and TOI-1231 b, needing decks at 47-238 Pa) are the three on which it moved, two to present and one to split; the three within the prior (TOI-270 d in both reductions and the primary K2-18 b) it did not, apart from a split on K2-18 b under the deep decks (Table 7). Where the failure on a real planet is the training prior, extending the prior repairs it, planet by planet. The AIRS-only variant finds five of six at no retraining, so routing remains the cheaper remedy; the retraining is the test of the class, not the contribution.

### 3.6 Recognising the two classes at deployment

**The retrieval is the selection method on clean spectra.** In the selection simulation at 60 % Tier-2 slots, ranking by the retrieval's present-posterior loses 0 of 359 CH₄-bearing targets (oracle 0; classifier 20), 2 of 364 H₂O targets (oracle 0; classifier 51) and 39 joint targets (oracle 3; classifier 95). On clean simulated spectra the existing method, a retrieval of every planet, is the selection oracle to within noise, at 365 core-hours for the population, two days on a laptop; the classifier and every diagnostic on it are dominated. No diagnostic can claim to improve selection against a retrieval on clean spectra, so the question becomes what a diagnostic buys at a fixed retrieval budget, and where the retrieval shares the classifier's failure.

**On clean spectra no cheap rule routes a retrieval budget better than chance.** When the fraction *f* of planets a rule ranks most suspect is re-scored by the retrieval and the rest keep the classifier (Table 8), nothing attainable beats a random budget by more than four CH₄ targets at any *f*, and the confidence-based rules (margin, ensemble, the input disagreement and the band metric) are no better than random up to *f* = 50 %. They send the retrieval to the planets the classifier is unsure about, which are the undecidable labels, where the retrieval cannot help either; the retrieval pays off on the planets the classifier is confidently wrong about, and no rule finds those. Predicting decidability from bulk features reached an AUROC of only 0.47-0.65.

**Table 8.** CH₄-bearing targets lost at the 60 % selection when the fraction *f* of planets a rule ranks most suspect is re-scored by the retrieval (clean spectra; 359 targets).

| Ordering | f = 10 % | 20 % | 30 % | 50 % | 70 % |
| :-- | --: | --: | --: | --: | --: |
| Random | 16 | 12 | 7 | 3 | 2 |
| Retrieval's own undecidability (unattainable) | 7 | 7 | 4 | 1 | 0 |
| Measured Δχ² (costs 57 % of the retrieval) | 13 | 8 | 4 | 2 | 0 |
| k-NN density | 13 | 8 | 5 | 3 | 1 |
| Mahalanobis | 13 | 9 | 7 | 3 | 1 |
| Predicted decidability | 16 | 13 | 9 | 2 | 1 |
| Margin | 17 | 13 | 9 | 6 | 2 |
| Ensemble | 17 | 15 | 11 | 3 | 1 |
| Input disagreement | 17 | 14 | 12 | 6 | 1 |
| Optical excess | 15 | 12 | 8 | 5 | 2 |
| Band metric | 19 | 15 | 13 | 8 | 2 |

**Routed to a retrieval that models the departure, a shift flag allocates a 20 % budget as well as an oracle.** The mixed-population simulation (Section 2.7; Table 9) lets the budget go to the free-haze retrieval. With a fifth of the 954 planets hazed, the classifier alone loses 29 of 359 CH₄ targets and finds 3 % of the hazed ones; the existing method, a clear-model retrieval of every planet, loses 69 and finds none. Re-scoring the 20 % of planets the optical score ranks most suspect by the free-haze retrieval loses 13 and finds 80 % of the hazed targets, the same as an oracle that knows which planets are hazed (12; 81 %), against 20 and 17 % for a random budget and 20-22 and 3 % for the margin and ensemble rules. The distance rules used as routers come close (Mahalanobis 13 lost, 64 % found; PCA 14, 58 %; k-NN 14, 54 %), because they also detect the shift, less selectively: the top 20 % by the optical score holds 98 % of the hazed and 1 % of the clean planets, by Mahalanobis 73 and 7 %. The confidence rules hold none of the hazed planets, because the classifier is confident on them. Routing the same 20 % to the existing clear-model retrieval loses 71 targets, worse than no retrieval at all. For H₂O the optical ordering loses 40 against 73 for the classifier alone and 40 for the oracle. At a hazed fraction of 10 % the ordering again matches the oracle (12 against 12); at 50 % a 20 % budget cannot cover the hazed half, and the ordering reaches the oracle at *f* = 50 % (5 against 5). The free-haze retrieval of every planet loses no CH₄ target and two H₂O targets at every hazed fraction, the clean retrieval's own numbers, so the limit on the routed pipeline is the budget, not the fallback. The claim is therefore about the use, not the score: a flag that detects the departure, routed to a retrieval that models it, turns a 20 % budget into the oracle's selection, while the same flag routed to the existing retrieval is worse than nothing and the confidence rules cannot find the planets at all. Every expectation committed for this test before the population data was met, except that two of the three distance rules fell more than 20 points behind the optical score (Table 10).

**Table 9.** Mixed population with a fifth of the planets hazed: CH₄-bearing targets lost at the 60 % selection (of 359; mean of 20 draws) when the fraction *f* of planets a rule ranks most suspect is re-scored by the free-haze retrieval, and the recall on the hazed targets at *f* = 20 %. Values are means over the draws, rounded. The last four rows use the existing clear-model retrieval, routed or on every planet, or no retrieval.

| Ordering of the budget | f = 10 % | 20 % | 30 % | 50 % | Hazed recall at 20 % |
| :-- | --: | --: | --: | --: | --: |
| Classifier alone (no budget) | 29 | 29 | 29 | 29 | 3 % |
| Random | 25 | 20 | 16 | 9 | 17 % |
| Optical excess | 20 | 13 | 8 | 4 | 80 % |
| Oracle (knows the hazed) | 21 | 12 | 9 | 5 | 81 % |
| Mahalanobis | 21 | 13 | 8 | 3 | 64 % |
| PCA reconstruction | 21 | 14 | 10 | 4 | 58 % |
| k-NN density | 22 | 14 | 7 | 3 | 54 % |
| Input disagreement | 24 | 18 | 14 | 7 | 37 % |
| Band metric | 25 | 21 | 18 | 10 | 13 % |
| Margin | 25 | 20 | 16 | 12 | 3 % |
| Ensemble | 27 | 22 | 18 | 13 | 3 % |
| Free-haze retrieval of every planet | 0 | 0 | 0 | 0 | 82 % |
| Optical excess → clear-model retrieval | 47 | 71 | 70 | 70 | 0 % |
| Random → clear-model retrieval | 30 | 30 | 32 | 38 | 2 % |
| Clear-model retrieval of every planet (existing method) | 69 | 69 | 69 | 69 | 0 % |
| Optical excess → AIRS-only variant, no retrieval | 20 | 20 | 20 | 20 | 58 % |

**Without a retrieval, the same flag routed to an input without the weak point recovers a third of the targets haze loses.** No candidate had positive selective credit under haze or spots at the 10 % clean decline rate (the best, input disagreement, −6.8 and −6.3; the five existing rules −17.8 to −25.7 under haze in this run), and the reason is instructive: the optical-excess score flags 100 % of hazy and 66 % of spotted planets while flagging 10 % of clean, 1.5 % of clouded and 2-3 % of noisier ones, so declining what it flags empties the hazy population, and credit is the wrong measure of a shift detector. Used as a router, with flagged planets re-scored by the AIRS-only variant, it raised CH₄ accuracy under haze from 66.6 to 78.3 % and CH₄ recall from 11.6 to 53.7 % (compound departure 5.4 to 50.3 %; spots 35.5 to 53.1 %), while under clean, cloud and noise the routed classifier stayed within 0.5 points of the frozen one, because almost nothing was flagged. In the selection simulation under haze the classifier alone loses 38 of 359 CH₄ targets; routed, 26; the AIRS-only variant used everywhere, 27; the best existing rule (k-NN, declined to the back) 42; the margin and ensemble rules 105 and 112, worse than the band metric alone (103). Under clean spectra the routed classifier loses 19 against 20 and the AIRS-only variant 27; under noise × 2, 37 against 37 and 57. The routing buys the AIRS-only variant's immunity to haze without paying its clean and noisy cost, and is never worse than the classifier; the remaining misses are the planets the AIRS-only learner also misses. For H₂O no candidate helps, because the water loss under haze is not routed through the optical points to the same degree. On the real planets the input-disagreement candidate flags the CH₄ cell on four of the six methane reductions; the optical-excess score, being molecule-independent, flags HAT-P-18 b (the spotted host) and one K2-18 b reanalysis and says nothing about the methane misses.

The picture for a mission pipeline is then specific. On clean spectra, retrieve every planet; the cost is two days. Where the retrieval's forward model is wrong in the way haze makes it wrong, the retrieval fails more confidently than the classifier and its evidence, not its posterior, is the warning; a one-number score on the optical inputs finds the affected planets, and a retrieval budget sent to them with the haze density free selects as an oracle would, while without a budget the same score routes them to an input that does not see the departure, at no retraining. And for the third to half of labels the spectra do not decide, nothing at Tier 1 helps, and those planets should reach Tier 2 as undecided rather than dismissed.

### 3.7 Expected and surprising results

**Table 10.** Expectations committed before the runs of Sections 3.3 to 3.6, and outcomes; those for the stress test of Section 3.2 are in the repository.

| Expectation | Outcome |
| :-- | :-- |
| WASP-39 b: all four learners report H₂O present, CH₄ and NH₃ absent at 10⁻⁴ | Not met for H₂O (three of four); met for CH₄ and NH₃ |
| WASP-39 b under Tier-1 noise: H₂O detected in > 80 % of draws, CH₄ and NH₃ in < 20 % | H₂O not met (58 %; two learners of four reach it); CH₄ and NH₃ met |
| The optical points change a WASP-39 b verdict at JWST precision | Not met at 10⁻⁴; met at 10⁻³ (all four absent with the points, two present without) and in the Tier-1 frequencies (58 against 86 %) |
| H₂O and CH₄ verdicts the same for every reduction; CO₂ not | Met |
| The round trip changes no verdict | Met for verdicts; the perceptron's H₂O score moved by 0.21 |
| Panel: ≥ 75 % of scored labels right | Not met with all inputs (63 %); met with the AIRS channels alone (78 %) |
| The no-optical or AIRS-only variant recovers the methane misses | Met for AIRS-only (five of six), not for no-optical |
| The distance rule flags the methane planets | Met for three of six; the other three at the 90th percentile of clean distances |
| No NH₃ false positive on the panel | Met |
| Δχ² and posterior agree, Spearman ≥ 0.7 | Met (0.80-0.89) |
| Undecidable fraction at Tier 1: 10-30 % (CH₄, CO₂), 20-40 % (H₂O, NH₃) | Met at the 0.84 level for all four; at 0.95 met for NH₃ only (CH₄ 32 %, CO₂ 44 %, H₂O 54 %) |
| The ceiling is ≥ 8 points above the classifier | Met (9-16 points) |
| Under haze ≥ 70 % of the classifier's errors recoverable; under noise × 3 ≤ 30 % | Haze met for CH₄ and CO₂ on all errors, for CH₄, CO₂, NH₃ on introduced errors; noise met for H₂O, CO₂, NH₃, missed by CH₄ (41 %) |
| A candidate diagnostic has positive selective credit under haze | Not met; as a router it recovers a third of the lost targets |
| Classifier + diagnostic loses fewer than half the CH₄ targets the classifier loses under haze | Not met (26 against 38, a third) |
| Predicted decidability separates the classes with AUROC ≥ 0.75 | Not met (0.47-0.65) |
| Mixed population, fifth hazed, 20 % budget to the free-haze retrieval: the optical ordering raises CH₄ recall on hazed targets from < 15 % to > 70 % and halves the targets lost | Met (3 → 80 %; 29 → 13) |
| It beats the confidence rules by > 40 points of hazed recall and random by > 30 | Met (77 over margin and ensemble, 67 over the band metric; 63 over random) |
| The distance rules as routers come within 20 points; the optical score is the most selective flag (≥ 99 % of hazed, ≤ 2 % of clean in its top 20 %) | Met for Mahalanobis (16 points), not for PCA (22) and k-NN (26); selectivity met (98 / 1 %) |
| Routing the clear-model retrieval is worse than no retrieval | Met (71 against 29 targets lost) |
| The free-haze retrieval of every planet loses at most the clean retrieval's targets + 5 | Met on the hazed planets (0 CH₄, 2 H₂O, as the clean retrieval); the clean planets await the free-haze run on clean spectra |
| The ranking of strategies is stable across lost-target weights | Met trivially: at a fixed slot count, slots wasted equal targets lost plus a constant |

Two results were not among the expectations. The scale of the Δχ² test was wrong by an order of magnitude against the posterior, which the calibration was designed to catch, and the uncalibrated prevalence first computed (78-88 % undecidable at 3σ) is not reported as a result. And the real warm giants turned out to lie outside the simulator population in feature amplitude while inside it in every bulk parameter and by every distance rule, which no expectation anticipated.

### 3.8 Limitations

The prevalence of undecidable labels and the error-class shares are statements under the consortium's test prior at the planets' Tier-1 transit counts, with the payload noise model of Section 2.2; the true joint distribution of Ariel's targets is unknown, and the muting census of Section 3.5 is the measurement of how far eleven real planets sit from that prior, not a replacement for it. The retrieval holds the planetary radius fixed with only a small depth offset free, and its forward model is the training model, so its ceiling is the ceiling under that model; a retrieval with a free radius and a full aerosol model is what a consortium would run on a flagged planet, and the free-haze retrieval, with one aerosol parameter free, is the nearest comparison here. The departure-aware error classes for spots and the deep deck rest on the calibrated Δχ² test rather than on a posterior; for haze both measures were run. The error class of the real misses rests on a second learner finding the label, not on a retrieval of the real spectra. The results concern the published design as re-implemented, with the deviations of Section 2.2, not the original trained models. Labels for the real planets come from the literature, and the ambiguous ones are not scored; coverage of the Ariel grid by JWST is partial for every panel planet, and the filled bins are reported. Haze strengths were anchored to retrieved hazes through an order-of-magnitude conversion that depends on the assumed pressure level. A second classifier, built for Ariel's carbon-to-oxygen objective, is summarised in Appendix B; it showed a nearly opposite profile at full resolution, so one classifier's error classes do not stand in for another's.

## 4. Conclusion

A classifier proposed for Ariel's Tier-1 survey was re-implemented from its published description and reproduced the published accuracies within 3.7 points. A nested-sampling retrieval of every one of its 955 test planets measured which labels the spectra decide at Tier-1 precision. Under the consortium's own test prior, a third to a half are undecidable, three quarters of the classifier's clean errors fall on those labels and are irrecoverable by any method, and the retrieval itself, at two days of computing for the population, is the Tier-2 selection oracle on clean spectra. The departures separate into the two classes the mission needs to tell apart: haze leaves the labels decidable and the errors it introduces are four fifths recoverable, while threefold noise doubles the undecidable fraction and only 10-56 % of its errors are recoverable.

On real JWST spectra the same two classes appear. The classifier reproduced the published composition of WASP-39 b, but missed methane on five of six reductions of the three methane-bearing sub-Neptunes while the same design restricted to its AIRS inputs found it on five of six, a recoverable error that enters through the optical inputs and, for the planets muted beyond the training prior, is repaired by extending that prior, as retraining with hazes showed. The real warm giants are muted three to five times beyond their clear twins, further than 99 % of the simulator population can go, a limit of the prior that no normalised-input rule can see. A retrieval with the training forward model fails on hazy spectra more confidently than the classifier, so the existing method is no fallback for the recoverable class unless it models the departure; with the haze density as one free parameter it recovers the labels the clean spectra decide.

The remedy's value depends on where it is applied. On clean spectra no cheap rule allocates a partial retrieval budget better than chance. On a mixed population, a one-number score on the optical inputs routed to a retrieval with the haze density free allocates a 20 % budget as well as an oracle that knows which planets are hazed, routed to the AIRS-only variant it recovers a third of the methane targets haze removes from the Tier-2 queue at no retraining, and routed to the existing retrieval it is worse than nothing. Before a simulator-trained classifier is used to select Ariel targets, the mission should know which of its labels the spectra decide, retrieve every planet where it can afford to, route the planets a shift flag marks to a retrieval that models the departure or, failing that, to an input that does not see it, and send the undecided planets to Tier 2 as undecided rather than dismissed. The question a selection pipeline must answer first, which of its answers the data decide, is one the data answer and the classifier cannot; that holds for any simulator-trained inference system used to make decisions. The procedures and the code that implement them are released with this paper.

## Acknowledgements

Computations used open-source software (TauREx 3, MultiREx, Exo-Transmit, FastChem, ExoSim 2, scikit-learn, XGBoost, nestle) and public data (the ExoMolOP tables, the PHOENIX stellar atlas, the Ariel Mission Candidate Sample, and the JWST spectra in the Zenodo, OSF and VizieR records cited). No financial support was received.

The author thanks Rachael Kaci, instructor of the Advanced Authentic Research (AAR) program at Henry M. Gunn High School, and research mentor Victoria Lloyd, for their guidance and feedback throughout this project.

In the interest of transparency, the author discloses the following assistance: Anthropic's Claude Code was used to help implement the Python analysis code and to manage the project's GitHub repository, and Google Gemini was used for polishing and proofreading the manuscript. These tools were used under the author's direction; the author designed the study, directed the analysis, interpreted the results, and is responsible for all content, having reviewed and validated all AI-assisted code and text.

## Data and Code Availability

All code and results are publicly available under the MIT License at https://github.com/oy2017/BioSignatureDetectionModel. The supplementary directory `ariel_tier1_trust` contains every result table and figure, the expectations committed before the runs, the evidence that the re-implementation is faithful, and the command that reproduces each result; the scripts are in the directory `v3` of the same repository: the classifiers as re-implemented, the stress test, the identifiability test and the retrieval of every planet, the JWST conversion with its round-trip validation, the panel, the muting census, the diagnostics and selection simulation, and the retraining test. Every reported number can be checked against the committed result files, and the findings log records each result as it came in. Regenerating the spectra requires TauREx 3.3.2 and the third-party inputs cited in Section 2, which a script in the repository downloads and verifies by checksum; the JWST spectra are fetched from their public records; the noise curves of the payload model are committed, and the Python environment is pinned. Every random draw is seeded.

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

## Appendix B. A second classifier

A gradient-boosted classifier (29) for the carbon-to-oxygen ratio (C/O > 1) (14, 26), trained on 18,040 atmospheres with equilibrium chemistry from FastChem (27) at the Tier-3 layout of the mission's radiometric model (28), reached 96.1 % on its own test sets. HCN and C₂H₂, abundant in carbon-rich atmospheres above about 800 K and absent from its training grid as from the Ariel Data Challenge grid (5), lowered its accuracy on carbon-rich planets from 97.6 to 48.9 % while it stayed confident; retraining with the absorbers restored 96.1 %, and distance scores ranked these errors well (AUROC 0.90-0.91). Haze at 3 × 10⁷ m⁻³ cost it only 5.1 points. At the same Tier-3 binning, one classifier was robust to omitted absorbers and fragile to haze, and the other the reverse. The stress test of one Ariel classifier did not predict the other's, so each classifier needs its own measurement of the two error classes.

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
24. Yang O. ExoTransmitOpacity reads Exo-Transmit table pressures as bar (x1e5), but the tables are in Pa. TauREx 3 issue 172, GitHub, 2026. https://github.com/ucl-exoplanets/taurex3/issues/172 (corrected in pull request 174, https://github.com/ucl-exoplanets/taurex3/pull/174)
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
39. Carter AL, May EM, Espinoza N, et al. A benchmark JWST near-infrared spectrum for the exoplanet WASP-39b. Nat Astron, 8: 1008-1019, 2024. https://doi.org/10.1038/s41550-024-02292-x
40. Powell D, Feinstein AD, Lee EKH, et al. Sulfur dioxide in the mid-infrared transmission spectrum of a hot exoplanet. Nature, 626: 979-983, 2024. https://doi.org/10.1038/s41586-024-07040-9
41. Ahrer E-M, Stevenson KB, Mansfield M, et al. Early Release Science of the exoplanet WASP-39b with JWST NIRCam. Nature, 614: 653-658, 2023. https://doi.org/10.1038/s41586-022-05590-4
42. Alderson L, Wakeford HR, Alam MK, et al. Early Release Science of the exoplanet WASP-39b with JWST NIRSpec G395H. Nature, 614: 664-669, 2023. https://doi.org/10.1038/s41586-022-05591-3
43. Feinstein AD, Radica M, Welbanks L, et al. Early Release Science of the exoplanet WASP-39b with JWST NIRISS. Nature, 614: 670-675, 2023. https://doi.org/10.1038/s41586-022-05674-1
44. Rustamkulov Z, Sing DK, Mukherjee S, et al. Early Release Science of the exoplanet WASP-39b with JWST NIRSpec PRISM. Nature, 614: 659-663, 2023. https://doi.org/10.1038/s41586-022-05677-y
45. Tsai S-M, Lee EKH, Powell D, et al. Photochemically produced SO2 in the atmosphere of WASP-39b. Nature, 617: 483-487, 2023. https://doi.org/10.1038/s41586-023-05902-2
46. Roy-Perez J, Pérez-Hoyos S, Barrado-Izagirre N, Chen-Chen H. The effect of JWST/NIRSpec data reduction on the retrieval of WASP-39b atmospheric properties. arXiv, 2602.06722, 2026. https://doi.org/10.48550/arXiv.2602.06722
47. Madhusudhan N, Sarkar S, Constantinou S, Holmberg M, Piette AAA, Moses JI. Carbon-bearing molecules in a possible Hycean atmosphere. Astrophys J Lett, 956: L13, 2023. https://doi.org/10.3847/2041-8213/acf577
48. Holmberg M, Madhusudhan N. Possible Hycean conditions in the sub-Neptune TOI-270 d. Astron Astrophys, 683: L2, 2024. https://doi.org/10.1051/0004-6361/202348238
49. Felix L, Kitzmann D, Demory B-O, Mordasini C. Competing chemical signatures in the atmosphere of TOI-270 d: inference of sulfur and carbon chemistry. Astron Astrophys, 701: A296, 2025. https://doi.org/10.1051/0004-6361/202555194
50. Schmidt SP, MacDonald RJ, Welbanks L, et al. A comprehensive reanalysis of K2-18 b's JWST NIRISS+NIRSpec transmission spectrum. Astron J, 170: 298, 2025. https://doi.org/10.3847/1538-3881/ae019a
51. Hu R, Bello-Arufe A, Tokadjian A, et al. A water-rich interior in the temperate sub-Neptune K2-18 b revealed by JWST. arXiv, 2507.12622, 2025. https://doi.org/10.48550/arXiv.2507.12622
52. Sarkar S, Madhusudhan N, et al. Transmission spectrum of the benchmark temperate exo-Neptune TOI-1231 b. Astrophys J Lett, 1005: L52, 2026. https://doi.org/10.3847/2041-8213/ae6507
53. Fu G, Espinoza N, Sing DK, et al. Water and an escaping helium tail detected in the hazy and methane-depleted atmosphere of HAT-P-18b with JWST NIRISS/SOSS. Astrophys J Lett, 940: L35, 2022. https://doi.org/10.3847/2041-8213/ac9977
54. Fu G, Welbanks L, Deming D, et al. Hydrogen sulfide and metal-enriched atmosphere for a Jupiter-mass exoplanet. Nature, 632: 752-756, 2024. https://doi.org/10.1038/s41586-024-07760-y
55. Davenport B, Kempton EM-R, Nixon MC, et al. TOI-421 b: a hot sub-Neptune with a haze-free, low mean molecular weight atmosphere. Astrophys J Lett, 984: L44, 2025. https://doi.org/10.3847/2041-8213/adcd76
56. Welbanks L, Bell TJ, Beatty TG, et al. A high internal heat flux and large core in a warm Neptune exoplanet. Nature, 630: 836-840, 2024. https://doi.org/10.1038/s41586-024-07514-w
57. Sing DK, Rustamkulov Z, Thorngren DP, et al. A warm Neptune's methane reveals core mass and vigorous atmospheric mixing. Nature, 630: 831-835, 2024. https://doi.org/10.1038/s41586-024-07395-z
58. Beatty TG, Welbanks L, Schlawin E, et al. Sulfur dioxide and other molecular species in the atmosphere of the sub-Neptune GJ 3470 b. Astrophys J Lett, 970: L10, 2024. https://doi.org/10.3847/2041-8213/ad55e9
59. Gressier A, et al. JWST-TST DREAMS: sulfur dioxide in the atmosphere of the Neptune-mass planet HAT-P-26 b from NIRSpec G395H transmission spectroscopy. Astron J, 170: 292, 2025. https://doi.org/10.3847/1538-3881/ae0929
60. Mayo AW, et al. Detection of H2O and CO2 in the atmosphere of the hot super-Neptune WASP-166b with JWST. Astron J, 2025. https://doi.org/10.3847/1538-3881/adda2e
61. Skilling J. Nested sampling for general Bayesian computation. Bayesian Anal, 1: 833-859, 2006. https://doi.org/10.1214/06-BA127
62. Feroz F, Hobson MP, Bridges M. MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics. Mon Not R Astron Soc, 398: 1601-1614, 2009. https://doi.org/10.1111/j.1365-2966.2009.14548.x
63. Forestano RT, Matchev KT, Matcheva K, Unlu EB. Supervised machine learning methods with uncertainty quantification for exoplanet atmospheric retrievals from transmission spectroscopy. arXiv, 2508.04982, 2025. https://doi.org/10.48550/arXiv.2508.04982
64. Lueber A, Novais A, Fisher C, Heng K. Information content of JWST spectra of WASP-39b. Astron Astrophys, 2024. https://doi.org/10.48550/arXiv.2405.02656
65. Bocchieri A, Mugnai LV, Pascale E, Changeat Q, Tinetti G. Detecting molecules in Ariel low resolution transmission spectra. Exp Astron, 56: 605-644, 2023. https://doi.org/10.1007/s10686-023-09911-x
