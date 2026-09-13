# When can a simulator-trained screen be trusted? A reliability map for machine-learning triage of Ariel transmission spectra

Owen Yang<sup>a</sup>

<sup>a</sup> Henry M. Gunn High School, 780 Arastradero Rd, Palo Alto, CA 94306, USA. Tel.: +1 650 686 7813. E-mail: owenhyang@gmail.com (corresponding author)

---

## Abstract

Machine-learning screens have been proposed to decide which of the ~1,000 exoplanets Ariel will observe deserve deeper follow-up, including a Tier-1 classification strategy published by the Ariel consortium. Because no Ariel data exist, every such screen is trained and validated on spectra from the same simulator, so its reported accuracy says nothing about deployment on atmospheres whose physics differs from the simulator's. We ask when such a screen can be trusted and answer with measurements. Two screens go through one procedure: the consortium's Tier-1 molecular screen, rebuilt from its published description, and a tuned screen for a stated Ariel science target, the carbon-rich (C/O > 1) regime. The simulator is broken on purpose along ten physically motivated axes, and for each we measure the loss, the part no retraining can recover, what a randomized training grid absorbs, whether a decline rule flags the errors, and what removes the loss — at the Tier-1 binning where triage would occur and on Ariel's actual target list. The map has three regions. Modelled mismatch is largely absorbed and the residual is irreducible information loss. Disequilibrium chemistry helps. Physics the simulator omitted cannot be trusted: HCN and C₂H₂, present at ~10⁻⁵ in carbon-rich atmospheres and absent from the field's standard Ariel training grid, drop the carbon-rich screen from 96.5 % to 71.9 % — chance on carbon-rich planets, with confident probabilities; no retraining on other ingredients helps, only a distance-based novelty score detects it, and adding the two species restores 96.4 %. The same omission leaves the molecule-presence screen untouched: omitted physics is fatal when it sits on the feature that defines the class. At Tier 1, randomization still helps but no decline rule works and most of every serious loss is unrecoverable by any training. The procedure is released for use on any screen.

**Keywords:** exoplanet atmospheres; transmission spectroscopy; Ariel; machine learning; classification; domain shift; simulation-based inference; out-of-distribution detection; selective prediction; forward-model misspecification; carbon-to-oxygen ratio; disequilibrium chemistry; HCN; acetylene; stellar contamination; opacity databases

## 1. Introduction

Ariel, launching in 2029, will obtain transmission spectra of roughly a thousand transiting exoplanets between 0.5 and 7.8 µm (1). The survey is tiered: a reconnaissance pass over all targets at coarse binning (Tier 1), a chemical survey of a few hundred at higher binning (Tier 2), and repeated observations of a few dozen at the instruments' native resolution (Tier 3). A planet reaches a tier when its spectrum, binned as that tier prescribes, is expected to reach a signal-to-noise ratio of seven on an assumed atmospheric modulation after a number of transits computed with the mission's radiometric model (2-4); the prescribed binnings are approximately R ≈ 1, 3 and 1 in the NIRSpec, AIRS-CH0 and AIRS-CH1 channels at Tier 1 and R ≈ 10, 50 and 10 at Tier 2 (5). Which planets advance from the reconnaissance pass to the deeper tiers is a decision that has to be made for a thousand objects.

Machine learning has been proposed for that decision. The Ariel consortium's population study of the Tier-1 survey presents "a strategy to select candidate planets for reobservation in Ariel's higher resolution Tier", a band metric to classify planets by composition without a retrieval, and four scikit-learn classifiers trained on simulated Tier-1 spectra to flag the presence of individual molecules, reaching 64–89 % per molecule (6). The annual Ariel Data Challenges have drawn some 23,000 submissions to atmospheric-inference tasks on a database of 105,887 simulated spectra (7-9), and a 2026 review counts machine-learning retrieval, detrending and surrogate modelling among the methods being prepared for the mission (10). Every one of these is trained on a forward model of an atmosphere and a model of the telescope, because no spectrum of Ariel quality exists for any target; the same is true of the machine-learning retrieval lineage from random forests to neural posterior estimation and flow matching (12-15).

Each is also validated the same way: on spectra held back from the same simulator. That test gives a reassuring number, but training and testing share every assumption the simulator makes — which molecules exist, which database provides their absorption, whether there are clouds or haze, how the host star's spots contaminate the signal, how the noise is distributed — and the real atmospheres will differ from those assumptions in ways the test cannot expose. The consortium's Tier-1 study says so itself: its spectra "will only be used as 'transmission spectral shapes' to test our methods against", with no claim about the atmospheric model's realism (6). The nearest exception in the literature tested a retrieval network on spectra with an absorber added, an absorber removed, and unocculted star spots, and found it more robust than nested sampling on those three changes (11); it did not ask what could be done about the losses, whether the network could tell, or how the answer depends on the binning at which a decision is made. Classical retrievals have had their forward-model dependence priced — across codes (39), across the equilibrium assumption (20) — and misspecification of simulation-trained inference is an active concern in cosmology (37, 38); the question has not been asked of exoplanet screens, where the mismatch is physics rather than simulation fidelity, and the one Ariel dataset built with a train/test shift varies the instrument, not the atmosphere (45).

A screen that fails quietly returns confident answers, and a mission acts on them. This paper asks the question a mission faces before launch — *when can a simulator-trained screen be trusted, and when can it not?* — and answers it operationally: on data whose physics differs from the training simulator, in ways we anticipate and in ways we deliberately hold out, which planets does the screen still classify at a stated accuracy, does it know which planets it cannot, and what fixes what?

The contributions are three. First, a reliability map, computed with one reusable procedure for two screens — the consortium's own Tier-1 design, rebuilt from its published description, and a tuned screen for a stated Ariel science target — that for each of ten simulator mismatches reports the loss, its irreducible part, what a randomized training grid absorbs, whether the errors can be flagged, and what removes the loss (Section 3.1). Second, the finding that two molecules chemistry has long assigned to carbon-rich atmospheres, and that the field's standard Ariel training grid omits, put a carbon-rich screen at chance on exactly the planets it exists to find, with high confidence, while leaving a molecule-presence screen untouched (Section 3.2). Third, the map at Tier 1, where triage would actually happen, where the map differs materially from Tier 3 (Section 3.3). No method used here is new: domain randomization, selective prediction, distance-based novelty scores, oracle bounds and held-out-axis evaluation are standard in machine learning and are cited where used. What is new is the question asked of an exoplanet screen, the physics-side axes, the framing of each failure as absorb-or-detect-or-fix, the answer as a table a mission can act on, and the finding about the standard grid.

## 2. Materials and Methods

### 2.1 The consortium's Tier-1 screen, rebuilt

The screen of reference (6) is reproduced from the text as literally as it allows, since its simulator wrapper is not public. Their training population (POP-III) takes the Ariel candidate list, repeats each planet four times, draws the atmospheric temperature uniformly in 0.7–1.05 of the planet's equilibrium temperature, the CH₄, H₂O, CO₂ and NH₃ abundances log-uniformly in 10⁻⁹–10⁻², and a grey cloud deck log-uniformly in 5 × 10²–10⁶ Pa, in an H₂/He atmosphere; their test population (POP-I) uses each planet once with abundances in 10⁻⁷–10⁻². Spectra are binned to the seven Tier-1 points (three photometric bands, one NIRSpec point, two AIRS-CH0 points, one AIRS-CH1 point) and scattered with the Tier-1 requirement noise. Each molecule is labelled present if its abundance exceeds 10⁻⁵, 10⁻⁴ or 10⁻³; inputs are normalised to zero mean and unit variance; the four classifiers are scikit-learn's k-nearest neighbours, multilayer perceptron, random forest and support-vector classifier at default settings (47). We use the 965 known planets of the current Mission Candidate Sample (3) as the planet list, our forward model (Section 2.2) in place of theirs, and the consortium simulator's noise shape (Section 2.3). At the 10⁻⁴ threshold our rebuild reaches CH₄ 70–72 %, H₂O 63–69 %, CO₂ 60 % and NH₃ 76–78 % against their 82–87, 71–78, 79–83 and 82–87 %; The shortfall is most likely ours: CH₄, H₂O and NH₃ reach their values when our noise is reduced to about a third, which points to the per-target noise (theirs comes from ArielRad after the integer number of transits each target needs; ours is set exactly at the requirement), while CO₂ does not reproduce at any noise level or with alternative placements of the AIRS-CH0 split, which points to the binning of its 4.3 µm band or to the opacity tables. The deviations and the checks are listed in Appendix A. Where the rebuild falls short of their Table 6 we report the mismatch costs as differences from our own clean values.

### 2.2 A carbon-rich screen

The second screen targets a chemical regime rather than a molecule: is the atmospheric carbon-to-oxygen ratio above one? The C/O ratio is a stated Ariel science objective, a tracer of where and how a planet formed (1), and C/O = 1 is a physical boundary — above it, oxygen is locked in CO and water is depleted, below it carbon is (16, 17) — so the label is a property of the atmosphere rather than a threshold imposed on the labelling convention. The grid holds 18,040 training and 9,016 test planets in five independent test sets (18,156 and 9,072 were drawn; 0.4–0.9 % failed to render) with bulk parameters drawn independently over ranges informed by the Ariel target samples: planet radius 1–26 R⊕, mass 1–300 M⊕, host temperature 2,500–7,500 K, atmospheric temperature 500–2,500 K, C/O uniform in 0.2–1.8 and metallicity uniform in −1 to 1.5 dex. Abundances of H₂O, CH₄, CO, CO₂, NH₃ and O₃ follow from chemical equilibrium computed with FastChem (19) at the photospheric level (10⁻² bar), held constant with altitude, and the label is C/O > 1, which the draw balances to 0.48–0.52 on every split. Spectra are rendered with MultiREx over TauREx 3 (23, 24) at native resolution, without noise, and binned by exact integration to the Ariel Tier-3 layout: three photometric bands below 1.1 µm, then R = 15 over 1.10–1.95 µm, R = 100 over 1.95–3.90 µm and R = 30 over 3.90–7.80 µm, 102 bins in all (2-4). Opacities are Exo-Transmit's tables (25), derived from the compilation of reference (42) with the line-list sources of reference (43); tables for CO, NH₃, HCN and C₂H₂ were added from the same source. Noise is Gaussian with a per-spectrum standard deviation equal to the peak-to-peak amplitude divided by 15, shaped in wavelength by the noise-to-signal curve of an ExoRad model of the payload for the host's temperature (4); the shape of the consortium's ExoSim 2 simulator (27) agrees with it to Spearman 0.985 for solar-type hosts and 0.80 for the coolest, and replacing one shape with the other costs the screen 0.1 points.

Nine pipelines were tuned by cross-validation on the training set: XGBoost (46), random forests and multilayer perceptrons on raw bins, on principal components, and on per-spectrum-normalised bins. Per-spectrum normalisation (subtracting each spectrum's mean and dividing by its standard deviation, so the classifier sees shape rather than depth) is worth four to ten points over the alternatives, and normalised XGBoost is the screen used throughout: 96.47 ± 0.26 % across the five test sets, 96.32 ± 0.08 % across ten training draws, expected calibration error 0.009. Moving the label's cut away from C/O = 1 and retraining lowers accuracy to 94.8 % at C/O = 0.79 and 85.7 % at 1.26 while the majority-class rate rises to 63 and 66 %, so what the screen learns is the chemistry transition at C/O ≈ 1, not an arbitrary line.

### 2.3 The mismatch axes

Each axis re-renders the *same* test planets with one ingredient changed, so every loss is a within-planet difference. The axes and the concern each represents are: an optically thick grey cloud deck at 10⁵ to 10 Pa (aerosols hide the lower atmosphere); a Lee et al. Mie haze at 2 × 10⁵ to 10¹⁰ m⁻³ (22); unocculted star spots at 2–20 % coverage, with the transit-light-source correction built from PHOENIX spectra at a spot contrast of 0.85 (21); white and time-correlated noise at effective signal-to-noise 12 to 5; a gain ramp across the detector; the same planets rendered by an independent radiative-transfer code, Exo-Transmit (25), with the same opacities; the same planets rendered with the ExoMol cross sections (26) in place of Exo-Transmit's tables for H₂O, CH₄, CO₂ and CO; quenched chemistry, in which the carbon–oxygen partition is frozen at the level where the mixing timescale for an eddy-diffusion coefficient K<sub>zz</sub> overtakes the CO–CH₄ conversion timescale of reference (18) on a Guillot profile anchored to the isothermal temperature, for K<sub>zz</sub> from 10⁷ to 10¹¹ cm² s⁻¹; HCN and C₂H₂ added at their FastChem equilibrium or quenched abundances (Section 3.2); and compounds of spots, haze and noise on the same planet. Cloud decks at 10² and 10 Pa, haze at 10¹⁰ m⁻³, faculae and the two non-native codes lie outside anything the randomized training grid contains and are tagged as such. Table 1 lists the axes, the concern each represents, and whether the randomized training grid of Section 2.4 contains it.

**Table 1.** The mismatch axes. Each re-renders the same test planets with one ingredient changed; the last column says whether the ingredient is drawn at random in the randomized training grid, so that a test on it is in-range, or is never shown to the screen, so that a test on it is a held-out axis.

<!-- table:axes -->
| Axis | What is varied | Concern it represents | In the randomized grid |
| :-- | :-- | :-- | :-- |
| Cloud deck | grey, optically thick cloud top at 10⁵, 10⁴, 10³, 10², 10 Pa | aerosols hide the lower atmosphere | 10³–10⁵ Pa, 50 % of planets |
| Haze | Lee et al. Mie haze, 0.1 µm, at 2 × 10⁵, 2 × 10⁶, 3 × 10⁷, 2.4 × 10⁸, 10¹⁰ m⁻³ | photochemical haze mutes and slopes the spectrum | 10⁵–3 × 10⁸ m⁻³, 60 % |
| Stellar contamination | unocculted spots at 2, 5, 10, 20 % coverage (contrast 0.85); faculae 5, 10 %; mixed | the transit light-source effect imprints the star on the planet | spots 0–20 %, 70 % |
| Noise level and colour | white and time-correlated (σ = 3 bins) at effective SNR 12, 10, 8, 5; a gain ramp ×0.25–2 | the deployed noise differs from the training noise | SNR 5–15, all planets |
| Radiative-transfer code | the same planets rendered by Exo-Transmit with the same opacities | code-to-code differences in the forward model | never (held out) |
| Opacity database | ExoMol cross sections for H₂O, CH₄, CO₂, CO in place of Exo-Transmit's tables | line-list differences between databases | never (held out) |
| Quenched chemistry | carbon–oxygen partition frozen at the quench level, K<sub>zz</sub> = 10⁷–10¹¹ cm² s⁻¹ | vertical mixing drives the photosphere out of equilibrium | 50 % of planets |
| Omitted absorbers | HCN and C₂H₂ added at their FastChem equilibrium or quenched abundances | species the training forward model does not contain | never (held out) |
| Compounds | spots 10–20 % × haze 2 × 10⁶–3 × 10⁷ × SNR 8–10 on the same planet | real planets are off on several axes at once | jointly, through the draws above |
| Binning and targets | Tier-1 (7 points) and Tier-2 (51) layouts; the 965 known Mission Candidate Sample planets under the mission's noise definition | the decision is made at Tier 1, on real targets | — |
<!-- /table:axes -->

### 2.4 The measurements

Five quantities are measured for every axis, all on identical test planets and noise realisations, and each relative quantity names its reference. (i) *Loss*: the accuracy the clean-trained screen gives up. (ii) *Irreducible loss*: clean accuracy minus the accuracy of an *oracle* screen trained only at the test condition — the target-only bound of domain adaptation — so that any recovery is measured against what retraining could achieve rather than against clean accuracy. (iii) *Absorption*: the accuracy of a screen trained on a *randomized* grid (28) (noise augmentation acts as a regulariser and cannot undo a particular draw (29); augmentation with a transformation teaches invariance to it (40)) in which every planet carries a random haze density (log-uniform, 10⁵–3 × 10⁸ m⁻³, 60 % of planets), cloud-top pressure (10³–10⁵ Pa, 50 %), spot coverage (0–20 %, 70 %), noise level (SNR 5–15) and equilibrium-or-quenched chemistry, drawn jointly; the same grid rebuilt without one ingredient gives the *held-out* variant that tests whether robustness transfers to physics never seen, the held-out-corruption protocol of reference (41). (iv) *Detection*: five decline rules — the probability margin (30), the disagreement of an XGBoost/random-forest/perceptron ensemble (33), the Mahalanobis distance from the training cloud in feature space (31), the k-nearest-neighbour distance (32) and a PCA reconstruction error — each thresholded to decline 10 % of *clean* test planets and never tuned on a shifted set; a rule is credited only with the accuracy of the accepted planets *above* the clean selective baseline at the same coverage (34), because declining low-confidence planets raises accuracy on clean data too. (v) *Fix*: what removes the loss — training on it, observing longer, or changing the forward model. The whole set is computed at Tier 3 and again at Tier 1 (seven points, Section 5). Expectations for the randomized grid, the omitted species and the consortium screen were committed to the repository before the corresponding runs (Appendix E); the misses are reported as misses.

## 3. Results and Discussion

### 3.1 The map at Tier 3

![Figure 1](figures/fig1_map.png)

**Figure 1.** The reliability map at Tier 3. Each row is one mismatch; the open circle is the accuracy lost by the clean-trained screen, the filled circle the loss of the randomized screen, and the black tick the irreducible part — clean accuracy minus a screen trained only at the test condition. Rows are grouped by region: modelled mismatch (blue), mismatch that helps (green) and physics the simulator omitted (orange). The clean-trained screen scores 96.47 % on clean spectra; the randomized screen 95.41 %.

Table 2 and Figure 1 give the map at Tier 3, and it has three regions.

**Table 2.** The reliability map at Tier 3, one row per mismatch. Loss is in accuracy points relative to the clean-trained screen's 96.47 % on clean spectra; irreducible is clean accuracy minus the oracle trained at the test condition; randomized is the loss of the screen trained on the randomized grid (95.41 % clean); detect names the decline rule whose score ranks the screen's errors with AUROC ≥ 0.85; fix is what removes the loss.

| Mismatch | Loss | Irreducible | Randomized | Detect | Fix |
| :-- | --: | --: | --: | :-- | :-- |
| Cloud deck, 10³ Pa | 4.0 | — | 3.7 | confidence | train on it |
| Haze, 3 × 10⁷ m⁻³ | 6.1 | 1.3 | 2.8 | confidence | train on it (95 % of ceiling) |
| Star spots, 20 % | 11.0 | 3.5 | 4.8 | confidence | train on it (100 %); M-dwarf hosts lose 16, FGK 9 |
| Spots 20 % + haze | 11.1 | — | 5.9 | confidence | sub-additive; randomization helps |
| White noise, SNR 5 | 7.5 | 4.8 | 5.6 | confidence | mostly irreducible; observe longer |
| Correlated noise, SNR 5 | 12.3 | 4.6 | 9.3 | confidence | idem; in-range training 86 % of ceiling |
| Other radiative-transfer code | 6.5 | 1.0 | 6.3 | confidence | only training on that code (85 % reducible) |
| Other opacity tables | 15.9 | 1.8 | 7.4 | confidence and distance | training on them (88 % reducible) |
| Quenched chemistry | −1.8 (gain) | — | −1.8 | nothing to detect | none; holds for K<sub>zz</sub> 10⁷–10¹¹ |
| HCN + C₂H₂ omitted | 24.5 | 0.0 | 20.7 | distance only | add the species: 96.4 % |
| HCN + C₂H₂, quenched | 30.0 | — | 20.8 | distance only | idem |

**Region 1: mismatch you modelled.** Clouds, haze, star spots, noise, a gain ramp, another radiative-transfer code and another opacity database each cost the clean-trained screen 4 to 16 points at the strengths shown. Training on any one of them, at the strength deployed, recovers 95–100 % of what the oracle reaches for spots and haze and 84–107 % for noise when the training noise levels include the test level; the residual is the irreducible part, 0.3–3.5 points for the deterministic re-renders and 4.6–4.8 points for noise at SNR 5, where the information has been removed from the data and only observing longer restores it. The randomized grid — one screen trained with every ingredient drawn at random — halves the loss on clouds, haze, spots and compounds at a cost of 1.06 points on clean spectra, but reaches only 31–83 % of the single-axis ceilings; our pre-registered expectation of ≥ 90 % was wrong, and joint randomization has a capacity price that single-axis training does not. Robustness transfers unevenly to ingredients the randomized grid never contained: 23 % of the achievable improvement for spots, 40 % for aerosols, 49 % for noise; the loss under the other opacity database falls from 15.9 to 7.4 points without those tables ever being seen, while the loss under the other radiative-transfer code does not move. Compound mismatches are sub-additive on this screen (spots and haze together cost 6 points less than the sum of their parts), so compounding is not the danger; the single largest modelled risk is the opacity database, 88 % of whose cost is reducible only by training on the alternative. Stellar contamination is the one axis whose cost depends on the host star: at 20 % coverage M-dwarf hosts lose 16 points, F, G and K hosts 9, as expected from the transit-light-source effect (21); every other axis is flat across host type.

The decline rules behave the same way throughout this region: the probability margin and ensemble disagreement rank the screen's errors well (AUROC 0.80–0.92) and the distance scores do not (0.37–0.72, mostly near 0.45), so confidence is the right instrument for a distortion of known physics. But no rule restores clean-level reliability. With the ensemble rule fixed to decline 10 % of clean planets, every in-range mismatch keeps its accepted planets at 94.9 % or better, at coverage that falls to 65 % at SNR 5 — and the credit against the clean selective baseline at the same coverage is negative for every rule on every distorted axis (mean −4 points): declining buys back part of the loss, never all of it. A split-conformal calibration (36) that guarantees 90 % coverage on clean data delivers 63–88 % under these mismatches — the guarantee is valid for the calibration distribution and needs the deployment distribution to hold (44) —, and its prediction sets come out empty rather than ambiguous, because the screen is wrong with confidence (Appendix C).

**The absorb/detect trade-off.** Randomizing an ingredient into training makes spectra carrying it look in-distribution, so the alarm for that ingredient goes off: the Mahalanobis shift score's ability to separate shifted from clean spectra falls from 0.84 to 0.51 for 20 % spots, 0.91 to 0.64 for haze and 1.00 to 0.35 for the compound, while ingredients the grid never contained stay fully visible (opacity tables 0.98, noise 1.00). The margin's error ranking, meanwhile, improves under randomization (0.85 to 0.91 for spots). A screen cannot be both robust to a mismatch and able to warn about it; a design that wants both keeps a separate, deliberately fragile model as the alarm.

**Region 2: mismatch that helps.** Quenching the carbon–oxygen partition at depth *raises* the clean-trained screen's accuracy from 96.5 to 98.3 %, by 6.6 points on planets below 1,000 K, at every eddy-diffusion coefficient from 10⁷ to 10¹¹ cm² s⁻¹. The reason is chemical: at depth a carbon-rich atmosphere is depleted of water by orders of magnitude (16, 17), quenching carries that depletion up to the photosphere, and the label's separation in water abundance on cool planets grows from 0.3 to 3.3 dex. The reverse deployment — a screen trained on quenched atmospheres applied to equilibrium ones — loses 10.1 points, of which mixing recovers 72 %. The design rule is to train on the chemistry in which the label is hardest to see. Whether this holds under photochemistry, which the forward model does not include, is not tested here.

**Region 3: physics the simulator omitted.** The last two rows of Table 2 are the subject of Section 3.2.

### 3.2 The omitted species

![Figure 2](figures/fig2_absorbers.png)

**Figure 2.** The omitted species. (a) A carbon-rich test planet at the Ariel Tier-3 binning, rendered by the training forward model (blue) and with HCN and C₂H₂ added at their equilibrium abundances (orange); the planet shown has the median effect among hot carbon-rich planets. The clean-trained screen's probability of "carbon-rich" for this planet falls from 1.00 to 0.00. (b) The same probability for all 4,508 carbon-rich test planets, clean (blue) and with the two species present (orange): the share called carbon-rich falls from 98 % to 48 %.

In an atmosphere with C/O ≥ 1 above about 800 K, HCN and C₂H₂ become major constituents, enhanced by three to six orders of magnitude over their solar-composition values, and "can be considered good tracers of the C/O ratio" (16); disequilibrium chemistry enhances them further (17). FastChem gives them at ~10⁻⁵ and ~3 × 10⁻⁵ on the carbon-rich planets of our grid (5 × 10⁻⁵ and 1.5 × 10⁻⁴ when quenched) and at 10⁻¹⁰ and 10⁻¹⁴ on the others. Neither molecule is in the forward model that trained the screen. Neither is in the Ariel Data Challenge database, whose forward model states that "the trace gases are H₂O, CH₄, CO, CO₂ and NH₃" (7), nor in the challenge that followed it (8), nor in the free-chemistry training models of reference (11). The omission is the field's standard, not ours.

Re-rendering the test planets with the two species added at their own abundances, with everything else unchanged and the six training gases verified to be identical, drops the clean-trained screen from 96.47 % to 71.93 % (71.9 ± 1.1 across the five test sets). Carbon-rich planets go from 97.6 % to 47.6 % — chance — while planets with negligible HCN and C₂H₂ are unaffected (96.2 %), which is the built-in null control; the share of test planets called carbon-rich halves from 0.52 to 0.26, so the screen calls carbon-rich planets oxygen-rich. With the species at their quenched abundances the screen falls to 66.5 %, 36 % on carbon-rich planets. The loss is not the property of one model: normalised random forests and perceptrons lose 23 points, XGBoost on principal components 14 and on raw bins 16; the Tier-2 screen falls from 96.0 to 70.7 % and the Tier-1 screen from 88.7 to 66.3 %, with carbon-rich planets at 45–46 % in both.

The mechanism is only partly the band overlap one would guess. On a hot carbon-rich planet of 1,400 ppm feature amplitude, HCN at 10⁻⁵ adds 300–600 ppm of absorption in bands at 1.5, 2.0, 3.0, 4.8 and 7 µm and C₂H₂ at 3 × 10⁻⁵ adds 200–470 ppm at 1.5, 3.0 and 7 µm — a third of the planet's own signal, spread across the range (Figure 2a). Restoring only the 2.75–3.05 µm bins, where the added absorption fills the gap between the water and methane bands, to their clean values recovers 5 of the 24.5 points; restoring 2.75–4.3 µm recovers 8. The rest is broadband: absorption at a third of the amplitude across 1.5–7 µm moves every bin once the spectrum is normalised, which is why the normalised pipelines lose the most.

What can be done about it is the point (Figure 3). The randomized grid, with every other ingredient varied, does not help: 75.8 %. The screen is confidently wrong: its expected calibration error rises from 0.009 to 0.25, the mean probability it assigns to a true carbon-rich planet falls from 0.96 to 0.46, the conformal coverage falls from 91 to 64 % with the sets empty, and the probability margin ranks the errors at AUROC 0.66 (Figure 2b). The distance scores, useless on every distorted axis, rank these errors at 0.90–0.92 and keep 95 % accuracy on the 61 % of planets they accept; on the randomized screen they are the only rules anywhere in the envelope whose credit against the clean selective baseline is positive (+0.3 to +4.3 points). A spectrum with a new absorber is *novel*, so distance from the training set is error; a distorted spectrum of known physics is not. And the loss is entirely reparable in the forward model: a screen trained with HCN and C₂H₂ in the grid scores 96.43 % on the same test set, an irreducible loss of −0.04 points, marginally above the original screen on its own data.

![Figure 3](figures/fig3_detect.png)

**Figure 3.** Confidence against distance. (a) The AUROC with which the probability margin (circles) and the Mahalanobis distance (squares) rank the clean-trained screen's errors, for each mismatch; 0.5 is chance. (b) The credit of the ensemble-disagreement and k-nearest-neighbour decline rules on the randomized screen — accepted-set accuracy minus the clean selective baseline at the same coverage — for the same mismatches. The shaded columns are the two omitted-species cases, on which the ordering of the two instruments inverts.

The consortium's molecule-presence screen gives the refinement that makes the result precise. With HCN and C₂H₂ added at 10⁻⁷–10⁻⁴ to its test population, its CH₄, H₂O, CO₂ and NH₃ classifiers lose 0.2–1.1 points (Appendix A); our pre-registered expectation of at least five points on CH₄ was wrong. The two species carry no information about whether methane exceeds 10⁻⁴, so a screen for methane ignores them; they carry the information that defines carbon-rich, so a screen for carbon-rich is broken by their absence. Omitted physics is fatal when it sits on the feature that defines the class being screened — and the species a generic training grid leaves out are exactly the ones that define a chemical regime.

### 3.3 Tier 1, where triage would happen

![Figure 4](figures/fig4_tiers.png)

**Figure 4.** The same mismatches at the two binnings. (a) Accuracy of the clean-trained screen (bars), the randomized screen (circles) and the ceiling from a screen trained at the test condition (ticks) at Tier 3 (102 points, blue) and Tier 1 (7 points, orange). (b) The screen on the 965 known planets of Ariel's Mission Candidate Sample under the mission's own noise definition, by tier and host type.

Everything in Section 3.1 is a Tier-3 statement. Triage would happen at Tier 1, on seven numbers per planet, so the map was computed again there (Table 3, Figure 4a) with a screen of the same design trained on the seven-point spectra: 88.7 % on clean spectra, 84.1 % at the tier's SNR of seven.

**Table 3.** The map at Tier 1 (seven points). Columns as in Table 2; the clean-trained Tier-1 screen scores 88.7 % on clean spectra and the randomized one 87.9 %. Accepted / coverage is for the best decline rule at a threshold that declines 10 % of clean planets; credit is against the clean selective baseline at that coverage.

| Mismatch | Clean-trained | Randomized | Ceiling | Best rule: accepted / coverage | Credit |
| :-- | --: | --: | --: | :-- | --: |
| Haze, 3 × 10⁷ m⁻³ | 60.3 | 69.3 | 75.5 | 70.7 / 87 % | −21 |
| Star spots, 20 % | 70.0 | 81.7 | 82.9 | 84.3 / 88 % | −7 |
| Spots 20 % + haze | 50.1 | 62.5 | — | 64.9 / 83 % | −28 |
| White noise, SNR 5 | 71.8 | 74.6 | 79.1 | — | — |
| Correlated noise, SNR 5 | 88.1 | 86.5 | 88.1 | 89.6 / 90 % | −1 |
| Other opacity tables | 64.8 | 66.8 | 79.3 | 67.5 / 96 % | −22 |
| HCN + C₂H₂ omitted | 66.3 | 69.0 | 82.3 | 71.0 / 86 % | −21 |
| Quenched chemistry | 92.5 | 94.2 | — | — | gain |

Three things change. First, the losses are larger: a haze that costs six points at Tier 3 costs 28 at Tier 1, because at seven points it removes most of the shape the screen reads, and spots and haze together put the screen at chance. Second, the ceilings are low: a screen trained at the test condition recovers only part of each loss — haze to 75 %, the other opacity tables to 79 %, the omitted species to 82 %, white noise to 79 % — so most of every serious loss is not in the data at this binning and no training can recover it. Read against those ceilings the randomized screen does well, reaching 91 % of the ceiling for spots and 60 % for haze, but the ceilings are the story. Third, no decline rule works: the error-ranking AUROC of every rule on every serious mismatch is 0.57–0.70 and the credits are −7 to −28 points; with seven numbers per planet there is not enough redundancy for either confidence or distance to separate right from wrong. Correlated noise, by contrast, is nearly free at Tier 1, because over seven wide bins it acts like a common offset that normalisation removes, whereas white noise costs 17 points.

The mission's real targets sharpen the picture (Figure 4b). Of the 965 known planets in the Mission Candidate Sample (3), 57 % lie inside the training grid's bulk-parameter box; 29 % exceed its 300 M⊕ mass cap, 12 % orbit stars larger than 1.7 R☉ and 9 % are cooler than 500 K. Rendered with their real hosts, radii, masses and temperatures, C/O drawn as in the grid, and noise set by the mission's own Tier-2 definition (SNR 7 on the five-scale-height modulation after the catalogue's number of transits, with the ExoSim 2 shape), the clean-trained screen scores 94.2 % at Tier-3 binning, 93.1 % at Tier 2 and 79.2 % at Tier 1, where M-dwarf hosts sit at 60 %. The M-dwarf deficit appears with no spots at all: it is the cool, small planets around M dwarfs, in the 500–1,000 K band where the label is hardest, not the star. Planets outside the box score no worse than those inside, so the box edge is not where the screen breaks. The achieved signal-to-noise on the planets' real amplitudes under this convention is 15.6 at the median, so the grid's SNR-15 convention is close to Tier-2 reality for this population. The distance-based decline rules, fixed on grid data, would decline 27–30 % of the real targets under the mission's noise definition against 9–11 % under the grid's: they respond to the noise convention, which is the behaviour Section 4 relies on and a warning for deploying them across simulators.

### 3.4 What a mission should do

The map converts into instructions.

*Audit the species list against the chemistry of the class you screen for.* A screen for a chemical regime needs the molecules that define that regime in its training grid; the standard five-gas grid lacks the two that define carbon-rich atmospheres, and adding them costs nothing but two opacity tables.

*Randomize what you are unsure of, and keep a separate alarm for what you did not model.* One randomized grid halves the loss from clouds, haze, spots and their compounds for a one-point cost and reaches a third to four-fifths of what single-axis training reaches; it also blinds the novelty alarm to whatever it contains, so the alarm should be a second, deliberately fragile model. Use confidence-based declining for distortions of modelled physics and distance-based declining for novel physics, and expect neither to restore clean-level reliability.

*Treat the opacity database as the largest reducible risk and noise as the largest irreducible one.* Sixteen points separate two opacity databases, 88 % of them recoverable only by training on the alternative; four to five of the eight to twelve points lost at SNR 5 are recoverable by nothing but observing time.

*Do not expect Tier-3 reliability at Tier 1.* At seven points, randomization still helps, declining does not, and most of every serious loss is unrecoverable by any training. A screen that ranks Tier-1 spectra should be reported at Tier-1 binning, against ceilings, and on the mission's target list.

*Run the procedure before a screen enters a ranking.* The scripts that produced every number here take any screen and any forward model; the requirements are the re-rendered mismatch sets, an oracle per axis, one randomized grid, the decline rules with clean-fixed thresholds, and the two binnings.

### 3.5 Limitations and what was not found

The map is conditional on its axis set. Every mismatch was generated by our simulators, so an ingredient outside the set — three-dimensional temperature structure, photochemistry, a wrong host-star model — is not priced; holding axes out narrows this and real spectra would test it, but nothing removes it. Abundances are constant with altitude, one quench scheme is used, the spot contrast is fixed at 0.85, and the noise convention is relative to each planet's amplitude except in the target-list test. The consortium screen is reproduced from its text, not its code, and reproduces its Table 6 only in part; its authors have not yet been consulted. The randomized screen has one draw of its ingredient assignments, so its error bars are those of the five test sets rather than of the grid.

Four pre-registered expectations were not met and are reported as such: the randomized grid reaches 31–83 % of the single-axis ceilings, not ≥ 90 %; compound mismatches are sub-additive, not super-additive; the omitted species cost the molecule-presence screen under a point, not five; and the divergence of the two noise simulators' shapes on cool hosts costs nothing. A rule from our own earlier work — that what augmentation can repair is set by how many values a mismatch draws per spectrum — did not survive measurement against the oracle ceilings: augmentation recovers essentially all recoverable loss on every axis, and the apparent rule was an artefact of dividing by clean accuracy. It is withdrawn. The inversion of confidence and distance on novel physics reproduces a known result of the failure-detection literature (35) rather than adding to it, and the M-dwarf concentration of contamination loss is expected physics (21).

## 4. Conclusion

A machine-learning screen trained on simulated Ariel spectra was tested on spectra whose physics differed from its simulator, for the consortium's own Tier-1 design and for a tuned carbon-rich screen, with one procedure that measured for each mismatch the loss, its irreducible part, what a randomized training grid absorbed, whether a decline rule flagged the errors, and what removed the loss. Modelled mismatch was largely absorbed and its residual was irreducible information loss; disequilibrium chemistry raised the carbon-rich screen's accuracy; and physics the simulator omitted — HCN and C₂H₂, present in carbon-rich atmospheres and absent from the field's standard training grid — put the carbon-rich screen at chance on carbon-rich planets with confident probabilities, was absorbed by nothing, was detected only by distance-based novelty scores, and was repaired only by adding the species to the forward model, while leaving a molecule-presence screen untouched. At the Tier-1 binning at which triage would occur, randomization still helped, no decline rule worked, and most of every serious loss was unrecoverable by any training. A screen validated only on the simulator that built it has not been validated; the procedure released here makes the validation that is possible before launch reproducible on any screen.

## Acknowledgements

Computations used open-source software (MultiREx, TauREx 3, FastChem, Exo-Transmit, ExoSim 2, scikit-learn, XGBoost) and public data (the ExoMolOP tables, the PHOENIX stellar atlas, the Ariel Mission Candidate Sample). No financial support was received.

## Data and Code Availability

Every number in this paper is read from a committed result file. The code, the result tables, the pre-registered expectations and their outcomes, the figures, and the documented changes to the forward model (a seeding fix and an aerosol feature in the MultiREx fork, and the added opacity tables) are at https://github.com/oy2017/BioSignatureDetectionModel, directory `v3`; the reliability-map procedure is the sequence in `v3/rerun_all.sh`. Spectra (5 GB) are regenerated by the scripts in about ten hours; the ExoMolOP tables, Exo-Transmit, FastChem, the PHOENIX atlas and ExoSim 2 are public.

## Appendix A. The consortium screen: recipe, deviations, reproduction

The rebuild follows reference (6) §II.2 and §II.5. Deviations: MultiREx/TauREx 3 with Exo-Transmit tables in place of TauREx 3 with ExoMol k-tables; the ExoSim 2 noise shape scaled to the Tier-1 requirement per target in place of ArielRad's Tier-1 noise; the 965 known planets of the 2026 candidate list in place of their 1,000 (which included TESS predictions); training spectra noised once rather than resampled. Reproduction at the 10⁻⁴ threshold (their Table 6 in parentheses): CH₄ 70–72 (82–87), H₂O 63–69 (71–78), CO₂ 60 (79–83), NH₃ 76–78 (82–87) %. Checks: moving the AIRS-CH0 split from 2.76 to 3.0 or 2.5 µm changes the four by at most 2 points; correcting the noise for the integer number of Tier-1 transits per target (a target needing N transits reaches the requirement with N − 0.5 on average) changes them by at most 1.4; scaling the noise to 0.75, 0.5, 0.35 and 0.25 of the requirement gives CH₄ 76, 78, 82, 86, H₂O 70, 72, 72, 86 and NH₃ 80, 82, 84, 61 %, so three of the four classifiers reach their published values at roughly a third of our noise, while CO₂ stays at the majority rate (60–62 %) until the noise is a quarter (71 %). Mismatch costs on their screen at the requirement noise, mean over the four classifiers: haze 3 × 10⁷ m⁻³ −8 to −17, spots 20 % −6 to −13, cloud deck at 10² Pa −6 to −16, noise doubled −5 to −8, HCN + C₂H₂ at 10⁻⁷–10⁻⁴ −0.2 to −1.1 points. Decline rules on their screen: ensemble disagreement across the four classifiers keeps ~100 % of planets under haze (all four agree on the wrong answer), while the k-nearest-neighbour distance declines 97 % — the same confidence-versus-distance split as on ours, with the distance rule acting as a population alarm rather than a per-planet fix. Table 4 gives every number.

**Table 4.** The consortium's Tier-1 screen, rebuilt: accuracy (%) at the 10⁻⁴ abundance threshold for each molecule and classifier on the clean test population and under each mismatch, with the published range from ref. (6) Table 6 for comparison. The "vote" rows are the majority vote of the four classifiers after the ensemble-disagreement or k-nearest-neighbour decline rule, on the accepted planets only.

<!-- table:consortium -->
| Molecule | Classifier | Their Table 6 | Clean | Haze 3 × 10⁷ | Cloud 10² Pa | HCN + C₂H₂ | Spots 10 % | Spots 20 % | Noise ×2 | Noise ×3 |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
<!-- /table:consortium -->

## Appendix B. Full envelope tables

Tables 5 and 6 give, for every mismatch and every decline rule at each binning, the accuracy of all planets and, per rule, the accepted-set accuracy with the coverage kept and the credit against the clean selective baseline at that coverage (points). Table 7 gives the clean-trained, randomized, held-out and ceiling accuracies at both binnings. The source files are `ariel_trust_envelope`, `tier1_trust_envelope`, `ariel_trust_randomized`, `tier1_trust_randomized`, `ariel_oracle` and `tier1_oracle` in `v3/results`.

**Table 5.** The envelope at Tier 3 (102 points), randomized screen. Cells: accepted-set accuracy % (coverage %; credit in points). Thresholds decline 10 % of clean planets.

<!-- table:envelope3 -->
| Mismatch | All | Ensemble | Margin | Mahalanobis | k-NN |
| :-- | --: | :-- | :-- | :-- | :-- |
| Clean | 95.4 | 96.7 (90 %; +0.0) | 98.4 (90 %; +0.0) | 95.2 (90 %; +0.0) | 94.9 (90 %; +0.0) |
| Cloud deck, 10⁵ Pa | 95.3 | 96.7 (90 %; -0.0) | 98.2 (90 %; -0.2) | 94.8 (85 %; -0.2) | 94.7 (88 %; -0.2) |
| Cloud deck, 10⁴ Pa | 94.5 | 96.5 (87 %; -0.6) | 97.9 (89 %; -0.6) | 93.4 (63 %; -1.2) | 93.4 (70 %; -0.5) |
| Cloud deck, 10³ Pa | 92.7 | 95.8 (76 %; -2.9) | 97.4 (85 %; -1.4) | 91.7 (25 %; -4.7) | 92.2 (39 %; -3.2) |
| Cloud deck, 10² Pa (out of range) | 86.7 | 92.8 (57 %; -6.9) | 93.9 (77 %; -5.5) | 89.5 (2 %; -6.3) | 88.5 (5 %; -9.8) |
| Cloud deck, 10 Pa (out of range) | 65.2 | 72.4 (21 %; -27.6) | 73.0 (55 %; -26.9) | 100.0 (0 %; +0.0) | 90.9 (0 %; -9.1) |
| Haze, 2 × 10⁵ m⁻³ | 95.4 | 96.6 (90 %; -0.0) | 98.3 (90 %; -0.1) | 95.0 (89 %; -0.1) | 94.9 (89 %; -0.0) |
| Haze, 2 × 10⁶ m⁻³ | 94.8 | 96.2 (91 %; -0.4) | 98.0 (90 %; -0.4) | 94.1 (82 %; -0.8) | 94.0 (85 %; -0.7) |
| Haze, 3 × 10⁷ m⁻³ | 93.7 | 95.1 (90 %; -1.6) | 97.4 (89 %; -1.2) | 92.7 (70 %; -2.0) | 92.3 (72 %; -1.7) |
| Haze, 2.4 × 10⁸ m⁻³ | 93.1 | 95.0 (85 %; -2.4) | 97.1 (87 %; -1.6) | 93.7 (40 %; -1.1) | 93.6 (49 %; -0.3) |
| Haze, 10¹⁰ m⁻³ (out of range) | 70.1 | 74.2 (31 %; -25.8) | 82.2 (37 %; -17.8) | 91.5 (4 %; -5.2) | 91.7 (7 %; -6.4) |
| Star spots, 2 % | 94.4 | 96.2 (90 %; -0.5) | 98.0 (90 %; -0.4) | 94.2 (86 %; -0.9) | 94.1 (86 %; -0.7) |
| Star spots, 5 % | 93.8 | 96.0 (89 %; -0.8) | 97.6 (89 %; -0.9) | 93.9 (82 %; -1.1) | 94.1 (79 %; -0.3) |
| Star spots, 10 % | 92.9 | 96.0 (87 %; -1.2) | 97.3 (87 %; -1.4) | 92.9 (80 %; -1.9) | 94.2 (71 %; +0.2) |
| Star spots, 20 % | 91.7 | 96.1 (82 %; -1.9) | 97.2 (84 %; -1.7) | 91.2 (82 %; -3.8) | 94.2 (63 %; +0.6) |
| Spots + faculae (out of range) | 93.0 | 95.9 (87 %; -1.2) | 97.2 (88 %; -1.4) | 93.3 (77 %; -1.4) | 94.1 (71 %; +0.1) |
| Faculae, 10 % (out of range) | 93.4 | 97.5 (83 %; -0.3) | 96.6 (90 %; -1.8) | 93.1 (91 %; -2.1) | 93.2 (86 %; -1.6) |
| Quenched chemistry | 98.2 | 98.5 (99 %; +2.8) | 98.7 (99 %; +2.7) | 98.0 (80 %; +3.1) | 98.0 (83 %; +3.4) |
| Other radiative-transfer code | 90.2 | 96.8 (84 %; -0.9) | 94.0 (93 %; -3.8) | 87.8 (57 %; -6.9) | 87.0 (67 %; -6.8) |
| Other opacity tables | 89.1 | 96.0 (52 %; -3.8) | 95.9 (78 %; -3.4) | 97.4 (4 %; +0.8) | 88.8 (21 %; -8.9) |
| Spots 10 % + haze 3 × 10⁷ | 92.1 | 95.6 (86 %; -1.8) | 97.0 (86 %; -1.8) | 91.3 (81 %; -3.6) | 93.9 (59 %; +0.3) |
| Spots 20 % + haze 3 × 10⁷ | 90.5 | 95.5 (81 %; -2.5) | 96.3 (83 %; -2.6) | 90.1 (89 %; -5.0) | 93.7 (54 %; -0.1) |
| HCN + C₂H₂ omitted | 75.8 | 86.6 (71 %; -12.6) | 77.7 (86 %; -21.1) | 94.1 (62 %; -0.5) | 94.0 (66 %; +0.3) |
| HCN + C₂H₂ omitted, quenched | 75.7 | 88.9 (73 %; -10.2) | 77.5 (92 %; -20.5) | 98.5 (43 %; +3.7) | 98.1 (53 %; +4.3) |
| White noise, SNR 12 | 94.4 | 96.7 (87 %; -0.4) | 97.7 (89 %; -0.8) | 93.5 (45 %; -1.3) | 92.8 (75 %; -1.3) |
| White noise, SNR 8 | 92.5 | 96.1 (80 %; -2.1) | 96.9 (88 %; -1.7) | 100.0 (0 %; +0.0) | 95.4 (23 %; -2.0) |
| White noise, SNR 5 | 90.8 | 95.8 (60 %; -3.9) | 95.9 (85 %; -2.9) | nan (0 %; +nan) | nan (0 %; +nan) |
| Correlated noise, SNR 12 | 93.7 | 96.0 (88 %; -1.0) | 97.2 (90 %; -1.2) | 92.6 (77 %; -2.2) | 92.0 (77 %; -2.3) |
| Correlated noise, SNR 8 | 91.1 | 95.9 (80 %; -2.4) | 95.0 (90 %; -3.5) | 92.0 (34 %; -3.3) | 91.6 (34 %; -4.4) |
| Correlated noise, SNR 5 | 87.2 | 94.9 (65 %; -4.7) | 91.2 (88 %; -7.4) | 89.3 (6 %; -7.8) | 91.6 (4 %; -6.3) |
<!-- /table:envelope3 -->

**Table 6.** The envelope at Tier 1 (7 points), randomized screen. Cells as in Table 5.

<!-- table:envelope1 -->
| Mismatch | All | Ensemble | Margin | Mahalanobis | k-NN |
| :-- | --: | :-- | :-- | :-- | :-- |
| Clean | 87.9 | 91.0 (90 %; +0.0) | 91.2 (90 %; +0.0) | 87.7 (90 %; +0.0) | 88.2 (90 %; +0.0) |
| Cloud deck, 10⁵ Pa | 86.7 | 90.0 (89 %; -1.3) | 90.4 (88 %; -1.5) | 86.5 (88 %; -1.2) | 87.4 (88 %; -0.8) |
| Cloud deck, 10⁴ Pa | 81.8 | 85.7 (86 %; -6.4) | 86.5 (85 %; -6.5) | 82.9 (77 %; -4.3) | 84.5 (78 %; -4.0) |
| Cloud deck, 10³ Pa | 75.6 | 79.0 (81 %; -14.0) | 79.9 (82 %; -14.0) | 78.3 (55 %; -8.0) | 80.0 (57 %; -9.3) |
| Cloud deck, 10² Pa (out of range) | 64.5 | 67.8 (72 %; -27.1) | 68.9 (74 %; -26.7) | 71.6 (27 %; -12.2) | 73.2 (27 %; -18.5) |
| Cloud deck, 10 Pa (out of range) | 52.7 | 53.1 (60 %; -43.0) | 53.8 (65 %; -43.0) | 56.8 (6 %; -18.6) | 55.4 (6 %; -40.5) |
| Haze, 2 × 10⁵ m⁻³ | 87.7 | 90.8 (90 %; -0.1) | 91.3 (90 %; +0.1) | 87.5 (91 %; -0.2) | 88.0 (89 %; -0.4) |
| Haze, 2 × 10⁶ m⁻³ | 87.4 | 90.0 (91 %; -0.7) | 90.9 (88 %; -1.0) | 86.3 (84 %; -1.3) | 87.0 (77 %; -1.5) |
| Haze, 3 × 10⁷ m⁻³ | 69.3 | 70.7 (87 %; -21.0) | 75.3 (66 %; -21.4) | 67.9 (73 %; -19.2) | 67.3 (64 %; -21.9) |
| Haze, 2.4 × 10⁸ m⁻³ | 60.5 | 61.3 (71 %; -33.6) | 65.9 (47 %; -32.4) | 58.0 (41 %; -27.5) | 60.4 (60 %; -28.9) |
| Haze, 10¹⁰ m⁻³ (out of range) | 50.2 | 51.6 (47 %; -45.8) | 51.5 (46 %; -46.8) | 61.5 (2 %; -8.9) | 52.3 (10 %; -42.6) |
| Star spots, 2 % | 87.8 | 90.2 (91 %; -0.5) | 91.1 (90 %; +0.0) | 87.9 (85 %; +0.3) | 88.8 (83 %; +0.4) |
| Star spots, 5 % | 86.5 | 89.0 (91 %; -1.8) | 90.8 (87 %; -1.4) | 86.5 (79 %; -0.9) | 87.5 (76 %; -1.0) |
| Star spots, 10 % | 84.9 | 87.1 (90 %; -4.0) | 90.7 (80 %; -3.5) | 84.2 (76 %; -3.0) | 84.7 (73 %; -3.9) |
| Star spots, 20 % | 81.7 | 84.3 (87 %; -7.2) | 91.1 (68 %; -5.5) | 79.4 (79 %; -7.9) | 79.5 (77 %; -9.0) |
| Spots + faculae (out of range) | 85.1 | 87.4 (90 %; -3.5) | 90.6 (82 %; -3.1) | 84.8 (74 %; -2.4) | 85.6 (72 %; -3.1) |
| Faculae, 10 % (out of range) | 84.6 | 87.4 (89 %; -3.9) | 87.8 (90 %; -3.3) | 85.9 (83 %; -1.7) | 86.1 (89 %; -2.2) |
| Quenched chemistry | 94.2 | 95.3 (96 %; +6.0) | 95.5 (96 %; +6.3) | 94.8 (82 %; +7.2) | 95.9 (80 %; +7.6) |
| Other radiative-transfer code | 86.8 | 89.9 (90 %; -1.0) | 90.1 (90 %; -0.9) | 86.6 (84 %; -1.0) | 87.1 (83 %; -1.2) |
| Other opacity tables | 66.8 | 67.5 (96 %; -21.8) | 67.4 (96 %; -21.9) | 68.2 (85 %; -19.4) | 68.0 (87 %; -20.3) |
| Spots 10 % + haze 3 × 10⁷ | 65.4 | 67.2 (86 %; -24.7) | 75.3 (46 %; -22.9) | 65.1 (92 %; -22.7) | 64.5 (89 %; -23.8) |
| Spots 20 % + haze 3 × 10⁷ | 62.5 | 64.9 (83 %; -27.8) | 76.2 (33 %; -23.0) | 62.4 (97 %; -25.5) | 62.1 (96 %; -26.0) |
| HCN + C₂H₂ omitted | 69.0 | 71.0 (86 %; -21.1) | 71.8 (85 %; -21.1) | 67.1 (91 %; -20.6) | 68.4 (91 %; -19.9) |
| HCN + C₂H₂ omitted, quenched | 70.6 | 72.1 (90 %; -18.8) | 72.7 (89 %; -18.8) | 67.0 (84 %; -20.6) | 68.9 (82 %; -19.5) |
| White noise, SNR 12 | 86.7 | 89.5 (90 %; -1.4) | 89.8 (91 %; -1.2) | 86.3 (86 %; -1.3) | 87.3 (86 %; -1.1) |
| White noise, SNR 8 | 83.1 | 85.9 (88 %; -5.4) | 86.1 (90 %; -5.1) | 82.0 (69 %; -5.0) | 84.1 (67 %; -4.7) |
| White noise, SNR 5 | 74.6 | 78.1 (82 %; -14.8) | 77.1 (88 %; -14.8) | 73.2 (42 %; -12.4) | 78.1 (39 %; -12.6) |
| Correlated noise, SNR 12 | 87.6 | 91.0 (90 %; +0.1) | 91.3 (90 %; +0.1) | 87.5 (89 %; -0.2) | 88.0 (89 %; -0.3) |
| Correlated noise, SNR 8 | 87.3 | 90.1 (90 %; -0.8) | 90.5 (91 %; -0.5) | 86.9 (87 %; -0.8) | 87.8 (86 %; -0.5) |
| Correlated noise, SNR 5 | 86.5 | 89.6 (90 %; -1.4) | 89.8 (90 %; -1.4) | 86.2 (80 %; -1.3) | 86.6 (78 %; -1.9) |
<!-- /table:envelope1 -->

**Table 7.** Clean-trained, randomized, held-out and ceiling accuracies (%) at both binnings. Held-out is the randomized grid rebuilt without the case's ingredient (for the two codes and the omitted species every grid is held out and the randomized value is repeated); ceiling is a screen trained at the test condition.

<!-- table:ceilings -->
| Mismatch | Tier 3: clean-trained | Tier 3: randomized | Tier 3: held-out | Tier 3: ceiling | Tier 1: clean-trained | Tier 1: randomized | Tier 1: ceiling |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Clean | 96.5 | 95.4 | — | — | 88.7 | 87.9 | — |
| Cloud deck, 10⁵ Pa | 96.4 | 95.3 | 95.3 | — | 87.8 | 86.7 | — |
| Cloud deck, 10⁴ Pa | 95.2 | 94.5 | 94.1 | — | 83.8 | 81.8 | — |
| Cloud deck, 10³ Pa | 92.4 | 92.7 | 92.2 | — | 76.5 | 75.6 | — |
| Cloud deck, 10² Pa (out of range) | 85.5 | 86.7 | 85.8 | — | 64.9 | 64.5 | — |
| Cloud deck, 10 Pa (out of range) | 63.7 | 65.2 | 64.5 | — | 52.2 | 52.7 | — |
| Haze, 2 × 10⁵ m⁻³ | 96.3 | 95.4 | 95.5 | — | 88.6 | 87.7 | — |
| Haze, 2 × 10⁶ m⁻³ | 94.7 | 94.8 | 94.6 | — | 87.3 | 87.4 | — |
| Haze, 3 × 10⁷ m⁻³ | 90.4 | 93.7 | 92.7 | 95.2 | 60.3 | 69.3 | 75.5 |
| Haze, 2.4 × 10⁸ m⁻³ | 88.1 | 93.1 | 91.5 | — | 50.2 | 60.5 | — |
| Haze, 10¹⁰ m⁻³ (out of range) | 61.3 | 70.1 | 66.2 | — | 52.6 | 50.2 | — |
| Star spots, 2 % | 94.0 | 94.4 | 94.1 | — | 87.4 | 87.8 | — |
| Star spots, 5 % | 91.5 | 93.8 | 92.3 | — | 84.1 | 86.5 | — |
| Star spots, 10 % | 88.8 | 92.9 | 89.8 | 94.2 | 78.3 | 84.9 | 85.5 |
| Star spots, 20 % | 85.5 | 91.7 | 86.5 | 93.0 | 70.0 | 81.7 | 82.9 |
| Spots + faculae (out of range) | 89.2 | 93.0 | 90.0 | — | 79.8 | 85.1 | — |
| Faculae, 10 % (out of range) | 93.0 | 93.4 | 93.3 | — | 86.9 | 84.6 | — |
| Quenched chemistry | 98.3 | 98.2 | 98.1 | — | 92.5 | 94.2 | — |
| Other radiative-transfer code | 90.0 | 90.2 | 90.2 | 95.5 | 86.8 | 86.8 | 87.5 |
| Other opacity tables | 80.6 | 89.1 | 89.1 | 94.6 | 64.8 | 66.8 | 79.3 |
| Spots 10 % + haze 3 × 10⁷ | 87.6 | 92.1 | — | — | 50.7 | 65.4 | — |
| Spots 20 % + haze 3 × 10⁷ | 85.4 | 90.5 | — | — | 50.1 | 62.5 | — |
| HCN + C₂H₂ omitted | 75.8 | 75.8 | — | 96.4 | 69.0 | 69.0 | 82.3 |
| HCN + C₂H₂ omitted, quenched | 75.7 | 75.7 | — | — | 70.6 | 70.6 | — |
| White noise, SNR 12 | 94.5 | 94.4 | 94.0 | — | 86.9 | 86.7 | — |
| White noise, SNR 8 | 92.2 | 92.5 | 92.1 | — | 81.0 | 83.1 | — |
| White noise, SNR 5 | 88.9 | 90.8 | 89.9 | — | 71.8 | 74.6 | — |
| Correlated noise, SNR 12 | 93.0 | 93.7 | 93.6 | — | 88.7 | 87.6 | — |
| Correlated noise, SNR 8 | 89.2 | 91.1 | 90.7 | — | 88.8 | 87.3 | — |
| Correlated noise, SNR 5 | 84.1 | 87.2 | 85.4 | — | 88.1 | 86.5 | — |
<!-- /table:ceilings -->

## Appendix C. Conformal prediction and calibration under mismatch

Split-conformal sets calibrated on clean test planets to 90 % coverage realise 90.8 % on clean spectra and 79 (cloud 10³ Pa), 88 (haze 3 × 10⁷), 81 (spots 20 %), 63 (other opacity tables), 64 and 62 (omitted species, equilibrium and quenched), 77 and 75 % (white and correlated noise at SNR 5); in every case the shortfall appears as empty sets, not ambiguous ones. The clean-trained screen's expected calibration error is 0.009 on clean spectra, 0.08–0.11 under haze and spots, 0.17 under the other opacity tables and 0.25–0.33 under the omitted species, where the mean probability assigned to a true carbon-rich planet falls from 0.96 to 0.46 and 0.34.

## Appendix D. Forward model and its audit

NH₃ carried no absorption in the forward model of our earlier work (MultiREx ships no NH₃ table), a documented simplification that this paper's thesis made indefensible; the Exo-Transmit NH₃ table was added and every result regenerated (haze and opacity-table costs rose; the contamination host gap narrowed; every qualitative statement survived). The MultiREx fork used here fixes a seeding defect in the released package — objects created within the same second draw identical parameters and shuffled clones reproduce their parent — which our grids never exercised because parameters are drawn externally, and adds the cloud and Mie-haze contributions. The quench prescription, the K<sub>zz</sub> sweep and the chemistry checks are in `v3/TRUST_IDEA.md`.

## Appendix E. Pre-registered expectations and outcomes

Committed before the runs: randomized grid within one point of clean (met: 1.06) and ≥ 90 % of ceilings (not met: 31–83 %); compounds super-additive (not met); probability margin the worst detector under re-rendered shifts (not met — it is the best on distorted physics and the worst on novel physics); distance scores detect re-rendered physics (met for detection, wrong about usefulness except on novel physics); the absorb/detect trade-off (met); omitted species cost the consortium screen ≥ 5 points on CH₄ (not met: ≤ 1.1); the Tier-1 consortium screen reproduced within ±5 points (met for NH₃ and CH₄ at reduced noise only).

## References

1. Tinetti G, Drossart P, Eccleston P, et al. A chemical survey of exoplanets with ARIEL. Exp Astron, 46: 135-209, 2018. https://doi.org/10.1007/s10686-018-9598-x
2. Edwards B, Mugnai L, Tinetti G, Pascale E, Sarkar S. An updated study of potential targets for Ariel. Astron J, 157: 242, 2019. https://doi.org/10.3847/1538-3881/ab1cb9
3. Edwards B, Tinetti G. The Ariel target list: the impact of TESS and the potential for characterizing multiple planets within a system. Astron J, 164: 15, 2022. https://doi.org/10.3847/1538-3881/ac6bf9
4. Mugnai LV, Pascale E, Edwards B, Papageorgiou A, Sarkar S. ArielRad: the Ariel radiometric model. Exp Astron, 50: 303-328, 2020. https://doi.org/10.1007/s10686-020-09676-7
5. Radica M, et al. On the information content of Ariel transmission spectra: reassessing the tier system. arXiv, 2604.07598, 2026. https://doi.org/10.48550/arXiv.2604.07598
6. Mugnai LV, Al-Refaie A, Bocchieri A, Changeat Q, Pascale E, Tinetti G. Alfnoor: assessing the information content of Ariel's low-resolution spectra with planetary population studies. Astron J, 162: 288, 2021. https://doi.org/10.3847/1538-3881/ac2e92
7. Changeat Q, Yip KH. ESA-Ariel Data Challenge NeurIPS 2022: introduction to exo-atmospheric studies and presentation of the Atmospheric Big Challenge (ABC) database. RAS Tech Instrum, 2: 45-61, 2023. https://doi.org/10.1093/rasti/rzad001
8. Aubin M, et al. Simulation-based inference for exoplanet atmospheric retrieval: insights from winning the Ariel Data Challenge 2023 using normalizing flows. arXiv, 2309.09337, 2023. https://doi.org/10.48550/arXiv.2309.09337
9. Yip KH, Changeat Q, Nikolaou N, et al. Lessons learned from the 1st Ariel machine learning challenge. Proc Mach Learn Res, 220, 2023. https://proceedings.mlr.press/v220/
10. Yakubu M, Jude VO. Machine learning and deep learning for exoplanet detection and atmospheric characterization with JWST and the upcoming Ariel mission. arXiv, 2606.23766, 2026. https://doi.org/10.48550/arXiv.2606.23766
11. Ardévol Martínez F, Min M, Kamp I, Palmer PI. Convolutional neural networks as an alternative to Bayesian retrievals for interpreting exoplanet transmission spectra. Astron Astrophys, 662: A108, 2022. https://doi.org/10.1051/0004-6361/202142976
12. Márquez-Neila P, Fisher C, Sznitman R, Heng K. Supervised machine learning for analysing spectra of exoplanetary atmospheres. Nat Astron, 2: 719-724, 2018. https://doi.org/10.1038/s41550-018-0504-2
13. Nixon MC, Madhusudhan N. Assessment of supervised machine learning for atmospheric retrieval of exoplanets. Mon Not R Astron Soc, 496: 269-281, 2020. https://doi.org/10.1093/mnras/staa1150
14. Vasist M, Rozet F, Absil O, Mollière P, Nasedkin E, Louppe G. Neural posterior estimation for exoplanetary atmospheric retrieval. Astron Astrophys, 672: A147, 2023. https://doi.org/10.1051/0004-6361/202245263
15. Gebhard TD, Wildberger J, Dax M, et al. Flow matching for atmospheric retrieval of exoplanets: where reliability meets adaptive noise levels. Astron Astrophys, 693: A42, 2025. https://doi.org/10.1051/0004-6361/202451861
16. Madhusudhan N. C/O ratio as a dimension for characterizing exoplanetary atmospheres. Astrophys J, 758: 36, 2012. https://doi.org/10.1088/0004-637X/758/1/36
17. Moses JI, Madhusudhan N, Visscher C, Freedman RS. Chemical consequences of the C/O ratio on hot Jupiters: examples from WASP-12b, CoRoT-2b, XO-1b, and HD 189733b. Astrophys J, 763: 25, 2013. https://doi.org/10.1088/0004-637X/763/1/25
18. Zahnle KJ, Marley MS. Methane, carbon monoxide, and ammonia in brown dwarfs and self-luminous giant planets. Astrophys J, 797: 41, 2014. https://doi.org/10.1088/0004-637X/797/1/41
19. Stock JW, Kitzmann D, Patzer ABC, Sedlmayr E. FastChem: a computer program for efficient complex chemical equilibrium calculations in the neutral/ionized gas phase with applications to stellar and planetary atmospheres. Mon Not R Astron Soc, 479: 865-874, 2018. https://doi.org/10.1093/mnras/sty1531
20. Kawashima Y, Min M. Implementation of disequilibrium chemistry to spectral retrieval code ARCiS and application to 16 exoplanet transmission spectra. Astron Astrophys, 656: A90, 2021. https://doi.org/10.1051/0004-6361/202141548
21. Rackham BV, Apai D, Giampapa MS. The transit light source effect: false spectral features and incorrect densities for M-dwarf transiting planets. Astrophys J, 853: 122, 2018. https://doi.org/10.3847/1538-4357/aaa08c
22. Lee J-M, Heng K, Irwin PGJ. Atmospheric retrieval analysis of the directly imaged exoplanet HR 8799b. Astrophys J, 778: 97, 2013. https://doi.org/10.1088/0004-637X/778/2/97
23. Al-Refaie AF, Changeat Q, Waldmann IP, Tinetti G. TauREx 3: a fast, dynamic, and extendable framework for retrievals. Astrophys J, 917: 37, 2021. https://doi.org/10.3847/1538-4357/ac0252
24. Duque-Castaño DS, Zuluaga JI, Flor-Torres L. Machine-assisted classification of potential biosignatures in Earth-like exoplanets using low signal-to-noise ratio transmission spectra. Mon Not R Astron Soc, 539: 1528-1552, 2025. https://doi.org/10.1093/mnras/staf605
25. Kempton EM-R, Lupu R, Owusu-Asare A, Slough P, Cale B. Exo-Transmit: an open-source code for calculating transmission spectra for exoplanet atmospheres of varied composition. Publ Astron Soc Pac, 129: 044402, 2017. https://doi.org/10.1088/1538-3873/aa61ef
26. Chubb KL, Rocchetto M, Yurchenko SN, et al. The ExoMolOP database: cross sections and k-tables for molecules of interest in high-temperature exoplanet atmospheres. Astron Astrophys, 646: A21, 2021. https://doi.org/10.1051/0004-6361/202038350
27. Mugnai LV, Bocchieri A, Pascale E, Lorenzani A, Papageorgiou A. ExoSim 2: the new exoplanet observation simulator applied to the Ariel space mission. Exp Astron, 59: 9, 2025. https://doi.org/10.1007/s10686-024-09976-2
28. Tobin J, Fong R, Ray A, Schneider J, Zaremba W, Abbeel P. Domain randomization for transferring deep neural networks from simulation to the real world. Proc IEEE/RSJ Int Conf Intell Robots Syst, 23-30, 2017. https://doi.org/10.1109/IROS.2017.8202133
29. Bishop CM. Training with noise is equivalent to Tikhonov regularization. Neural Comput, 7: 108-116, 1995. https://doi.org/10.1162/neco.1995.7.1.108
30. Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. Int Conf Learn Represent, 2017. https://doi.org/10.48550/arXiv.1610.02136
31. Lee K, Lee K, Lee H, Shin J. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Adv Neural Inf Process Syst, 31, 2018. https://doi.org/10.48550/arXiv.1807.03888
32. Sun Y, Ming Y, Zhu X, Li Y. Out-of-distribution detection with deep nearest neighbors. Proc Int Conf Mach Learn, 162: 20827-20840, 2022. https://doi.org/10.48550/arXiv.2204.06507
33. Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv Neural Inf Process Syst, 30, 2017. https://doi.org/10.48550/arXiv.1612.01474
34. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. Adv Neural Inf Process Syst, 30, 2017. https://doi.org/10.48550/arXiv.1705.08500
35. Jaeger PF, Lüth CT, Klein L, Bungert TJ. A call to reflect on evaluation practices for failure detection in image classification. Int Conf Learn Represent, 2023. https://doi.org/10.48550/arXiv.2211.15259
36. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv, 2107.07511, 2021. https://doi.org/10.48550/arXiv.2107.07511
37. Schmitt M, Bürkner P-C, Köthe U, Radev ST. Detecting model misspecification in amortized Bayesian inference with neural networks. arXiv, 2112.08866, 2021. https://doi.org/10.48550/arXiv.2112.08866
38. Cannon P, Ward D, Schmon SM. Investigating the impact of model misspecification in neural simulation-based inference. arXiv, 2209.01845, 2022. https://doi.org/10.48550/arXiv.2209.01845
39. Barstow JK, Changeat Q, Garland R, Line MR, Rocchetto M, Waldmann IP. A comparison of exoplanet spectroscopic retrieval tools. Mon Not R Astron Soc, 493: 4884-4909, 2020. https://doi.org/10.1093/mnras/staa548
40. Chen S, Dobriban E, Lee JH. A group-theoretic framework for data augmentation. J Mach Learn Res, 21: 1-71, 2020. https://doi.org/10.48550/arXiv.1907.10905
41. Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and perturbations. Int Conf Learn Represent, 2019. https://doi.org/10.48550/arXiv.1903.12261
42. Freedman RS, Lustig-Yaeger J, Fortney JJ, Lupu RE, Marley MS, Lodders K. Gaseous mean opacities for giant planet and ultracool dwarf atmospheres over a range of metallicities and temperatures. Astrophys J Suppl Ser, 214: 25, 2014. https://doi.org/10.1088/0067-0049/214/2/25
43. Lupu RE, Zahnle K, Marley MS, et al. The atmospheres of Earthlike planets after giant impact events. Astrophys J, 784: 27, 2014. https://doi.org/10.1088/0004-637X/784/1/27
44. Tibshirani RJ, Foygel Barber R, Candès EJ, Ramdas A. Conformal prediction under covariate shift. Adv Neural Inf Process Syst, 32, 2019. https://doi.org/10.48550/arXiv.1904.06019
45. Mugnai LV, et al. A public dataset of Ariel simulated observations for developing exoplanetary atmosphere data reduction pipelines. arXiv, 2605.03719, 2026. https://doi.org/10.48550/arXiv.2605.03719
46. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min, 785-794, 2016. https://doi.org/10.1145/2939672.2939785
47. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res, 12: 2825-2830, 2011. https://doi.org/10.48550/arXiv.1201.0490
