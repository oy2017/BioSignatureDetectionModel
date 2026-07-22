# Revision Work — for review

*A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra* — Journal of High School Science, revise and resubmit.

---

## Reviewer 1

> **Reading guide for the quotes below.** In each "Reviewer 1 wrote" block, his **main concern** is in **bold**, and every experiment or action he requested is tagged by what we did: ✅ done in full · ⚠️ partial · ❌ not done this cycle. Phrase colours — blue for the concern, green / amber / red for status — render in a markdown editor or an exported PDF; on GitHub the **bold** and the ✅ ⚠️ ❌ markers still show, but the text colour does not.

### Overview — opening assessment and overall recommendation

**Reviewer 1 wrote (opening):**

> The manuscript presents a technically well-executed machine learning benchmark and the comparison among XGBoost, Random Forest, MLP, and CNN is carefully conducted. The statistical analyses are generally appropriate, the experimental pipeline is well documented, and the manuscript is clearly written. However, <span style="color:#0b5cad">**I believe the central scientific claims are presently insufficiently supported**</span>. The primary issue is not the implementation of the machine learning algorithms themselves but rather whether the experiments actually demonstrate what the manuscript claims. In its current form, the work demonstrates that an XGBoost classifier can successfully recover predefined biosignature labels from spectra generated within a single synthetic simulation framework. It does not yet demonstrate that the proposed methodology identifies biosignatures in actual exoplanet transmission spectra or that the mechanistic interpretation of the PCA representation is correct. These issues require additional experimental validation before the manuscript is suitable for publication.

**Reviewer 1 wrote (recommendation):**

> Overall, I believe the manuscript contains a promising machine learning framework and an extensive computational evaluation. However, the central scientific questions remain unresolved. Specifically, <span style="color:#0b5cad">**the manuscript has not yet demonstrated that the classifier generalizes beyond the synthetic TauREx environment, that the PCA interpretation accurately reflects the underlying atmospheric physics, or that the whitening procedure preferentially enhances chemically meaningful information rather than simulator-specific low-variance structure**</span>. These are not matters of presentation but of experimental validation. Accordingly, I recommend major revision.

**What it means.** The benchmark and statistics are sound; the objection is that the paper claims more than it shows — real detection and a physical reading of the PCA — and the fix is new experiments plus narrower claims.

**How this is addressed.** Three threads, each detailed below: new domain-shift experiments (R1-1), a conceded and reworded task definition backed by two analyses (R1-2), and a withdrawn PCA interpretation replaced by a measured result plus whitening ablations (R1-3 to R1-6).

**For your input.** Scope wording for the title: insert "Synthetic" before "Exoplanet Transmission Spectra," or leave the title and carry the scoping in the first and last sentences of the abstract?

### R1-1 — Confined to a single simulator; robustness under domain shift

**Reviewer 1 wrote:**

> The most significant concern is that <span style="color:#0b5cad">**the entire study is developed, trained, validated, and tested exclusively within a single synthetic ecosystem**</span>. Every transmission spectrum is generated using the TauREx/MultiREx forward model, every class label is derived directly from the same simulated atmospheric parameters, and every evaluation is performed on additional spectra generated under essentially identical modeling assumptions. Consequently, the classifier is never evaluated outside the statistical distribution of the simulator that created both the inputs and the labels. High predictive accuracy under these conditions demonstrates internal consistency within the simulation environment but does not establish that the learned decision boundary represents genuine atmospheric physics rather than simulator-specific statistical regularities. The central question for a biosignature classifier is not whether it generalizes to additional TauREx simulations, but whether it generalizes to real exoplanet observations. The manuscript does not presently address this question.
>
> This distinction is especially important because real Ariel observations will differ substantially from the synthetic spectra used in this work. Ariel will observe heterogeneous transmission spectra containing instrumental systematics, wavelength-dependent detector effects, correlated noise, stellar contamination, imperfect calibration, incomplete molecular opacity databases, cloud and haze variability, photochemical complexity, and three-dimensional atmospheric structure. In contrast, the present study assumes one-dimensional atmospheres, Gaussian noise with a fixed signal-to-noise ratio, predefined atmospheric compositions, and idealized observing conditions. These assumptions substantially reduce the complexity of the classification problem. Before this work can be considered applicable to Ariel, the authors should demonstrate that the classifier remains robust under realistic domain shift. At minimum, the manuscript should include experiments using spectra generated by <span style="color:#1a7f37">**independent radiative transfer codes ✅**</span>, <span style="color:#1a7f37">**alternative molecular opacity databases ✅**</span>, <span style="color:#1a7f37">**different cloud and haze prescriptions ✅**</span>, <span style="color:#9a6700">**different stellar contamination models ⚠️**</span>, <span style="color:#1a7f37">**varying spectral resolutions and signal-to-noise ratios ✅**</span>, <span style="color:#9a6700">**more realistic instrumental systematics ⚠️**</span>, and <span style="color:#9a6700">**domain-shift or transfer-learning experiments ⚠️**</span>.

**What it means.** Everything happens inside one simulator, so high accuracy could reflect simulator quirks. He lists seven axes along which robustness must be shown before the method is Ariel-applicable. (His clause about labels deriving from the generation parameters is answered under R1-2.)

**What we did — five experiments.** In every case the trained pipeline is left untouched; only the evaluation data changes. Baseline throughout: 88.9% accuracy, Brier 0.080.

1. **Injected observational systematics (sweep).** Six perturbation families, each swept over five strengths, applied only to the test sets at the raw transit-depth stage. The entire preprocessing chain (raw scaler, 102-component PCA, post-PCA scaler) is fit on clean training data only and applied unchanged; strengths are anchored to the measured per-spectrum noise floor (median 1.04×10⁻⁴, which is only 15.8% of the total spectral scatter). Results (XGBoost): correlated noise at effective SNR 5 −22.0 points; white noise SNR 5 −15.1; resolution R 200→75 −15.2; stellar-contamination proxy at 2× floor −5.5; gain ramp −2.7; baseline offset −0.2. **Finding:** calibration-type errors are largely tolerated (< 2 points); correlated noise is the dominant vulnerability and is worse than white noise at equal SNR (75.6% vs 80.7% at SNR 10). *Evidence:* [`domain_shift_sweep.py`](../domain_shift_sweep.py) → [`H2_domain_shift_sweep.txt`](../final_results/H2_domain_shift_sweep.txt) ([csv](../final_results/H2_domain_shift_sweep.csv)); figures [`domain_shift_accuracy.png`](../final_results/domain_shift_accuracy.png), [`domain_shift_calibration.png`](../final_results/domain_shift_calibration.png).

2. **Out-of-envelope generalisation.** Train on planets with radius ≤ 15 R⊕, test on radius > 15. Accuracy 75.8%, against a same-sized random-split control at 84.1% and the full in-distribution 88.9%. Holding training-set size fixed, the genuine extrapolation penalty is −8.3 points (the rest of the naïve gap is just less training data). *Evidence:* [`domain_shift_sweep.py`](../domain_shift_sweep.py) `--mode extrapolation` → [`H2_extrapolation_split.txt`](../final_results/H2_extrapolation_split.txt).

3. **Cloud and haze prescriptions (paired).** A grey cloud deck and a Lee–Mie photochemical haze are each added to the forward model, and the five committed clear test sets are re-rendered with the aerosol added and nothing else changed — so every aerosol spectrum is paired with a clear-sky spectrum of the *same planet*, and the degradation is a within-planet difference. (The no-aerosol path reproduces the committed spectra to 1.5×10⁻¹⁵ before any set is generated, confirming only the aerosol differs.) Deck: −2.3 (10⁵ Pa), −10.5 (10⁴), −17.3 (10³), −25.3 (10² Pa); haze reaches comparable magnitude at strong muting. Informative range 10⁵–10² Pa; below that the spectra are featureless. The classifiers never saw an aerosol in training. *Evidence:* [`generate_aerosol_paired.py`](../generate_aerosol_paired.py) → [`evaluate_aerosol_paired.py`](../evaluate_aerosol_paired.py) → [`H2_aerosol_paired.txt`](../final_results/H2_aerosol_paired.txt) ([csv](../final_results/H2_aerosol_paired.csv)).

4. **Independent radiative transfer code.** The test planets are recomputed with Exo-Transmit, a separately written code. Its opacity tables are byte-identical to the ones the training simulator uses (md5-verified), so *only the radiative transfer implementation changes*. Accuracy 88.9% → 84.8% (−4.1). The two codes agree closely on the spectra: median per-planet correlation 0.997, mean transit depth within 0.7%. **Caveat we state:** Exo-Transmit assumes constant gravity through the atmosphere while the training simulator integrates hydrostatically, which alone accounts for a 4–12% contrast difference — so −4.1 is an *upper bound* on the pure numerical-scheme difference. *Evidence:* [`generate_exotransmit_testset.py`](../generate_exotransmit_testset.py) (+ [`exotransmit_harness.py`](../exotransmit_harness.py)) → [`evaluate_exotransmit.py`](../evaluate_exotransmit.py) → [`H2_exotransmit.txt`](../final_results/H2_exotransmit.txt) ([csv](../final_results/H2_exotransmit.csv)).

5. **Alternative molecular opacity data.** The H₂O, CH₄ and CO₂ tables are replaced with ExoMol line lists (POKAZATEL, YT34to10, UCL-4000), code and structure fixed, planets re-rendered and paired (harness validated to reproduce the originals to 1.2×10⁻¹⁴). Accuracy 88.9% → 72.8% (−16.1); additionally replacing ozone with HITRAN → 66.0% (−22.9). **This failure is a bias shift, not signal loss:** feature amplitude barely moves (0.95 → 0.93), but the predicted-positive rate moves from 49.5% to 61.8% (and to 44.0% with ozone replaced) against a true 50.3%. *Evidence:* [`generate_opacity_swap_testset.py`](../generate_opacity_swap_testset.py) → [`evaluate_opacity_swap.py`](../evaluate_opacity_swap.py) → [`H2_opacity_swap.txt`](../final_results/H2_opacity_swap.txt), [`H2_opacity_swap_o3.txt`](../final_results/H2_opacity_swap_o3.txt).

#### Results — R1-1

**Injected systematics** (XGBoost, worst strength per family; full sweep in [`H2_domain_shift_sweep.txt`](../final_results/H2_domain_shift_sweep.txt)):

| Perturbation | Accuracy | Change | Brier |
| :-- | --: | --: | --: |
| Clean baseline | 88.9% | — | 0.080 |
| Baseline offset | 88.7% | −0.2 | 0.080 |
| Gain ramp | 86.2% | −2.7 | 0.097 |
| Stellar contamination | 83.4% | −5.5 | 0.119 |
| Resolution R 200→75 | 73.7% | −15.2 | 0.197 |
| White noise SNR 15→5 | 73.9% | −15.1 | 0.190 |
| Correlated noise SNR 15→5 | 66.9% | −22.0 | 0.250 |

![Accuracy under each injected systematic versus perturbation strength](../final_results/domain_shift_accuracy.png)

![Calibration (Brier score) under each injected systematic versus perturbation strength](../final_results/domain_shift_calibration.png)

**Out-of-envelope generalisation** ([`H2_extrapolation_split.txt`](../final_results/H2_extrapolation_split.txt)) — train on small planets, test on large:

| Condition | Accuracy | Brier |
| :-- | --: | --: |
| Extrapolation (train R ≤ 15 R⊕, test R > 15) | 75.8% | 0.175 |
| Control (random split, same training-set size) | 84.1% | 0.116 |
| Full training set (in-distribution) | 88.9% | 0.080 |

Holding training-set size fixed, the genuine extrapolation penalty is **−8.3 points**; the rest of the naïve gap is just reduced training data.

**Aerosols** (paired, XGBoost; [`H2_aerosol_paired.txt`](../final_results/H2_aerosol_paired.txt)). "amp" is median feature amplitude relative to a clear atmosphere:

| Aerosol | amp | Accuracy | Change | Brier |
| :-- | --: | --: | --: | --: |
| Clear (baseline) | 0.97× | 88.9% | — | 0.080 |
| Cloud deck 10⁵ Pa | 0.94× | 86.6% | −2.3 | 0.098 |
| Cloud deck 10⁴ Pa | 0.83× | 78.4% | −10.5 | 0.151 |
| Cloud deck 10³ Pa | 0.58× | 71.6% | −17.3 | 0.212 |
| Cloud deck 10² Pa | 0.30× | 63.7% | −25.3 | 0.271 |
| Haze 2×10⁶ m⁻³ | 0.79× | 82.0% | −7.0 | 0.132 |
| Haze 1×10¹⁰ m⁻³ | 0.25× | 63.3% | −25.6 | 0.264 |

**Independent code and alternative opacity data** ([`H2_exotransmit.txt`](../final_results/H2_exotransmit.txt), [`H2_opacity_swap.txt`](../final_results/H2_opacity_swap.txt)):

| | Baseline | Exo-Transmit (code changed) | ExoMol opacity (3 molecules) | + HITRAN ozone |
| :-- | --: | --: | --: | --: |
| Accuracy | 88.9% | 84.8% (−4.1) | 72.8% (−16.1) | 66.0% (−22.9) |
| Brier | 0.080 | 0.112 | 0.203 | 0.253 |
| Predicted-positive rate (true 50.3%) | 49.5% | — | 61.8% | 44.0% |

**Coverage of the seven axes.** "The seven axes" are the seven kinds of robustness test the reviewer explicitly lists in the last sentence of his comment above (independent code, alternative opacity, clouds/hazes, stellar contamination, resolution/SNR, instrument systematics, domain-shift/transfer-learning). Four are covered in full, three in part:

| # | Axis the reviewer requested | Status | Covered by | Evidence |
| :-- | :-- | :-: | :-- | :-- |
| 1 | Independent radiative transfer codes | ✅ full | Experiment 4 (Exo-Transmit) | [H2_exotransmit.txt](../final_results/H2_exotransmit.txt) |
| 2 | Alternative molecular opacity databases | ✅ full | Experiment 5 (opacity swap) | [H2_opacity_swap.txt](../final_results/H2_opacity_swap.txt) |
| 3 | Different cloud and haze prescriptions | ✅ full | Experiment 3 (aerosols) | [H2_aerosol_paired.txt](../final_results/H2_aerosol_paired.txt) |
| 4 | Different stellar contamination models | ◐ partial | Experiment 1 — a single 1/λ proxy, **not** a physical spot model | [H2_domain_shift_sweep.txt](../final_results/H2_domain_shift_sweep.txt) |
| 5 | Varying spectral resolution and SNR | ✅ full | Experiment 1 (sweep) | [H2_domain_shift_sweep.txt](../final_results/H2_domain_shift_sweep.txt) |
| 6 | More realistic instrumental systematics | ◐ partial | Experiment 1 — parametric forms, **not** Ariel's instrument model | [H2_domain_shift_sweep.txt](../final_results/H2_domain_shift_sweep.txt) |
| 7 | Domain-shift **or** transfer-learning | ◐ partial | Experiments 1 & 2 — domain shift done; transfer learning not run | [H2_extrapolation_split.txt](../final_results/H2_extrapolation_split.txt) |

**Status.** *Done in full — 4 of the 7 axes:* independent radiative transfer code (−4.1 points), alternative opacity data (−16.1, or −22.9 with ozone), cloud and haze prescriptions, and resolution/SNR. *Partial — 3 of the 7:* stellar contamination (a single 1/λ proxy, not a physical spot model), instrumental systematics (parametric forms, not Ariel's instrument model), and transfer learning (the domain-shift half is done; transfer learning itself was not run).

**For your input.** Is presenting the four completed axes and naming the three partial ones as next steps sufficient for this cycle, or would you want any of the three attempted before resubmission?

### R1-2 — The positive class is a labelling rule, not a detection

**Reviewer 1 wrote:**

> A related concern is the definition of the positive class. The manuscript defines biosignatures using fixed abundance thresholds (CH₄ > 10⁻⁶ and O₃ > 10⁻⁷), and every atmosphere satisfying these thresholds is labeled as positive. Consequently, <span style="color:#0b5cad">**the classifier is not discovering biosignatures but rather learning a predefined labeling rule imposed during simulation**</span>. This distinction is important because Ariel itself does not operate by measuring atmospheric abundances and applying deterministic thresholds. Ariel measures transmission spectra, after which atmospheric abundances must be inferred through retrieval algorithms that incorporate observational uncertainties, parameter degeneracies, stellar context, and competing abiotic explanations. The proposed machine learning framework therefore learns to reproduce a synthetic labeling convention rather than the operational biosignature identification process that Ariel will ultimately employ. Additional experiments using <span style="color:#b3261e">**retrieval-derived abundances with realistic uncertainties ❌**</span> or <span style="color:#9a6700">**probabilistic labels ⚠️**</span> would substantially strengthen the relevance of the proposed framework.

**What it means.** The labels are computed from the same abundances that generated the spectra, so the classifier recovers a labelling convention rather than inferring a detection. Retrieval-derived or probabilistic labels would make the task closer to the real one.

**What we did.** We concede the point rather than contest it, and the response has two parts.

- **Reword (the concession, and the largest part of the fix).** Rename the task wherever it describes what the model does — from "biosignature detection" to "recovery of an abundance-threshold labelling" — keeping "biosignature" only for the underlying scientific motivation. This costs nothing but words and stops the paper claiming more than the experiment shows.
- **Two supporting analyses** that quantify how much the labelling convention governs the result:
  - **Threshold sensitivity.** Both cutoffs are moved together by ±0.25 and ±0.5 dex, the data relabelled, and the pipeline retrained from scratch at each setting. Accuracy stays between 86.99% and 88.91% across a full dex of movement (spread < 2 points), so the headline is not an artefact of the particular cutoffs. *Evidence:* [`analyze_threshold_sensitivity.py`](../analyze_threshold_sensitivity.py) → [`H2_threshold_sensitivity.txt`](../final_results/H2_threshold_sensitivity.txt) ([csv](../final_results/H2_threshold_sensitivity.csv)).
  - **Margin analysis.** The test set is binned by dex distance to the nearest label flip. Within 0.25 dex of the cutoff (8.8% of the set) the classifier scores 61.9% against a 64.4% majority baseline — i.e. it recovers *no* usable information there — and its mean predicted probability is 0.515, meaning it correctly reports its own uncertainty. At ≥ 1 dex from the cutoff (64.8% of the set) it reaches 95.3%. This is the reviewer's objection, measured: near-threshold labels separate atmospheres that are physically near-identical. *Evidence:* [`analyze_label_margin.py`](../analyze_label_margin.py) → [`H2_label_margin.txt`](../final_results/H2_label_margin.txt) ([csv](../final_results/H2_label_margin.csv)), figure [`label_margin.png`](../final_results/label_margin.png).

**Margin analysis** ([`H2_label_margin.txt`](../final_results/H2_label_margin.txt)) — accuracy by distance to the labelling cutoff, each bin scored against its own majority-class baseline:

| Distance to cutoff (dex) | n | Accuracy | Majority baseline | Gain | Mean predicted prob. |
| :-- | --: | --: | --: | --: | --: |
| 0 – 0.25 | 236 | 61.9% | 64.4% | **−2.5** | 0.515 |
| 0.25 – 0.5 | 258 | 76.0% | 67.1% | +8.9 | 0.586 |
| 0.5 – 1 | 454 | 85.9% | 62.1% | +23.8 | 0.593 |
| 1 – 2 | 908 | 94.8% | 50.2% | +44.6 | 0.521 |
| > 2 | 841 | 95.7% | 65.0% | +30.7 | 0.388 |

![Classification accuracy versus distance from the labelling threshold](../final_results/label_margin.png)

**Threshold sensitivity** ([`H2_threshold_sensitivity.txt`](../final_results/H2_threshold_sensitivity.txt)) — move both cutoffs together, relabel, retrain from scratch:

| Shift (dex) | CH₄ / O₃ cutoff | Accuracy | Majority baseline | Gain over baseline |
| :-- | :-- | --: | --: | --: |
| −0.50 | −6.50 / −7.50 | 87.0% | 56.6% | +30.4 |
| −0.25 | −6.25 / −7.25 | 88.2% | 53.4% | +34.7 |
| 0 (original) | −6.00 / −7.00 | 88.9% | 50.3% | +38.6 |
| +0.25 | −5.75 / −6.75 | 88.7% | 55.3% | +33.4 |
| +0.50 | −5.50 / −6.50 | 87.8% | 61.7% | +26.1 |

**Status.** *Done:* the task is renamed from "biosignature detection" to "recovery of an abundance-threshold labelling" — this rename **is** the concession. The two analyses are not a defence of the labelling rule; they measure how far it governs the result: threshold sensitivity shows accuracy stays within 2 points as both cutoffs move a full dex (so the headline is not an artefact of the chosen cutoff), and the margin analysis shows that within 0.25 dex of the cutoff the classifier beats no majority baseline (those labels are genuinely ambiguous). *Not done:* retrieval-derived labels — the one experiment he explicitly asks for here.

**For your input.** Retrieval on the full benchmark is months of compute. A middle option is to run retrieval on a stratified sample of 50–100 spectra and report how often the retrieval-based label differs from the injected one. Attempt that this cycle, or concede retrieval as the primary next step?

### R1-3 to R1-5 — the PCA feature space (one connected finding)

Reviewer 1's next three comments are one question — *what is the PCA feature space actually doing?* — approached from three sides: the mechanistic interpretation (R1-3), whether whitening amplifies chemistry (R1-4), and the variance-ordering inconsistency (R1-5). They are answered by one set of **eight experiments**, which together cover all four analyses he requested under R1-3 and all four ablations he requested under R1-5. Each comment is responded to individually below; the shared finding and the proposed replacement text are stated once here.

**Proposed replacement for the withdrawn interpretation** (every clause is backed by one of the eight experiments):

> No principal component isolates the biosignature signal. Across all 102 components the largest single-component discriminative AUC is 0.66, and the two components carrying 98.4% of the spectral variance classify at chance and can be removed with no loss of accuracy — so the variance ordering is not an ordering of biosignature information. The molecular signals are not isolated in individual components either: no component is dominated by a single molecule (the component most associated with methane is only about 5% methane by variance), and each molecule's influence is spread across thirty to forty components and partly entangled with the overall-depth component. Classification succeeds only by nonlinearly combining many individually weak components — a linear classifier or a supervised projection on the same features reaches 68–74%, against 89% for the gradient-boosted trees. The whitening applied before the neural networks is therefore an optimization step for those scale-sensitive models, not a chemistry-selective operation: the tree ensemble is invariant to it and is the most accurate model, and no supervised re-weighting of the components improves on it.

**The eight experiments behind it:**

| Experiment | What it shows | Evidence |
| :-- | :-- | :-- |
| Per-component discriminative power | no component discriminative (max AUC 0.66); variance rank ≠ discriminative rank | [txt](../final_results/H2_pc_discriminative_power.txt), [fig](../final_results/pc_discriminative_power.png) |
| Selective component removal | PC0+PC1 (98.4% variance) classify at chance; removing them is free | [txt](../final_results/H2_pc_range_ablation.txt) |
| CH₄/O₃ projection | molecular imprints not isolated; entangled with the overall-depth component | [txt](../final_results/H2_chem_projection.txt), [fig](../final_results/chem_projection.png) |
| Loading-vector analysis | interpretable loading shapes (depth, slope, molecular bands), but weak | [txt](../final_results/H2_pc_drivers.txt), [fig](../final_results/pc_loadings.png) |
| Variance decomposition by parameter | each molecule spreads over 30–40 components; ≤ 5% of its own primary one | [txt](../final_results/H2_pc_drivers.txt) |
| Supervised dimensionality reduction | PCA not privileged; linear methods cap at 68–74% vs 89% (task is nonlinear) | [txt](../final_results/H2_supervised_dr.txt) |
| Whitening on / off | XGBoost invariant; MLP substitutable; CNN needs it | [txt](../final_results/H2_whitening_necessity.txt) |
| Alternative feature weighting | no weighting beats whitening; supervised re-weighting does not help | [txt](../final_results/H2_feature_weighting.txt) |

**One reconciliation to keep straight** (or a careful reviewer flags an apparent contradiction): the ozone results are viewed two ways. The variance decomposition finds ozone abundance drives a dedicated low-variance component (PC4); the projection finds the ozone spectral imprint also overlaps the overall-depth component (PC0), because ozone shifts the continuum level. Both are true and both say the same thing — ozone is not cleanly isolated.

### R1-3 — The PCA interpretation is not supported by the mathematics

**Reviewer 1 wrote:**

> I also have substantial concerns regarding the interpretation of the PCA representation. The manuscript repeatedly argues that the first principal components primarily encode broad physical structure while higher-order components encode chemically informative absorption features. However, <span style="color:#0b5cad">**this interpretation is not supported by the mathematics of principal component analysis**</span>. Principal components are orthogonal linear combinations of all original wavelength variables. They are not independent physical variables, nor do they uniquely correspond to individual atmospheric processes. Every principal component generally contains different weighted mixtures of continuum structure, planetary radius, atmospheric temperature, molecular absorption, scattering, clouds, and measurement noise. Consequently, methane and ozone information is expected to be distributed across multiple principal components rather than residing exclusively within higher-order components. Demonstrating that PC0 correlates strongly with mean transit depth does not imply that chemically relevant information is absent from that component, nor does it establish that chemistry has been isolated into PCs 2–101. These interpretations require considerably stronger evidence than presently provided. The manuscript should include <span style="color:#1a7f37">**loading-vector analyses ✅**</span>, <span style="color:#1a7f37">**variance decomposition by physical parameter ✅**</span>, <span style="color:#1a7f37">**reconstruction studies after selectively removing principal components ✅**</span>, and <span style="color:#1a7f37">**explicit quantification of how methane and ozone absorption features project onto individual principal components ✅**</span> before drawing these mechanistic conclusions.

**What it means.** The paper's claim that leading components hold "physics" and later ones hold "chemistry" is unjustified, because each principal component mixes all wavelengths.

**What we did — run all four of his analyses, withdraw the old claim, replace it with the finding above.** We do not defend the old interpretation; we withdraw it. Rather than leave the feature space uncharacterised, we ran all four analyses he requested (plus a replacement for the Pearson correlation he rejected). They refute the old attribution and support the replacement claim stated in the preamble above.

His four requested analyses:

- **Reconstruction / selective component removal.** PC0+PC1 alone classify at 52.1% (chance); removing them costs nothing (88.6% → 88.3%); at matched dimensionality the low-variance components beat the high-variance ones (PCs 2–51 at 86.2% vs PCs 0–49 at 85.7%). *Evidence:* [`ablate_pc_ranges.py`](../ablate_pc_ranges.py) → [`H2_pc_range_ablation.txt`](../final_results/H2_pc_range_ablation.txt).
- **CH₄/O₃ projection onto components.** No molecule's imprint is isolated in a clean component: the ozone imprint overlaps the overall-depth component (PC0) and the methane imprint sits in low-variance components, both entangled with the continuum. *Evidence:* [`analyze_chem_projection.py`](../analyze_chem_projection.py) → [`H2_chem_projection.txt`](../final_results/H2_chem_projection.txt), figure [`chem_projection.png`](../final_results/chem_projection.png).
- **Loading-vector analysis.** The leading loadings have interpretable shapes (PC0 flat / overall depth, PC1 spectral slope, low-variance components resembling molecular bands) — but the association is weak, not a clean assignment. *Evidence:* [`analyze_pc_physical_drivers.py`](../analyze_pc_physical_drivers.py) → [`H2_pc_drivers.txt`](../final_results/H2_pc_drivers.txt), figure [`pc_loadings.png`](../final_results/pc_loadings.png).
- **Variance decomposition by parameter.** Each active molecule's influence spreads across 30–40 components (to reach 90% of its footprint), and even the component most associated with a molecule is only ~5–6% explained by it — so no component isolates a molecule. *Evidence:* [`analyze_pc_physical_drivers.py`](../analyze_pc_physical_drivers.py) → [`H2_pc_drivers.txt`](../final_results/H2_pc_drivers.txt).

Replacing the Pearson correlation he rejected:

- **Per-component discriminative power.** No component is strongly discriminative — the maximum single-feature AUC across all 102 is 0.66 (at PC9); PC0 = 0.51 (chance); 14 of the 20 most discriminative components fall outside the 20 highest-variance ones. *Evidence:* [`analyze_pc_discriminative_power.py`](../analyze_pc_discriminative_power.py) → [`H2_pc_discriminative_power.txt`](../final_results/H2_pc_discriminative_power.txt) ([csv](../final_results/H2_pc_discriminative_power.csv)), figure [`pc_discriminative_power.png`](../final_results/pc_discriminative_power.png).

Component-removal results ([`H2_pc_range_ablation.txt`](../final_results/H2_pc_range_ablation.txt)):

| Components kept | % of variance | XGBoost accuracy |
| :-- | --: | --: |
| All 102 | ~100% | 88.6% |
| Drop the two highest-variance (PCs 2–101) | 1.6% | 88.3% |
| The two highest-variance only (PC0–PC1) | 98.4% | **52.1% (chance)** |
| First 50 components (PCs 0–49) | ~100% | 85.7% |
| Low-variance 50 (PCs 2–51) | 1.6% | **86.2%** |

Per-component discriminative power — no single component is strong, and the highest-variance ones (shaded) sit at chance:

![Single-feature AUC, mutual information, and explained variance for each of the 102 principal components](../final_results/pc_discriminative_power.png)

**Replacement claim:** the tightened claim in the R1-3–R1-5 preamble above — no component isolates the biosignature signal; molecular information is distributed, not isolated; classification is a nonlinear aggregation of many weak components. Measured across the eight experiments, not asserted.

**Status.** *Withdrawn:* the manuscript's claim that PC0/PC1 encode physical structure while components 2–101 encode the chemical absorption features. *Replacement:* the preamble finding above. *All four of his requested analyses are now done* — the first round ran only reconstruction/removal; the loading-vector analysis, variance decomposition by parameter, and the CH₄/O₃ projection have since been added, and all three confirm that no component isolates a molecule.

**For your input.** (1) Is withdrawing-and-replacing acceptable, or would you prefer we run his full set of four? (2) Keep the replacement claim, or simply retract the old claim without asserting a new one?

### R1-4 — Whitening does not selectively amplify chemistry

**Reviewer 1 wrote:**

> This concern becomes even more significant following the whitening procedure. After PCA, every retained principal component is standardized to unit variance before training the neural network models. While this may improve optimization for gradient-based learning, it fundamentally changes the geometry of the feature space. Whitening does not selectively amplify chemically informative features. Instead, it amplifies every low-variance principal component equally, regardless of whether that component contains molecular absorption features, continuum information, numerical artifacts, simulator-specific structure, or measurement noise. Since each principal component is itself a linear combination of all original wavelengths, whitening rescales mixtures of physical and chemical information rather than independently equalizing "physical" and "chemical" features. Consequently, <span style="color:#0b5cad">**the mechanistic explanation presented in the manuscript—that whitening specifically increases the influence of chemically informative components—is not mathematically demonstrated**</span>.

**What it means.** The paper says whitening boosts chemically informative components; whitening in fact rescales every low-variance component equally regardless of content, so it cannot be selectively boosting chemistry.

**What we did — concede, and show the concession costs nothing.** We concede the claim: whitening does not selectively amplify chemistry, and we stop describing it that way (the "chemically relevant" language is removed). The experiment below is not a defence of the old claim — it establishes what whitening *actually* does and shows the concession does not weaken the paper. It crosses component range × whitening on/off for each model:

- **XGBoost is provably unaffected** — identical metrics with and without whitening at every range tested (as expected: tree splits are invariant to per-feature rescaling).
- **For the MLP whitening is substitutable** — dropping the two highest-variance components and omitting whitening entirely gives 79.5% ± 2.2, against 78.8% ± 2.2 for the whitened full-component pipeline (statistically the same).
- **Only the CNN still needs it** (≈ 76% whitened vs ≈ 67% unwhitened).

The key point for the paper: the *recommended* model, XGBoost, uses no whitening at all, so this concern does not reach the headline result. *Evidence:* [`test_whitening_necessity.py`](../test_whitening_necessity.py) → [`H2_whitening_necessity.txt`](../final_results/H2_whitening_necessity.txt).

**Whitening on / off, per model** ([`H2_whitening_necessity.txt`](../final_results/H2_whitening_necessity.txt)):

| Model | Whitened (all 102 components) | Unwhitened | Effect of whitening |
| :-- | --: | --: | :-- |
| XGBoost | 88.2% | 88.2% (identical, every range) | none — provably unaffected |
| MLP | 78.8% ± 2.2 | 79.5% ± 2.2 (drop PC0–PC1, no whitening) | substitutable |
| CNN | 75.8% ± 1.4 | 67.2% ± 3.2 | still required |

**Status.** *Conceded:* the claim that whitening amplifies chemistry, and the "chemically relevant" wording, are removed. *Shown by the experiment (not a defence of the old claim):* whitening is a plain optimisation step — XGBoost is identical with and without it, the MLP is equally good without it, only the CNN needs it — and the recommended model, XGBoost, uses none, so the concern does not touch the headline result.

**For your input.** None — we believe the concession fully covers it.

### R1-5 — Conceptual inconsistency in the variance-ordering rationale

**Reviewer 1 wrote:**

> More fundamentally, <span style="color:#0b5cad">**the manuscript appears to contain a conceptual inconsistency**</span>. It argues that PCA naturally orders the data such that dominant physical information resides in the first principal components while chemically relevant information occupies lower-variance components. However, the subsequent whitening operation deliberately removes precisely this variance hierarchy by forcing every retained component to have identical variance. If the PCA variance ordering reflects meaningful atmospheric physics, it is unclear why the subsequent learning algorithm should deliberately eliminate that ordering. Conversely, if equalizing all component variances improves classification, this suggests that explained variance itself may not correspond to discriminative importance for biosignature detection. These two interpretations cannot simultaneously serve as the mechanistic explanation for the reported performance. The manuscript should explicitly resolve this conceptual inconsistency through additional ablation studies comparing <span style="color:#1a7f37">**whitening ✅**</span>, <span style="color:#1a7f37">**removal of leading principal components ✅**</span>, <span style="color:#1a7f37">**supervised dimensionality reduction techniques such as Linear Discriminant Analysis or Partial Least Squares ✅**</span>, and <span style="color:#1a7f37">**alternative feature weighting strategies ✅**</span>.

**What it means.** The paper can't both claim the variance ordering is physically meaningful and then whiten it away. He wants ablations, including supervised dimensionality reduction, to establish what is really happening.

**What we did.** We adopt his own resolution: explained variance is unsupervised and discriminative power is supervised, so there is no contradiction once the claim that the ordering is physically meaningful is dropped (which R1-3 does). All four of his requested ablations are done:

- **Whitening on/off** — [`test_whitening_necessity.py`](../test_whitening_necessity.py) (see R1-4).
- **Removal of leading components** — [`ablate_pc_ranges.py`](../ablate_pc_ranges.py) (see R1-3).
- **Supervised dimensionality reduction** — PLS and LDA compared against PCA at matched component counts, holding the classifier fixed. At two components a label-informed projection beats the two highest-variance principal components by 8.6 points (60.7% vs 52.1%); beyond about five components the two become indistinguishable, and at 102 they differ by 0.3. The linear methods cap at 73.9% (PLS-DA) and 68.2% (LDA), well below 88.9% — so PCA is an adequate basis, not a transformation that isolates anything, and the task is substantially non-linear. *Evidence:* [`supervised_dr_comparison.py`](../supervised_dr_comparison.py) → [`H2_supervised_dr.txt`](../final_results/H2_supervised_dr.txt) ([csv](../final_results/H2_supervised_dr.csv)).
- **Alternative feature weighting** — four weightings of the 102 components, classifier fixed. XGBoost is invariant to all of them (scale-invariant); for the MLP, whitening is not beaten by weighting each component by its discriminative power, and emphasising the high-variance components collapses it. So no weighting isolates a better signal — whitening's role is to stop the two high-variance, non-discriminative components from dominating. *Evidence:* [`compare_feature_weighting.py`](../compare_feature_weighting.py) → [`H2_feature_weighting.txt`](../final_results/H2_feature_weighting.txt).

**Supervised vs unsupervised projection** at matched dimensionality, classifier held fixed ([`H2_supervised_dr.txt`](../final_results/H2_supervised_dr.txt)):

| Components | PCA + XGBoost | PLS + XGBoost (supervised) | Difference |
| --: | --: | --: | --: |
| 2 | 52.1% | 60.7% | **+8.6** |
| 5 | 72.7% | 72.2% | −0.5 |
| 20 | 81.9% | 83.2% | +1.3 |
| 102 | 88.9% | 88.6% | −0.3 |

Linear classifiers on the same features cap far lower — PLS-DA 73.9%, LDA 68.2% against 88.9% for the nonlinear ensemble — so PCA is an adequate basis, not a transformation that isolates anything, and the task is substantially non-linear.

**Feature-weighting schemes** on the 102 components, classifier fixed ([`H2_feature_weighting.txt`](../final_results/H2_feature_weighting.txt)):

| Weighting | XGBoost | MLP (3 restarts) |
| :-- | --: | --: |
| None (raw scores) | 88.0% | 76.7% |
| Whitening (unit variance) | 87.9% | 78.3% |
| Supervised (by discriminative power) | 88.3% | 76.0% |
| Emphasise high-variance components | 87.9% | **59.4%** |

**Status.** *Done — all four requested ablations:* whitening on/off, removal of leading principal components, supervised dimensionality reduction (PLS and LDA), and alternative feature weighting. Together they resolve the inconsistency: explained variance is unsupervised and does not track discriminative importance, so whitening away the variance ordering costs nothing and no weighting of it isolates a better signal.

**For your input.** None outstanding for this comment.

### R1-6 — Whitening may reduce robustness on real data

**Reviewer 1 wrote:**

> An equally important concern is that <span style="color:#0b5cad">**the whitening procedure may substantially reduce robustness to real observational data**</span> even if it improves performance within the synthetic simulation environment. Within TauREx, whitening appears to enhance classification by increasing the influence of low-variance components that contain discriminative methane and ozone information. However, because the entire pipeline is trained and evaluated on spectra generated by the same simulator, there is no evidence that these low-variance components represent exclusively chemically informative features. In real telescope observations, low-variance principal components are also expected to contain detector artifacts, calibration residuals, correlated instrumental noise, stellar contamination, imperfect molecular opacity models, and other observational effects. Whitening cannot distinguish chemically meaningful variance from these alternative sources of variability; it amplifies all low-variance components equally. Consequently, a preprocessing strategy that improves classification by amplifying subtle methane and ozone signatures in synthetic data may equally amplify observational noise and instrumental artifacts when deployed on real Ariel spectra. Addressing this issue requires <span style="color:#9a6700">**experiments on independent simulation environments and, where possible, observational spectra ⚠️**</span> to determine whether whitening genuinely improves out-of-distribution generalization rather than simply increasing performance within the synthetic training distribution.

**What it means.** On real data the low-variance components also carry instrument noise and artifacts; since whitening amplifies all of them equally, it may amplify noise as much as signal. He wants this tested on independent environments and, if possible, real spectra.

**What we did.** A controlled comparison in which two otherwise identical neural networks differ *only* in whether whitening is applied, each trained five times to average out initialisation, with the difference measured paired within each restart. Whitening gives a small advantage on clean data (+3.6 ± 3.1 points) but a larger penalty under perturbation (−11.0 ± 1.7 at effective SNR 5); the two curves cross by SNR 12 in four of the five restarts. Normalised by each model's headroom above chance, whitening degrades faster in five of six perturbation families. This confirms his hypothesis, and it reinforces the recommendation of XGBoost, which uses no whitening and is the most robust model in every family tested. *Evidence:* [`domain_shift_mlp_restarts.py`](../domain_shift_mlp_restarts.py) → [`H2_whitening_restarts.txt`](../final_results/H2_whitening_restarts.txt) ([csv](../final_results/H2_whitening_restarts.csv)); the MLP run-to-run scatter that justifies the five-restart averaging is characterised in [`measure_mlp_reproducibility.py`](../measure_mlp_reproducibility.py) → [`H2_mlp_reproducibility.txt`](../final_results/H2_mlp_reproducibility.txt).

**Whitened vs unwhitened MLP** under white noise, means over 5 restarts, differences paired within restart ([`H2_whitening_restarts.txt`](../final_results/H2_whitening_restarts.txt)):

| White noise | Whitened | Unwhitened | Whitened − Unwhitened |
| :-- | --: | --: | --: |
| SNR 15 (clean) | 79.3% | 75.7% | **+3.6** |
| SNR 12 | 69.1% | 74.7% | −5.6 ← curves cross |
| SNR 10 | 64.9% | 73.8% | −8.9 |
| SNR 5 | 57.0% | 68.0% | **−11.0** |

**Caveat we state plainly.** This evidence comes from perturbing spectra *within* the same simulator. It supports his hypothesis but is not the independent-environment / observational-spectra test he specifically named.

**Status.** *Done:* a controlled whitened-vs-unwhitened comparison (five restarts, paired), which confirms his hypothesis — whitening helps on clean data and hurts under noise. *Not done as he specified:* his requested test was on independent simulation environments and, where possible, real spectra; ours is within-simulator perturbations only.

**For your input.** Is the within-simulator evidence acceptable with the caveat stated, or would you want the whitening-robustness comparison re-run on the independent Exo-Transmit spectra we already generated for R1-1?

---

## Reviewer 2

### R2-1 — Verify reference links

**Reviewer 2 wrote:**

> Please verify that all links to the references point to the correct source.

**What we did / status.** In progress — auditing all 42 references (only four currently carry URLs). The public repository linked from the Data Availability statement has been corrected to report the same headline numbers and resolution as the manuscript. **For your input:** none.

### R2-2 — Acknowledgments and disclosure of assistance

**Reviewer 2 wrote:**

> The authors must disclose and acknowledge any assistance received in the preparation of this manuscript, including but not limited to editorial, technical, analytical, or writing support. All such contributions must be clearly stated in the Acknowledgments section.

**What we did / status.** The manuscript has no Acknowledgments section; one will be added. **For your input:** I need the list of contributions to disclose — any editorial, technical, analytical, or writing help, and any AI tools used in analysis or drafting. Can you confirm what to acknowledge?

### R2-3 — Reference quality

**Reviewer 2 wrote:**

> Include enough recent references along with foundational ones. Ensure references directly support your claims. Avoid "padding." Use credible sources (peer-reviewed journals, reputable books, official reports).

**What we did / status.** Two bare, undated web references will be replaced and the rest audited for direct support. The opacity data used to generate every spectrum was previously uncited; the required citations have been identified (Freedman et al. 2008 & 2014 and Lupu et al. 2014 for the base tables; Chubb et al. 2021, Polyansky et al. 2018, Yurchenko et al. 2017 & 2020 for the ExoMol line lists used in the new experiments; Kempton et al. 2017 for Exo-Transmit), all verified against DOI records. **For your input:** any specific references you want added or removed.

### R2-4 — State all assumptions explicitly

**Reviewer 2 wrote:**

> All assumptions (including implicit assumptions) must be explicitly stated and clearly justified. The authors should explain why each assumption is reasonable and discuss its impact on the results and conclusions.

**What we did / status.** An Assumptions subsection is planned, listing each with justification and likely effect: 1D atmospheres; Gaussian noise at fixed SNR 15; uniform R = 200; the fixed CH₄/O₃ thresholds; enforced 50/50 class balance; H₂-dominated composition; and the single molecular opacity compilation. Several of these connect directly to the R1-1 experiments (which now quantify their effect). **For your input:** none.

### R2-5 — Avoid overreaching conclusions

**Reviewer 2 wrote:**

> Avoid overreaching conclusions that extend beyond what is supported by the data and analysis.

**What we did / status.** The same scope reduction as the Overview — presenting the work as a synthetic-data benchmark and the task as label recovery. This is the second reviewer independently flagging overreach. **For your input:** none beyond the title question in the Overview.

### R2-6 — Tense and person

**Reviewer 2 wrote:**

> Verify that you have used past perfect tence [sic] and third person throughout the manuscript wherever applicable.

**What we did / status.** A consistency pass converting the current first-person plural to third person and checking tense throughout. **For your input:** none.
