# Response to Reviewers

*A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra* — Journal of High School Science, revise and resubmit.

Each comment below is given in the reviewer's own words, followed by what it asks for, what we changed in response, and — where a decision is still open — a question for you. Related comments that the reviewer raised as one continuous argument are grouped together.

---

## Reviewer 1

### Overview — opening assessment and overall recommendation

**Reviewer 1 wrote (opening):**

> The manuscript presents a technically well-executed machine learning benchmark and the comparison among XGBoost, Random Forest, MLP, and CNN is carefully conducted. The statistical analyses are generally appropriate, the experimental pipeline is well documented, and the manuscript is clearly written. However, I believe the central scientific claims are presently insufficiently supported. The primary issue is not the implementation of the machine learning algorithms themselves but rather whether the experiments actually demonstrate what the manuscript claims. In its current form, the work demonstrates that an XGBoost classifier can successfully recover predefined biosignature labels from spectra generated within a single synthetic simulation framework. It does not yet demonstrate that the proposed methodology identifies biosignatures in actual exoplanet transmission spectra or that the mechanistic interpretation of the PCA representation is correct. These issues require additional experimental validation before the manuscript is suitable for publication.

**Reviewer 1 wrote (recommendation):**

> Overall, I believe the manuscript contains a promising machine learning framework and an extensive computational evaluation. However, the central scientific questions remain unresolved. Specifically, the manuscript has not yet demonstrated that the classifier generalizes beyond the synthetic TauREx environment, that the PCA interpretation accurately reflects the underlying atmospheric physics, or that the whitening procedure preferentially enhances chemically meaningful information rather than simulator-specific low-variance structure. These are not matters of presentation but of experimental validation. Accordingly, I recommend major revision.

**What it means.** The benchmark and statistics are considered sound; the objection is that the paper claims more than it shows — real biosignature detection and a physical reading of the PCA — when the experiments establish neither. The remedy he asks for is additional experiments plus narrower claims.

**What we did.** We narrowed the scope throughout: the work is now presented as a controlled benchmark of model families, calibration and robustness on synthetic Ariel-like spectra, and the task as recovering an abundance-threshold labelling rather than detecting biosignatures. We added the validation experiments requested (R1-1), withdrew the PCA interpretation and replaced it with a directly measured result (R1-3), and completed the whitening ablations (R1-4, R1-5, R1-6). The specifics are in the comments below.

**For your input.** The title currently reads "…in Exoplanet Transmission Spectra." We can either insert "Synthetic" before "Exoplanet," or leave the title and carry the scoping in the first and last sentences of the abstract. Which do you prefer?

### R1-1 — Confined to a single simulator; robustness under domain shift

**Reviewer 1 wrote:**

> The most significant concern is that the entire study is developed, trained, validated, and tested exclusively within a single synthetic ecosystem. Every transmission spectrum is generated using the TauREx/MultiREx forward model, every class label is derived directly from the same simulated atmospheric parameters, and every evaluation is performed on additional spectra generated under essentially identical modeling assumptions. Consequently, the classifier is never evaluated outside the statistical distribution of the simulator that created both the inputs and the labels. High predictive accuracy under these conditions demonstrates internal consistency within the simulation environment but does not establish that the learned decision boundary represents genuine atmospheric physics rather than simulator-specific statistical regularities. The central question for a biosignature classifier is not whether it generalizes to additional TauREx simulations, but whether it generalizes to real exoplanet observations. The manuscript does not presently address this question.
>
> This distinction is especially important because real Ariel observations will differ substantially from the synthetic spectra used in this work. Ariel will observe heterogeneous transmission spectra containing instrumental systematics, wavelength-dependent detector effects, correlated noise, stellar contamination, imperfect calibration, incomplete molecular opacity databases, cloud and haze variability, photochemical complexity, and three-dimensional atmospheric structure. In contrast, the present study assumes one-dimensional atmospheres, Gaussian noise with a fixed signal-to-noise ratio, predefined atmospheric compositions, and idealized observing conditions. These assumptions substantially reduce the complexity of the classification problem. Before this work can be considered applicable to Ariel, the authors should demonstrate that the classifier remains robust under realistic domain shift. At minimum, the manuscript should include experiments using spectra generated by independent radiative transfer codes, alternative molecular opacity databases, different cloud and haze prescriptions, different stellar contamination models, varying spectral resolutions and signal-to-noise ratios, more realistic instrumental systematics, and domain-shift or transfer-learning experiments.

**What it means.** Everything — training, tuning, evaluation — happens inside one simulator, so high accuracy could reflect quirks of that simulator rather than real physics. Before the method can be called Ariel-applicable, it has to be shown robust when the data is generated differently, across seven specific axes. (His clause about labels being derived from the generation parameters is a separate concern, addressed under R1-2.)

**What we did.** We evaluated the trained pipeline on spectra regenerated for the same test planets under different physics, holding the model fixed. Of the seven requested axes, four are covered in full:

- **Independent radiative transfer code.** Re-rendering the test planets with Exo-Transmit, an independently developed code, lowers accuracy from 88.9% to 84.8% (−4.1 points). The two codes agree closely on the spectra themselves (median per-planet correlation 0.997).
- **Alternative molecular opacity data.** Substituting ExoMol line lists for H₂O, CH₄ and CO₂ lowers accuracy to 72.8% (−16.1); additionally replacing ozone with HITRAN lowers it to 66.0% (−22.9).
- **Cloud and haze prescriptions.** An untrained-for grey cloud deck and a Lee–Mie photochemical haze were each added to the forward model. Accuracy degrades smoothly as the aerosol thickens — for example a cloud deck at 10⁴ Pa costs 10.5 points — and the results bound the method's applicability to atmospheres whose aerosols leave most of the spectral feature amplitude intact.
- **Varying resolution and signal-to-noise.** A sweep degrading resolution (R 200→75) and noise (SNR 15→5, white and correlated) shows graceful degradation, with correlated noise the most damaging (−13 points at effective SNR 10).

Training stays entirely within TauREx; only evaluation leaves it, which is the reviewer's stated remedy. Three axes are partial: stellar contamination is represented by a single wavelength-dependent proxy rather than a physical spot model; the instrumental systematics are plausible parametric forms rather than Ariel's published noise model; and transfer learning (as opposed to domain shift) was not run. These are stated as limitations and named as next steps.

**For your input.** Three of the seven axes are partial (physical stellar-spot contamination model; systematics derived from Ariel's instrument model; transfer learning). Our plan is to present the four completed axes and name these three as next steps in the limitations. Is that sufficient for this cycle, or would you want any of the three attempted before resubmission?

### R1-2 — The positive class is a labelling rule, not a detection

**Reviewer 1 wrote:**

> A related concern is the definition of the positive class. The manuscript defines biosignatures using fixed abundance thresholds (CH₄ > 10⁻⁶ and O₃ > 10⁻⁷), and every atmosphere satisfying these thresholds is labeled as positive. Consequently, the classifier is not discovering biosignatures but rather learning a predefined labeling rule imposed during simulation. This distinction is important because Ariel itself does not operate by measuring atmospheric abundances and applying deterministic thresholds. Ariel measures transmission spectra, after which atmospheric abundances must be inferred through retrieval algorithms that incorporate observational uncertainties, parameter degeneracies, stellar context, and competing abiotic explanations. The proposed machine learning framework therefore learns to reproduce a synthetic labeling convention rather than the operational biosignature identification process that Ariel will ultimately employ. Additional experiments using retrieval-derived abundances with realistic uncertainties or probabilistic labels would substantially strengthen the relevance of the proposed framework.

**What it means.** The labels are computed from the same abundances used to generate the spectra, so the classifier is recovering a labelling convention, not inferring a detection the way a real observation would require. Retrieval-derived or probabilistic labels would make the task closer to the real one.

**What we did.** We concede the point and rename the task throughout the paper — from "biosignature detection" to recovery of an abundance-threshold labelling — so the manuscript no longer claims more than the experiment shows. Two analyses characterise how much the labelling convention governs the result: a threshold-sensitivity test showing overall accuracy changes by under two points when both cutoffs are moved by up to half a dex, and a margin analysis showing that for planets within a quarter-dex of the cutoff the classifier recovers no information beyond a majority-class baseline (the label there is genuinely ambiguous). Making the labels observationally meaningful would require retrieval, which we have not done this cycle.

**For your input.** Retrieval-derived labels are the one experiment the reviewer asks for here that we have not attempted; on the full benchmark it is months of compute. A middle option is to run retrieval on a stratified sample of 50–100 spectra and report how often the retrieval-based label differs from the injected one. Do you want us to attempt that this cycle, or to concede it as the primary next step?

### R1-3 — The PCA interpretation is not supported by the mathematics

**Reviewer 1 wrote:**

> I also have substantial concerns regarding the interpretation of the PCA representation. The manuscript repeatedly argues that the first principal components primarily encode broad physical structure while higher-order components encode chemically informative absorption features. However, this interpretation is not supported by the mathematics of principal component analysis. Principal components are orthogonal linear combinations of all original wavelength variables. They are not independent physical variables, nor do they uniquely correspond to individual atmospheric processes. Every principal component generally contains different weighted mixtures of continuum structure, planetary radius, atmospheric temperature, molecular absorption, scattering, clouds, and measurement noise. Consequently, methane and ozone information is expected to be distributed across multiple principal components rather than residing exclusively within higher-order components. Demonstrating that PC0 correlates strongly with mean transit depth does not imply that chemically relevant information is absent from that component, nor does it establish that chemistry has been isolated into PCs 2–101. These interpretations require considerably stronger evidence than presently provided. The manuscript should include loading-vector analyses, variance decomposition by physical parameter, reconstruction studies after selectively removing principal components, and explicit quantification of how methane and ozone absorption features project onto individual principal components before drawing these mechanistic conclusions.

**What it means.** The paper's claim that the leading components hold "physics" and later components hold "chemistry" is not justified, because each principal component is a mixture of all wavelengths and does not map onto a single physical process.

**What we did.** We withdraw the interpretation rather than attempt to defend it, and replace it with a claim that is directly measurable: no individual principal component is strongly discriminative (the strongest single-component AUC is 0.66), the two components carrying 98% of the variance are among the least informative, and classification performance comes from aggregating many weakly informative components. A selective-removal experiment supports this — discarding the two highest-variance components costs nothing, and at matched dimensionality the low-variance components out-perform the high-variance ones.

**For your input.** The reviewer lists four analyses (loading vectors, variance decomposition by parameter, reconstruction after removal, and CH₄/O₃ projection). We ran the reconstruction/removal one; we did not run the other three, because they would support the physical interpretation we are withdrawing. Two questions: (1) is withdrawing-and-replacing acceptable to you, or would you prefer we run the full set? (2) Are you comfortable keeping the replacement claim ("performance arises from aggregating many weakly informative components"), or would you rather we simply retract the old claim without asserting a new one?

### R1-4 — Whitening does not selectively amplify chemistry

**Reviewer 1 wrote:**

> This concern becomes even more significant following the whitening procedure. After PCA, every retained principal component is standardized to unit variance before training the neural network models. While this may improve optimization for gradient-based learning, it fundamentally changes the geometry of the feature space. Whitening does not selectively amplify chemically informative features. Instead, it amplifies every low-variance principal component equally, regardless of whether that component contains molecular absorption features, continuum information, numerical artifacts, simulator-specific structure, or measurement noise. Since each principal component is itself a linear combination of all original wavelengths, whitening rescales mixtures of physical and chemical information rather than independently equalizing "physical" and "chemical" features. Consequently, the mechanistic explanation presented in the manuscript—that whitening specifically increases the influence of chemically informative components—is not mathematically demonstrated.

**What it means.** The paper says whitening boosts the chemically informative components; the reviewer points out that whitening rescales every low-variance component equally regardless of content, so it cannot be selectively boosting chemistry.

**What we did.** We concede this and remove the "chemically relevant" framing. Whitening is now described only as an optimisation step for the gradient-based models, with no claim about what it selects for. The recommended model, XGBoost, uses no whitening at all — it returns identical results with and without it — so this concern does not affect the headline result. We also show the step is substitutable: dropping the two highest-variance components without any whitening gives the neural network equivalent performance.

**For your input.** None — we believe the concession fully addresses this.

### R1-5 — Conceptual inconsistency in the variance-ordering rationale

**Reviewer 1 wrote:**

> More fundamentally, the manuscript appears to contain a conceptual inconsistency. It argues that PCA naturally orders the data such that dominant physical information resides in the first principal components while chemically relevant information occupies lower-variance components. However, the subsequent whitening operation deliberately removes precisely this variance hierarchy by forcing every retained component to have identical variance. If the PCA variance ordering reflects meaningful atmospheric physics, it is unclear why the subsequent learning algorithm should deliberately eliminate that ordering. Conversely, if equalizing all component variances improves classification, this suggests that explained variance itself may not correspond to discriminative importance for biosignature detection. These two interpretations cannot simultaneously serve as the mechanistic explanation for the reported performance. The manuscript should explicitly resolve this conceptual inconsistency through additional ablation studies comparing whitening, removal of leading principal components, supervised dimensionality reduction techniques such as Linear Discriminant Analysis or Partial Least Squares, and alternative feature weighting strategies.

**What it means.** The paper can't both claim the variance ordering is physically meaningful and then use whitening to destroy that ordering. The reviewer wants ablations to determine what is really going on, including supervised dimensionality reduction.

**What we did.** We adopt the reviewer's own resolution: explained variance is unsupervised and discriminative power is supervised, so there is no contradiction once the claim that the ordering is physically meaningful is dropped (which R1-3 does). We ran three of the four requested ablations — whitening on/off, removal of leading components, and supervised dimensionality reduction (LDA and PLS). The supervised comparison shows that at two components a label-informed projection beats the two highest-variance principal components by about 9 points, but beyond roughly five components the two become indistinguishable — so PCA is an adequate basis, not a transformation that isolates anything.

**For your input.** The fourth requested item, "alternative feature weighting strategies," we did not run, since the other three already resolve the inconsistency he identified. Is resting on the three sufficient, or would you like the fourth included?

### R1-6 — Whitening may reduce robustness on real data

**Reviewer 1 wrote:**

> An equally important concern is that the whitening procedure may substantially reduce robustness to real observational data even if it improves performance within the synthetic simulation environment. Within TauREx, whitening appears to enhance classification by increasing the influence of low-variance components that contain discriminative methane and ozone information. However, because the entire pipeline is trained and evaluated on spectra generated by the same simulator, there is no evidence that these low-variance components represent exclusively chemically informative features. In real telescope observations, low-variance principal components are also expected to contain detector artifacts, calibration residuals, correlated instrumental noise, stellar contamination, imperfect molecular opacity models, and other observational effects. Whitening cannot distinguish chemically meaningful variance from these alternative sources of variability; it amplifies all low-variance components equally. Consequently, a preprocessing strategy that improves classification by amplifying subtle methane and ozone signatures in synthetic data may equally amplify observational noise and instrumental artifacts when deployed on real Ariel spectra. Addressing this issue requires experiments on independent simulation environments and, where possible, observational spectra to determine whether whitening genuinely improves out-of-distribution generalization rather than simply increasing performance within the synthetic training distribution.

**What it means.** On real data, the low-variance components will also carry instrument noise and artifacts; since whitening amplifies all of them equally, it may amplify noise as much as signal, hurting robustness. He wants this tested on independent environments and, if possible, real spectra.

**What we did.** We ran a controlled comparison in which two otherwise identical neural networks differ only in whether whitening is applied, each trained five times to average out initialisation. Whitening gives a small advantage on clean data (about 3.6 points) but a larger penalty under perturbation (about 11 points worse at the lowest signal-to-noise tested), with the two curves crossing as noise increases. This confirms the reviewer's hypothesis. It also reinforces the recommendation of XGBoost, which uses no whitening and is the most robust model in every case tested.

**For your input.** The reviewer asked specifically for tests on independent simulation environments and observational spectra. Our evidence comes from perturbing spectra within the same simulator, which supports his hypothesis but is not the independent-environment test he named. We plan to state this limitation plainly. Is that acceptable, or would you want the whitening robustness question re-run on the independent (Exo-Transmit) spectra we already generated for R1-1?

---

## Reviewer 2

### R2-1 — Verify reference links

**Reviewer 2 wrote:**

> Please verify that all links to the references point to the correct source.

**What it means.** Confirm each reference resolves to the correct source.

**What we did.** We are auditing all 42 references; the repository linked from the Data Availability statement has been corrected to match the manuscript. Only four references currently carry URLs.

**For your input.** None.

### R2-2 — Acknowledgments and disclosure of assistance

**Reviewer 2 wrote:**

> The authors must disclose and acknowledge any assistance received in the preparation of this manuscript, including but not limited to editorial, technical, analytical, or writing support. All such contributions must be clearly stated in the Acknowledgments section.

**What it means.** Add an Acknowledgments section disclosing all help received, including any AI tools.

**What we did.** The manuscript currently has no Acknowledgments section; we will add one before the References.

**For your input.** We need to know what to disclose — any editorial, technical, analytical, or writing assistance, and any AI tools used in analysis or drafting. Could you confirm the list of contributions to acknowledge?

### R2-3 — Reference quality

**Reviewer 2 wrote:**

> Include enough recent references along with foundational ones. Ensure references directly support your claims. Avoid "padding." Use credible sources (peer-reviewed journals, reputable books, official reports).

**What it means.** Use credible, directly relevant references; add recent work alongside foundational ones; remove filler.

**What we did.** We are replacing two bare, undated web references and auditing the rest for direct support. We are adding the citations required for the opacity data used in the simulations (which were previously uncited) and for the alternative opacity data used in the new experiments, plus foundational references for the analysis methods.

**For your input.** None, unless you have specific references you want included or removed.

### R2-4 — State all assumptions explicitly

**Reviewer 2 wrote:**

> All assumptions (including implicit assumptions) must be explicitly stated and clearly justified. The authors should explain why each assumption is reasonable and discuss its impact on the results and conclusions.

**What it means.** List every assumption — including implicit ones — with a justification and its likely effect on the results.

**What we did.** We are adding an Assumptions subsection to the Methods, covering the 1D atmosphere, fixed Gaussian noise, uniform resolution, the fixed abundance thresholds, the enforced class balance, the H₂-dominated composition, and the single molecular opacity compilation — each with its justification and likely effect. Several of these connect directly to the new robustness experiments (R1-1).

**For your input.** None.

### R2-5 — Avoid overreaching conclusions

**Reviewer 2 wrote:**

> Avoid overreaching conclusions that extend beyond what is supported by the data and analysis.

**What it means.** The same overreach concern Reviewer 1 raised — do not claim beyond what the data supports.

**What we did.** This is addressed by the same scope reduction described in the Overview: presenting the work as a synthetic-data benchmark and the task as label recovery.

**For your input.** None beyond the title question in the Overview.

### R2-6 — Tense and person

**Reviewer 2 wrote:**

> Verify that you have used past perfect tence [sic] and third person throughout the manuscript wherever applicable.

**What it means.** Make tense and person consistent — past perfect and third person where applicable. (The manuscript currently uses first-person plural.)

**What we did.** We will make a consistency pass converting first-person plural to third person and checking tense throughout.

**For your input.** None.
