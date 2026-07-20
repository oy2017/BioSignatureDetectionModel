# Revision Plan

Tracking the response to peer review for *A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra* (Journal of High School Science, revise and resubmit, July 2026).

Reviewer comments are summarised here in the authors' own words. The verbatim text is kept outside this repository, as peer review correspondence is confidential; the response document submitted to the editor quotes each comment in full.

Comment IDs (R1-n, R2-n) are stable across this file, commit messages, and the response document. To find the work behind any item: `git log --grep="R1-5"`.

## Status

| ID | Concern | Analysis | Manuscript |
| :-- | :-- | :-- | :-- |
| R1-1 | Claims exceed what the experiments show (umbrella) | — | ☐ |
| R1-2 | Study is confined to a single simulator | ☐ | ☐ |
| R1-3 | No demonstration of robustness under domain shift | ☐ | ☐ |
| R1-4 | Positive class is a labelling rule, not a detection | ☐ | ☐ |
| R1-5 | PCA interpretation unsupported by the mathematics | ☑ | ☐ |
| R1-6 | Whitening does not selectively amplify chemistry | ☑ | ☐ |
| R1-7 | Variance-ordering rationale is self-contradictory | ◐ | ☐ |
| R1-8 | Whitening may reduce robustness on real data | ☐ | ☐ |
| R1-9 | Recommends major revision (umbrella) | — | ☐ |
| R2-1 | Verify reference links resolve correctly | ◐ | ☐ |
| R2-2 | Add Acknowledgments; disclose all assistance | ☐ | ☐ |
| R2-3 | Reference quality; avoid padding | ☐ | ☐ |
| R2-4 | State all assumptions, including implicit ones | ☐ | ☐ |
| R2-5 | Avoid conclusions beyond what the data supports | ☐ | ☐ |
| R2-6 | Tense and person consistency | ☐ | ☐ |

☑ complete ◐ partial ☐ open — **3 of 15 analyses done, 0 manuscript sections rewritten.**

> The reviewer noted explicitly that these are matters of experimental validation rather than presentation. A rewrite-only response will not be sufficient; the revision needs new experiments alongside narrowed claims.

---

## Reviewer 1

### R1-1 — Claims exceed what the experiments demonstrate

*Summary:* The benchmark and statistics are sound, but the work shows that a classifier can recover predefined labels inside one simulation framework — not that the method identifies biosignatures in real spectra, nor that the PCA interpretation is correct.

**Plan:** Narrow the framing throughout. Present the work as a controlled benchmark of model families and calibration behaviour on synthetic spectra, and describe the task as recovering an abundance-threshold labelling.

**Where:** _(pending — Title, Abstract, §1, §5)_

### R1-2 — Confined to a single simulator

*Summary:* Training, validation, and testing all occur within TauREx/MultiREx, with labels derived from the same parameters that generated the spectra. Accuracy therefore demonstrates internal consistency rather than learned atmospheric physics.

**Plan:** Concede directly. Add explicit Limitations text stating that cross-simulator validation was not performed and naming it as the required next step. Pair with the R1-3 robustness study so the response is not presentation-only. A second radiative transfer code is out of reach this cycle.

**Where:** _(pending)_

### R1-3 — No demonstration of robustness under domain shift

*Summary:* Real Ariel data will carry instrumental systematics, correlated noise, stellar contamination, clouds and hazes, and 3D structure, none of which the 1D fixed-SNR simulation includes. Requests validation under realistic domain shift across seven specific axes.

**Plan:** Six of the seven requested axes are reachable — alternative opacity sources, cloud/haze prescriptions, a stellar contamination proxy, varying resolution and SNR, injected instrumental systematics, and domain-shift experiments against perturbed variants. Only the independent radiative transfer code is not. Train on the clean distribution, evaluate on perturbed test sets, report degradation honestly.

**Where:** _(pending — new results subsection + Limitations)_

### R1-4 — Positive class is a labelling rule

*Summary:* Labels come from fixed abundance thresholds applied to simulation inputs. Real observations require retrieval, which introduces uncertainty, parameter degeneracy, stellar context, and abiotic alternatives. Retrieval-derived abundances **or probabilistic labels** would strengthen relevance.

**Plan:**
1. Rename the task honestly throughout — the largest part of the fix, and free.
2. Threshold sensitivity: relabel at ±0.25 and ±0.5 dex; report the class-balance shift.
3. Margin analysis: bin test planets by distance to the decision boundary and plot accuracy per bin. Expect chance-level performance near the boundary, demonstrating that near-threshold labels are arbitrary rather than that the model fails. Connects directly to the existing §4.3 error-clustering result.
4. Detectability weighting: compute scale height from planet radius, mass, and temperature; flag planets whose feature amplitude falls below the SNR=15 noise floor. These are labelled positive but carry no detectable signal.
5. Optional, high impact: run retrieval on 50–100 spectra and plot posterior width against true injected abundance.

*Note:* the Introduction already concedes that biosignatures require geological, atmospheric, and stellar context, while the label ignores all three. Align the two.

**Where:** _(pending — §3.1, §4.3, §5)_

### R1-5 — PCA interpretation unsupported ☑

*Summary:* Principal components are linear combinations of all wavelengths, not independent physical variables. Chemical information should be distributed across many components. PC0 correlating with mean transit depth does not establish that chemistry is absent from it or isolated into PCs 2–101.

**Response:** The reviewer is correct. Rather than attempt to substantiate the physical attribution, it is **withdrawn** and replaced with a measured claim about information location that the critique does not reach.

**Evidence:**
- `ablate_pc_ranges.py` → PC0+PC1 hold 98.41% of variance but classify at **52.13%** — chance on balanced classes. Removing them costs nothing (88.58% → 88.25%). At matched dimensionality PCs 2–51 (86.17%) beat PCs 0–49 (85.65%).
- `analyze_pc_discriminative_power.py` → No component is individually strong (max AUC **0.663** at PC9). PC0 sits at **0.506**. 14 of the 20 most discriminative components fall outside the 20 highest-variance ones. Replaces the Pearson correlation the reviewer rejected.

**New claim:** classification arises from aggregating many weakly informative components; the two carrying nearly all the variance are the least informative; variance rank and discriminative rank are substantially decoupled.

**Deliberately not done:** loading-vector analysis, variance decomposition by parameter, and CH₄/O₃ differential projection all support the attribution being withdrawn. State this explicitly so it does not read as evasion.

**Precedent:** Jolliffe (1982) on low-variance components as important predictors; PCA detrending practice in transit spectroscopy.

**Where:** _(pending — §4.2, Abstract, §5)_

### R1-6 — Whitening does not selectively amplify chemistry ☑

*Summary:* Whitening rescales every low-variance component equally regardless of content, so the claim that it specifically boosts chemically informative features is not demonstrated.

**Response:** Conceded. Remove the "chemically relevant" framing from the Abstract and §3.2. Whitening is an optimisation remedy for gradient-based learners, with no claim about what it selects for.

**Evidence:** `test_whitening_necessity.py`
- XGBoost is **provably unaffected** — identical metrics with and without whitening at all four ranges tested. Demonstrates the scale-invariance §3.2 currently only asserts.
- MLP: whitening is **substitutable**. Dropping PC0–PC1 without whitening gives 79.49% ± 2.18% versus 78.76% ± 2.17% whitened on all components, and is far more stable (unwhitened on all components: ±5.63%).
- CNN: still requires whitening (67.21% vs 75.80%).

**Key framing:** the recommended model never uses whitening, so this concern does not reach the headline result.

**Where:** _(pending — Abstract, §3.2, §4.1)_

### R1-7 — Variance-ordering rationale is self-contradictory ◐

*Summary:* The manuscript claims the PCA variance ordering is physically meaningful, then whitening deliberately destroys that ordering. Both cannot be the mechanism. Requests ablations including supervised dimensionality reduction.

**Response:** Adopt the reviewer's second alternative explicitly — explained variance is unsupervised, discriminative power is supervised, and there is no contradiction in the highest-variance component not being the most discriminative. The dilemma dissolves once the claim that the ordering is physically meaningful is dropped (see R1-5).

| Requested ablation | Status |
| :-- | :-- |
| Whitening on/off | ☑ `test_whitening_necessity.py` |
| Removal of leading components | ☑ `ablate_pc_ranges.py` |
| Supervised DR (LDA / PLS) | ☐ **outstanding** |
| Alternative feature weighting | ☐ optional |

**Next:** PLS-DA on the standardised 550-bin spectra, and LDA with shrinkage on the 102 components. PLS is the pointed comparison since it maximises covariance with the label.

**Where:** _(pending — §4.2)_

### R1-8 — Whitening may reduce robustness on real data

*Summary:* On real observations, low-variance components will also hold detector artifacts, calibration residuals, and contamination. Since whitening amplifies all of them equally, it may amplify noise rather than signal outside the simulator.

**Plan:** Run the R1-3 perturbation sweep with whitened-MLP and unwhitened-XGBoost side by side and report which degrades faster. Be prepared for this to confirm the reviewer — if whitening degrades under injected systematics, report it. The recommended model does not use whitening, so the recommendation stands either way.

**Where:** _(pending)_

### R1-9 — Overall recommendation

*Summary:* Recommends major revision. The generalisation, PCA interpretation, and whitening questions all remain open, and are described as requiring experimental validation rather than rewriting.

**Where:** _(pending — summarise in cover letter)_

---

## Reviewer 2

### R2-1 — Verify reference links ◐

**Done:** The repository linked from the Data Availability statement had a stale README reporting different headline numbers than the manuscript, and an incorrect spectral resolution. Corrected — see the commit history.

**Outstanding:** Confirm all 42 references resolve. Only four currently carry URLs.

### R2-2 — Acknowledgments and disclosure of assistance

**Outstanding.** The manuscript currently has **no Acknowledgments section at all**. Add one, disclosing all editorial, technical, analytical, and writing assistance, including any AI tools used.

### R2-3 — Reference quality

**Outstanding.** References 1 and 2 are bare undated NASA entries — replace or remove. Audit all 42 for direct support of the claims they are attached to. Add recent work alongside the foundational citations. Also delete the stray reference-manager artifact in §4.1 (`("Website," n.d.)` appears mid-sentence after the bootstrap p-value).

### R2-4 — State all assumptions explicitly

**Outstanding.** Add an Assumptions subsection to §3: 1D spherically symmetric atmospheres; Gaussian noise at fixed SNR = 15; uniform R = 200 across all channels where the real instrument is heterogeneous; fixed abundance thresholds; enforced 50/50 class balance; H₂-dominated composition. Each with justification and likely effect on results. Overlaps R1-2 and R1-3.

### R2-5 — Avoid overreaching conclusions

**Outstanding.** Same scope reduction as R1-1. Note this is the second reviewer independently flagging overreach — treat as high priority.

### R2-6 — Tense and person

**Outstanding.** Consistency pass. The manuscript currently uses first-person plural throughout.

---

## Working notes

**Reproduction environment.** All results verified under the manuscript's pinned versions (Python 3.10, TensorFlow 2.21.0, scikit-learn 1.7.2, XGBoost 3.2.0). Random Forest reproduces at 86.51% ± 1.96%, matching Table 3 exactly.

**XGBoost run-to-run variance.** `subsample=0.8` means sampled rows depend on training-row ordering even at fixed `random_state`; accuracy moves roughly 0.4 percentage points across shuffles. Consider reporting a mean over restarts rather than a single run.

**Resolution confirmed.** MultiREx `wavenumber_grid` uses `np.logspace`, giving constant resolving power. Measured from the data columns, R = λ/Δλ = 199 across the band. The manuscript's "550 bins at R = 200" is correct. Note that two earlier commit messages describe the pipeline as "R=550" — that is the bin count, not the resolving power.

**Unverified physics risk.** The strongest ozone infrared band (9.6 μm) lies outside Ariel's 0.5–7.8 μm window. In-band detection relies on the weak Chappuis band near 0.6 μm and a feature near 4.74 μm. Worth checking what actually drives O₃ classification before a reviewer raises it.

**Figure 4 check.** `final_results/plots/corner_plot_errors_scatter_xgboost.png` was deleted from the working tree while the CNN, MLP, and Random Forest versions were regenerated. `final_results/figure 4.png` still exists — confirm it is current before resubmission.
