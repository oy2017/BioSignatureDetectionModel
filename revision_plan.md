# Revision Plan

Tracking the response to peer review for *A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra* (Journal of High School Science, revise and resubmit, July 2026).

Reviewer comments are summarised here in the authors' own words. The verbatim text is kept outside this repository, as peer review correspondence is confidential; the response document submitted to the editor quotes each comment in full.

Comment IDs (R1-n, R2-n) are stable across this file, commit messages, and the response document. To find the work behind any item: `git log --grep="R1-5"`.

## Status

| ID | Concern | Analysis | Manuscript |
| :-- | :-- | :-- | :-- |
| R1-1 | Claims exceed what the experiments show (umbrella) | — | ☐ |
| R1-2 | Study is confined to a single simulator | ◐ | ☐ |
| R1-3 | No demonstration of robustness under domain shift | ◐ | ☐ |
| R1-4 | Positive class is a labelling rule, not a detection | ◐ | ☐ |
| R1-5 | PCA interpretation unsupported by the mathematics | ☑ | ☐ |
| R1-6 | Whitening does not selectively amplify chemistry | ☑ | ☐ |
| R1-7 | Variance-ordering rationale is self-contradictory | ◐ | ☐ |
| R1-8 | Whitening may reduce robustness on real data | ◐ | ☐ |
| R1-9 | Recommends major revision (umbrella) | — | ☐ |
| R2-1 | Verify reference links resolve correctly | ◐ | ☐ |
| R2-2 | Add Acknowledgments; disclose all assistance | ☐ | ☐ |
| R2-3 | Reference quality; avoid padding | ☐ | ☐ |
| R2-4 | State all assumptions, including implicit ones | ☐ | ☐ |
| R2-5 | Avoid conclusions beyond what the data supports | ☐ | ☐ |
| R2-6 | Tense and person consistency | ☐ | ☐ |

☑ complete ◐ partial ☐ open — **2 of 15 analyses complete, 6 partial, 0 manuscript sections rewritten.**

Status is scored against what each reviewer explicitly asked for. R1-3 covers
about 4 of the seven requested axes; R1-8's evidence is within-simulator rather
than the independent simulation environments and observational spectra he asked
for, so both stay partial.

> The reviewer noted explicitly that these are matters of experimental validation rather than presentation. A rewrite-only response will not be sufficient; the revision needs new experiments alongside narrowed claims.

---

## Reviewer 1

### R1-1 — Claims exceed what the experiments demonstrate

*Summary:* The benchmark and statistics are sound, but the work shows that a classifier can recover predefined labels inside one simulation framework — not that the method identifies biosignatures in real spectra, nor that the PCA interpretation is correct.

**Plan:** Narrow the framing throughout. Present the work as a controlled benchmark of model families and calibration behaviour on synthetic spectra, and describe the task as recovering an abundance-threshold labelling.

**Where:** Title, Abstract (scope sentences), §1 (task framing and contributions), §5 (limitations).

### R1-2 — Confined to a single simulator ◐

*Summary:* The study is developed, trained, validated and tested inside one
synthetic ecosystem: every spectrum comes from TauREx/MultiREx, every label is
derived from the same parameters that generated the spectra, and every
evaluation uses spectra built under essentially identical modelling
assumptions.

**Status:** ◐ PARTIAL — `generate_exotransmit_testset.py`, `evaluate_exotransmit.py`, `generate_opacity_swap_testset.py`, `evaluate_opacity_swap.py`. The comment makes two distinct charges, in different states; answer them separately.

#### Charge A — every spectrum comes from one forward model

Two experiments test this, each changing one component of the forward model and holding the other fixed. Both are evaluated on the 2697 committed held-out planets, re-rendered rather than resampled, so each perturbed spectrum has a clear-sky counterpart for the identical planet and the 88.91% baseline is a within-planet comparator.

**Independent radiative transfer code.** Exo-Transmit (Kempton et al. 2017) is a different implementation in a different language with a different solver, and is the code MultiREx's opacity tables originally came from — the `.dat` files in `multirex/data` are byte-identical to `Exo_Transmit/Opac` (md5 verified). Running it therefore changes the radiative transfer while every cross section stays the same file. Accuracy 88.91% → **84.80% (−4.1)**. Most of the difference traces to Exo-Transmit assuming constant gravity where TauREx varies it with altitude — a hydrostatic bookkeeping difference rather than a disagreement about radiative transfer. `final_results/H2_exotransmit.{txt,csv}`.

**Alternative molecular opacity data.** The H₂O, CH₄ and CO₂ tables are replaced with ExoMol line lists (POKAZATEL, YT34to10, UCL-4000) while the radiative transfer code, atmospheric structure and labelling rule stay fixed. Accuracy 88.91% → **72.82% (−16.1)**, or **66.04% (−22.9)** when ozone is also replaced with HITRAN. `final_results/H2_opacity_swap{,_o3}.{txt,csv}`.

Full method and per-set numbers for both are in R1-3, axes 1 and 2.

**What this supports:** the pipeline transfers across radiative transfer implementations at a cost of about four accuracy points, and loses about sixteen when the molecular opacity data changes. Training remains entirely within TauREx; only evaluation leaves it, which is what the reviewer's stated remedy asks for — "experiments using spectra generated by independent radiative transfer codes, alternative molecular opacity databases".

#### Charge B — labels derived from the generation parameters

This charge is not touched by the forward-model work. Both experiments above recompute the *spectra* from the same injected abundances; the CH₄/O₃ threshold rule never moves. Accuracy still demonstrates that a labelling convention is recoverable from spectra, not that abundances can be inferred from observations.

**Plan.** Concede it directly. Retrieval-derived labels are the only thing that would address it, and running retrieval on the full benchmark is out of reach this cycle (§R1-4, item 5). State in §5 that the labels are a function of the generation parameters rather than of inferred abundances, and name retrieval-derived labelling as the required next step. The status stays ◐ because this half is unaddressed.

**Where:** §3.1 (labelling statement, charge B), new §4 Robustness subsection (both cost figures, charge A), §5 (limitations — cross-simulator evaluation is done and should be reported with both numbers; retrieval-derived labels named as the remaining next step).

### R1-3 — No demonstration of robustness under domain shift

*Summary:* Real Ariel data will carry instrumental systematics, correlated noise, stellar contamination, clouds and hazes, and 3D structure, none of which the 1D fixed-SNR simulation includes. Requests validation under realistic domain shift across seven specific axes.

**Status:** ◐ PARTIAL — `domain_shift_sweep.py`, `generate_cloudy_testset.py`, `evaluate_cloudy.py`, `generate_hazy_testset.py`, `evaluate_hazy.py`, `generate_opacity_swap_testset.py`, `evaluate_opacity_swap.py`. **About 4 of seven requested axes covered.**

#### What was run, in brief

Five distinct experiments, each evaluated on the five committed held-out test
sets and each leaving the trained pipeline untouched. Baseline throughout is
88.91% accuracy, Brier 0.0802, measured on those same planets.

| # | Experiment | What changes | Cost to XGBoost | Scripts |
| :-- | :-- | :-- | --: | :-- |
| 1 | Injected systematics | perturbations applied to finished spectra | up to −22.0 | `domain_shift_sweep.py` |
| 2 | Out-of-envelope split | training restricted to R_p ≤ 15 R⊕ | −8.3 | `domain_shift_sweep.py --mode extrapolation` |
| 3 | Aerosols | cloud deck or haze added to the forward model | −25.3 / −25.6 | `generate_aerosol_paired.py`, `evaluate_aerosol_paired.py` |
| 4 | Independent radiative transfer code | Exo-Transmit instead of TauREx, opacity fixed | −4.1 | `generate_exotransmit_testset.py`, `evaluate_exotransmit.py` |
| 5 | Alternative opacity data | ExoMol / HITRAN line lists, code fixed | −16.1 / −22.9 | `generate_opacity_swap_testset.py`, `evaluate_opacity_swap.py` |

Experiments 3 to 5 change the forward model itself and are evaluated on the
committed planets re-rendered with the new physics, so each perturbed spectrum
is paired with a clear-sky spectrum of the same planet. Experiment 1 perturbs
finished spectra instead. Each is written up below under its number.

#### Coverage against what was actually requested

He asked for "experiments using spectra generated by independent radiative transfer codes, alternative molecular opacity databases, different cloud and haze prescriptions, different stellar contamination models, varying spectral resolutions and signal-to-noise ratios, more realistic instrumental systematics, and domain-shift or transfer-learning experiments."

Seven requested axes, mapped to the five experiments above:

| # | Requested axis | Experiment(s) | Status | Coverage |
| :-- | :-- | :--: | :--: | :-- |
| 1 | Independent radiative transfer codes | 4 | ✅ | Exo-Transmit, paired on the committed planets with byte-identical opacity tables |
| 2 | Alternative molecular opacity databases | 5 | ✅ | H₂O/CH₄/CO₂ (and O₃, second arm) swapped to ExoMol/HITRAN line lists, paired |
| 3 | Different cloud and haze prescriptions | 3 | ✅ | grey deck and Lee-Mie haze, both added to the forward model and evaluated |
| 4 | Different stellar contamination models | 1 | ◐ | one 1/λ multiplicative proxy in the sweep |
| 5 | Varying resolution and SNR | 1 | ✅ | resolution R 200→75; white and correlated noise SNR 15→5 |
| 6 | More realistic instrumental systematics | 1 | ◐ | gain ramp, baseline offset, correlated noise — parametric forms |
| 7 | Domain-shift **or** transfer-learning | 1, 2 | ◐ | injected-systematics sweep and out-of-envelope split are domain shift |

Four axes are covered in full (1, 2, 3, 5); three are partial (4, 6, 7). The gaps and how each is addressed:

- **Axis 4 — stellar contamination (Experiment 1).** Covered by a single 1/λ multiplicative proxy, where he asked for models, plural, and a proxy is not a spot model. **Plan:** state it as a limitation. A physical spot/faculae contamination model (e.g. wavelength-dependent covering fractions from a stellar-surface code) is the next step and is out of scope this cycle; name it in §5.
- **Axis 6 — instrumental systematics (Experiment 1).** The three forms are plausible parametric choices, not derived from Ariel's instrument model, and the correlation length and stellar-contamination wavelength dependence are chosen rather than calibrated. **Plan:** state as a limitation. Deriving them from Ariel's published noise model is the next step; the instrument model is not available to this work, so this is named in §5 rather than closed.
- **Axis 7 — domain-shift or transfer-learning (Experiments 1, 2).** The requirement is disjunctive ("domain-shift **or** transfer-learning") and the domain-shift half is comprehensively covered. Transfer learning is not done. **Plan:** rest on the domain-shift half, which satisfies the disjunction, and note transfer learning (H₂→N₂, requiring N₂ regenerated at 550 bins) as an optional extension. It is low-value here because N₂ feature amplitude is ~0.03× that of H₂, i.e. the near-featureless regime, so the result is largely predictable.

**On axis 2.** In the passage listing what real Ariel data will contain, he names "incomplete molecular opacity databases" alongside the systematics and aerosols — so the opacity swap answers a concern he raised both in the axis list and in his description of real observations.

#### Experiment 1 — Injected systematics

Six perturbation families applied to the five held-out test sets, each swept over five strengths. Outputs: `final_results/H2_domain_shift_sweep.{csv,txt}`, `domain_shift_accuracy.png`, `domain_shift_calibration.png`.

```bash
python domain_shift_sweep.py                      # perturbation sweep
python domain_shift_sweep.py --mode extrapolation # out-of-envelope split
```

##### Method — the six things that make this a valid domain-shift test

1. **Models are trained once on the clean training set and never retrained.** The model is fixed; only the test distribution moves. Retraining on perturbed data would be data augmentation — a different experiment.
2. **The whole preprocessing chain is frozen.** The raw `StandardScaler`, the 102-component PCA basis, and the post-PCA scaler are all fit on clean training data only and applied unchanged to every perturbed set. Refitting any of them on perturbed data would silently invalidate the result.
3. **Only test sets are perturbed**, at the raw transit-depth stage, before any scaling. Values are not clipped to [0,1] — real observations are not clipped, and clipping would hide the failure mode.
4. **Strengths are anchored to the physical noise floor of the data**, not to arbitrary multiples of spectral scatter. Anchoring to total scatter would overstate every perturbation by roughly 6×, since only 15.8% of the scatter is noise.
5. **Calibration is reported alongside accuracy.** The manuscript claims *calibrated* triage, so a model whose accuracy survives while its Brier score collapses has still failed for the stated purpose.
6. **The resolution family has a true zero point.** Degradation uses flux-conserving binning — averaging within log-spaced bins — rather than interpolating onto a coarser grid and back. Double linear interpolation low-pass filters the spectrum even when the target grid matches the input, which would leave the family with no valid baseline; with binning, the native-resolution case (R = 200 on an R = 199 grid) costs 2.6 points.

The noise floor is measured per spectrum with a successive-difference estimator: astrophysical structure is smooth between adjacent wavelength bins while injected noise is not, so `std(diff(spectrum))/sqrt(2)` isolates the noise. Measured median is 1.04e-4 in transit depth against a total wavelength scatter of 8.11e-4 — **only 15.8% of the spectral scatter is noise.**

##### Results — XGBoost (the recommended pipeline)

Clean baseline: **88.92% accuracy, Brier 0.0802.**

| Perturbation | Strength | Accuracy | Brier |
| :-- | :-- | --: | --: |
| Baseline offset | 2× noise floor | 88.73% (−0.2) | 0.0802 |
| Gain ramp | 1× noise floor | 88.51% (−0.4) | 0.0834 |
| Gain ramp | 2× noise floor | 86.21% (−2.7) | 0.0971 |
| Stellar contamination | 1× noise floor | 87.29% (−1.6) | 0.0900 |
| Stellar contamination | 2× noise floor | 83.40% (−5.5) | 0.1189 |
| Resolution | R = 200 (native) | 86.32% (−2.6) | 0.0970 |
| Resolution | R = 100 | 78.39% (−10.5) | 0.1620 |
| Resolution | R = 75 | 73.68% (−15.2) | 0.1970 |
| White noise | SNR 15 → 10 | 80.68% (−8.2) | 0.1389 |
| White noise | SNR 15 → 5 | 73.86% (−15.1) | 0.1903 |
| Correlated noise | SNR 15 → 10 | 75.57% (−13.4) | 0.1842 |
| Correlated noise | SNR 15 → 5 | 66.89% (−22.0) | 0.2502 |

##### Findings

**Calibration-type systematics are largely tolerated.** Gain drift, baseline offsets and stellar contamination at amplitudes comparable to the noise floor cost under two accuracy points.

**Photon noise degrades gracefully.** Losing a third of the SNR costs 8 points — a reportable sensitivity, not a collapse.

**Correlated noise is markedly worse than white noise at identical effective SNR** — 75.6% vs 80.7% at SNR 10. Real instrumental noise is correlated, so this is the most operationally important vulnerability found, and it is specific rather than a blanket fragility claim.

**Baseline offset produces no degradation at all.** An additive offset moves only PC0, which carries no label information (AUC 0.506), so the model is immune to it.

#### Experiment 2 — Out-of-envelope split (partial answer to R1-2)

`final_results/H2_extrapolation_split.txt`. Train on R_p ≤ 15 R⊕, test on R_p > 15 — predicting outside the training envelope.

| Condition | Accuracy | Brier |
| :-- | --: | --: |
| Extrapolation (train R≤15, test R>15) | 75.81% | 0.1750 |
| Control (random split, same n = 1588) | 84.13% | 0.1162 |
| Full training set, in-distribution | 88.92% | 0.0802 |

The control holds training-set size fixed and removes only the distribution shift, averaged over 5 random draws. Of the naive 13-point gap, roughly **4.8 points is reduced training data and 8.3 points is the genuine extrapolation penalty.**

#### Experiment 3 — Aerosols

Reviewer 1 asked for different cloud and haze prescriptions. Two were added to
the forward model — a grey deck and a Lee-Mie haze — and the five committed clear
test sets re-rendered with each, changing nothing else, so every aerosol spectrum
is paired with a clear spectrum of the same planet and the degradation is a
within-planet difference. The no-aerosol path reproduces the committed spectra to
1.5e-15 before any set is generated. The training set contains no aerosols, so
this is a distribution shift from different forward-model physics rather than a
perturbation of existing spectra. The haze uses TauREx's `LeeMieContribution`
(Lee et al. 2013, particle radius 0.1 µm, whole-column), with `lee_mie_mix_ratio`
an absolute particle number density in particles/m³. `generate_aerosol_paired.py`
→ `evaluate_aerosol_paired.py` → `final_results/H2_aerosol_paired.{txt,csv}`.

| Aerosol | amp | XGBoost | change | Brier | MLP (5 restarts) | lost | gained |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| committed clear (same planets) | 0.97x | 88.91% | baseline | 0.0802 | 77.9% ± 2.7 | — | — |
| deck 1e5 Pa | 0.94x | 86.61% | **−2.30** | 0.0975 | 76.3% ± 2.2 | 105 | 43 |
| deck 1e4 Pa | 0.83x | 78.38% | **−10.53** | 0.1509 | 69.3% ± 1.9 | 366 | 82 |
| deck 1e3 Pa | 0.58x | 71.64% | **−17.28** | 0.2117 | 63.8% ± 1.3 | 579 | 113 |
| deck 1e2 Pa | 0.30x | 63.66% | **−25.25** | 0.2706 | 58.3% ± 1.4 | 802 | 121 |
| deck 1e1 Pa | 0.03x | 52.30% | −36.61 | 0.3624 | 52.1% ± 0.3 | 1125 | 138 |
| haze 2.0e5 m⁻³ | 0.95x | 88.88% | **−0.04** | 0.0841 | 75.7% ± 2.0 | 70 | 69 |
| haze 2.0e6 m⁻³ | 0.79x | 81.97% | **−6.95** | 0.1323 | 61.2% ± 1.3 | 280 | 93 |
| haze 3.0e7 m⁻³ | 0.59x | 65.27% | **−23.64** | 0.2782 | 53.0% ± 1.3 | 759 | 122 |
| haze 2.4e8 m⁻³ | 0.43x | 64.94% | **−23.98** | 0.2480 | 59.5% ± 3.8 | 776 | 130 |
| haze 1.0e10 m⁻³ | 0.25x | 63.30% | **−25.61** | 0.2640 | 63.1% ± 3.5 | 855 | 165 |

`lost` and `gained` count planets that flipped correct→incorrect and
incorrect→correct.

XGBoost degrades monotonically with cloud-top altitude and with haze density.
The informative range is 1e5 to 1e2 Pa: below it the deck suppresses feature
amplitude to near zero (0.03x at 1e1 Pa), so the near-chance accuracy there
reflects absent signal rather than classifier failure. MLP figures are means ±
one standard deviation over five restarts and must be quoted as such; single runs
vary by several points (`final_results/H2_mlp_reproducibility.txt`). XGBoost
reproduces exactly given fixed inputs.

#### Experiment 4 — Independent radiative transfer code

`generate_exotransmit_testset.py` → `evaluate_exotransmit.py` →
`final_results/H2_exotransmit.{txt,csv}`.

**Exo-Transmit is the code MultiREx's opacity tables came from.** The `.dat`
files in `multirex/data` are byte-identical to `Exo_Transmit/Opac` (md5
verified), so running it changes the radiative transfer implementation —
different language, different authors, different solver — while every cross
section stays the same file. That isolates the code from the opacity data, which
the swap experiment below varies instead. Spectra are computed for the exact
2697 planets in the committed clear test sets; nothing is resampled.

| Model | TauREx | Exo-Transmit | change |
| :-- | --: | --: | --: |
| XGBoost | 88.91% | **84.80%** | −4.12 |
| MLP (5 restarts) | 77.76% ± 1.4 | 69.53% ± 0.8 | −8.23 |
| XGBoost Brier | 0.0802 | 0.1119 | +0.0317 |

The codes agree closely on the spectra themselves: median per-planet correlation
**0.9965**, mean transit depth within 0.7%.

##### Bounding caveat

Exo-Transmit assumes constant gravity through the atmosphere while TauREx
integrates with gravity falling as altitude rises. Rayleigh-only comparisons
(all molecular opacity off, pure H₂) give mean transit depths agreeing to 0.4%,
confirming geometry and mean molecular weight are correct, while spectral
contrast differs by 4–12%, scaling with atmospheric vertical extent. This is a
difference between the codes, not a setting — removing it would mean editing
Exo-Transmit's source — so −4.1 is an upper bound on the difference attributable
to the numerical scheme alone.

#### Experiment 5 — Alternative opacity data

`generate_opacity_swap_testset.py` regenerates the forward model for the **exact planets in the five committed clear test sets** with the H₂O, CH₄ and CO₂ opacity tables replaced by ExoMol line lists (POKAZATEL, YT34to10, UCL-4000, in TauREx format from ExoMolOP); `evaluate_opacity_swap.py` scores them against the frozen pipeline. Outputs: `final_results/H2_opacity_swap.{txt,csv}`.

The radiative transfer code, atmospheric structure, geometry, wavelength grid, labelling rule and the planets themselves are held fixed; only the molecular opacity data changes, and pairing is exact — every swapped planet is the same planet regenerated. The harness is validated by regenerating with the *original* tables, which reproduces the committed spectra to 1.2e-14 relative (`--validate`).

Two arms are reported. The **primary** arm replaces the three molecules ExoMolOP provides, keeping the alternative compilation internally coherent — ExoMol spectroscopy throughout. The **complete** arm additionally replaces O₃ with HITRAN cross sections converted from petitRADTRANS (`convert_hitran_o3_to_taurex.py`), covering every spectrally active molecule including both that define the label, at the cost of mixing source families.

| | Baseline | Primary (ExoMol H₂O/CH₄/CO₂) | Complete (+ HITRAN O₃) |
| :-- | --: | --: | --: |
| Accuracy | 88.91% | **72.82% (−16.09)** | **66.04% (−22.88)** |
| F1 | 0.889 | 0.758 | 0.640 |
| Brier | 0.0802 | 0.2033 | 0.2529 |
| Predicted-positive rate | 49.5% | 61.8% | 44.0% |
| Predictions changed | — | 26.8% | 32.7% |

(True positive rate 50.3%. Outputs: `final_results/H2_opacity_swap{,_o3}.{txt,csv}`.)

Changing the opacity data costs 16.1 accuracy points for a coherent alternative compilation (ExoMol throughout, ozone unchanged), rising to 22.9 when ozone is replaced with HITRAN as well — the only arm in which both label-determining molecules move. Report both. Feature amplitude is essentially unchanged (0.95x → 0.93x), so this is a shift in the decision rather than the signal loss that drives the aerosol results: the predicted-positive rate moves from 49.5% to 61.8% (primary) and to 44.0% (complete) against a true 50.3%.

**Caveats to state.**
- CO and NH₃ appear in every composition but have no opacity data in either database — they act only through mean molecular weight. Disclose under R2-4: two of the six "background" gases are spectrally inert.
- Native resolutions differ (ExoMolOP R=15000 tables vs the Exo-Transmit grid), so a small part of the difference is tabulation rather than line-list content.
- Ozone cannot be independently adjudicated. Every ozone tabulation in circulation (Freedman/Lupu, HITRAN, DACE's HITRAN2020) traces to HITRAN, because ozone dissociates above ~500 K and hot-atmosphere databases do not produce an independent list. Treat the ozone tabulation as an irreducible forward-model uncertainty.

**Outstanding for this axis:** nothing blocking; the ExoMolOP swap covers it in both arms. Next steps for R1-3 overall are the three partial axes: a physical stellar spot/faculae contamination model (axis 4), instrumental systematics derived from Ariel's published noise model (axis 6), and transfer learning H₂→N₂ (axis 7; requires regenerating N₂ at 550 bins). The independent radiative transfer code (axis 1, Experiment 4) is done and must not be listed as outstanding.

#### Caveats to state in the manuscript

- These are **parametric models of systematics, not real instrument data.** The correlated-noise correlation length (Gaussian kernel, sigma = 8 bins) and the stellar-contamination wavelength dependence (a 1/λ trend, not a spot model) are plausible choices, not calibrated ones.
- Everything remains **inside TauREx.** Perturbed TauREx spectra are still TauREx spectra. This does not answer R1-2 in full; only an independent radiative transfer code would.
- The resolution family perturbs an already-generated R=199 grid rather than regenerating at lower resolution, so it conflates resampling with true resolution loss, though flux-conserving binning minimises this.

**Where:** new §4.5 Robustness subsection, new §3 Assumptions subsection, §5 (limitations).


### R1-4 — Positive class is a labelling rule

*Summary:* Labels come from fixed abundance thresholds applied to simulation inputs. Real observations require retrieval, which introduces uncertainty, parameter degeneracy, stellar context, and abiotic alternatives. Retrieval-derived abundances **or probabilistic labels** would strengthen relevance.

**Status:** ◐ PARTIAL — `analyze_label_margin.py`, `analyze_threshold_sensitivity.py`.

**The response, in one paragraph.** The reviewer is right that the positive class is a labelling convention, not a detection, and the response concedes this rather than contesting it. The concession is made concrete in two ways. First, the task is renamed throughout the paper — from "biosignature detection" to recovery of an abundance-threshold labelling — so the manuscript stops claiming more than the experiment shows; this is the largest part of the fix and it is free. Second, two analyses on the existing data characterise how much the labelling convention actually governs the result: a threshold-sensitivity test showing the headline accuracy does not depend on the particular cutoff chosen, and a margin analysis showing the classifier extracts no information for planets near the cutoff, where the label is genuinely arbitrary. What the response does **not** do is make the labels observationally meaningful — that would require retrieval-derived abundances, which is out of reach this cycle and is named as the primary next step. The reviewer's suggestion of probabilistic labels is addressed in spirit by the margin analysis (near-threshold planets are shown to be intrinsically ambiguous) rather than by relabelling. The four items below implement this: item 1 is the renaming, items 2–3 the two analyses, item 4 the conceded retrieval work.

**Plan:**
1. ☑ Rename the task honestly throughout — the largest part of the fix, and free. Instructions in the rewrite section below.
2. ☑ Threshold sensitivity: relabel at ±0.25 and ±0.5 dex; report the class-balance shift.
3. ☑ Margin analysis: accuracy binned by distance to the labelling threshold.
4. ☐ Out of reach — retrieval on 50–100 spectra with posterior width against injected abundance. Concede explicitly. Retrieval is the single out-of-reach item across R1-2 and R1-4; the cross-simulator request is answered. Retrieval on the full benchmark is months of compute and would replace the benchmark rather than revise it.

#### Margin analysis — results

`final_results/H2_label_margin.{txt,csv}`, figure `label_margin.png`. Margin is the dex distance to the nearest label flip: for positives the smaller excess over the two cutoffs, for negatives the larger deficit (both gases must rise for the label to change). Class balance varies between bins, so accuracy is reported against each bin's majority-class baseline — the score obtainable with no spectral information at all.

| Margin (dex) | n | Positives | Accuracy | Majority baseline | Gain |
| :-- | --: | --: | --: | --: | --: |
| 0 – 0.25 | 236 | 64.4% | 61.86% | 64.41% | **−2.54** |
| 0.25 – 0.5 | 258 | 67.1% | 75.97% | 67.05% | +8.91 |
| 0.5 – 1 | 454 | 62.1% | 85.90% | 62.11% | +23.79 |
| 1 – 2 | 908 | 50.2% | 94.82% | 50.22% | +44.60 |
| > 2 | 841 | 35.0% | 95.72% | 65.04% | +30.68 |

**Within 0.25 dex of the cutoff the classifier recovers no usable information at all** — it does not beat the majority baseline (−2.5 points), and its mean predicted probability there is 0.515, i.e. it reports its own uncertainty correctly rather than guessing confidently. At a dex or more from the cutoff it reaches 95.25%. This is the reviewer's own objection, measured: labels within a quarter-dex of an arbitrary cutoff separate atmospheres that are physically near-identical, and no spectrum can recover that distinction. 8.8% of the test set sits in that regime.

The near-threshold bin is where the residual error concentrates, which explains part of the 11% overall error rate as a property of the labelling rather than of the model — connect this to the §4.3 error-clustering result.

**The same table also bounds the headline.** 64.8% of the test set lies a dex or more from the cutoff, where accuracy is 95.25% — comparisons between atmospheres differing by more than a factor of ten in the deciding abundance, which are easy calls. The 88.92% headline is therefore weighted toward well-separated cases, and performance where the distinction is genuinely hard is much weaker. This follows from the generation design (abundances drawn uniformly across several decades of log abundance); the margin analysis makes it visible. Report both halves of the table together, and record the difficulty distribution in the §3 Assumptions subsection (R2-4).

#### Threshold sensitivity — results

`final_results/H2_threshold_sensitivity.{txt,csv}`. Both cutoffs moved together, data relabelled, pipeline retrained from scratch at each setting; spectra unchanged.

| Shift | CH₄ / O₃ cutoff | Test positives | Accuracy | Majority baseline | Gain |
| :-- | :-- | --: | --: | --: | --: |
| −0.50 | −6.50 / −7.50 | 56.6% | 86.99% | 56.58% | +30.40 |
| −0.25 | −6.25 / −7.25 | 53.4% | 88.17% | 53.43% | +34.74 |
| 0 | −6.00 / −7.00 | 50.3% | 88.91% | 50.32% | +38.60 |
| +0.25 | −5.75 / −6.75 | 44.7% | 88.69% | 55.32% | +33.37 |
| +0.50 | −5.50 / −6.50 | 38.3% | 87.84% | 61.74% | +26.10 |

**Accuracy varies by under 2 points across a full dex of cutoff movement** (86.99–88.91%), so the headline number is not an artefact of the particular cutoffs chosen. The gain over the majority baseline peaks at the original cutoffs (+38.6) and falls to +26.1 at +0.5 dex — expected, since the generation plan enforces class balance at the original values and shifted labellings are progressively unbalanced. The gain column is the meaningful comparison across shifts: raw accuracy at +0.5 dex (87.84%) sits against a 61.74% majority baseline, so stability in the raw figure would overstate what the classifier is doing.

*Note:* the Introduction already concedes that biosignatures require geological, atmospheric, and stellar context, while the label ignores all three. Align the two.

**Where:** Title and Abstract (task naming), §1 (task framing), §3.1 (labelling convention statement), §4.3 (margin analysis), §5 (limitations).

### R1-5 — PCA interpretation unsupported ☑

*Summary:* Principal components are linear combinations of all wavelengths, not independent physical variables. Chemical information should be distributed across many components. PC0 correlating with mean transit depth does not establish that chemistry is absent from it or isolated into PCs 2–101.

**Response:** The reviewer is correct. Rather than attempt to substantiate the physical attribution, it is **withdrawn** and replaced with a measured claim about information location that the critique does not reach.

**Evidence:**
- `ablate_pc_ranges.py` → PC0+PC1 hold 98.41% of variance but classify at **52.13%** — chance on balanced classes. Removing them costs nothing (88.58% → 88.25%). At matched dimensionality PCs 2–51 (86.17%) beat PCs 0–49 (85.65%).
- `analyze_pc_discriminative_power.py` → No component is individually strong (max AUC **0.663** at PC9). PC0 sits at **0.506**. 14 of the 20 most discriminative components fall outside the 20 highest-variance ones. Replaces the Pearson correlation the reviewer rejected.

**New claim:** classification arises from aggregating many weakly informative components; the two carrying nearly all the variance are the least informative; variance rank and discriminative rank are substantially decoupled.

**Marked complete conditionally.** Only one of his four requested analyses was run (selective component removal). The other three — loading-vector analysis, variance decomposition by physical parameter, and CH₄/O₃ projection — were skipped because his sentence ends "before drawing these mechanistic conclusions," and the conclusions are being withdrawn. That reasoning holds only if the retraction is stated explicitly and prominently. If any version of the physical attribution survives into the revised text, these three analyses become mandatory again.

**Not done, and why:** loading-vector analysis, variance decomposition by parameter, and CH₄/O₃ differential projection all support the attribution being withdrawn, so they were not run. The response should say so directly.

**Precedent:** Jolliffe (1982) on low-variance components as important predictors; PCA detrending practice in transit spectroscopy.

**Where:** §4.2 (full rewrite, with the retraction stated explicitly), Abstract (delete the chemistry attribution), §4.1 (delete the attribution clause), §5 (delete the closing claim about later components preserving chemical structure).

### R1-6 — Whitening does not selectively amplify chemistry ☑

*Summary:* Whitening rescales every low-variance component equally regardless of content, so the claim that it specifically boosts chemically informative features is not demonstrated.

**Response:** Conceded. Remove the "chemically relevant" framing from the Abstract and §3.2. Whitening is an optimisation remedy for gradient-based learners, with no claim about what it selects for.

**Evidence:** `test_whitening_necessity.py`
- XGBoost is **provably unaffected** — identical metrics with and without whitening at all four ranges tested. Demonstrates the scale-invariance §3.2 currently only asserts.
- MLP: whitening is **substitutable**. Dropping PC0–PC1 without whitening gives 79.49% ± 2.18% versus 78.76% ± 2.17% whitened on all components, and is far more stable (unwhitened on all components: ±5.63%).
- CNN: still requires whitening (67.21% vs 75.80%).

**Key framing:** the recommended model never uses whitening, so this concern does not reach the headline result.

**Where:** Abstract (delete "chemically relevant" framing), §3.2 (reframe whitening as an optimisation remedy), §4.1.

### R1-7 — Variance-ordering rationale is self-contradictory ◐

*Summary:* The manuscript claims the PCA variance ordering is physically meaningful, then whitening deliberately destroys that ordering. Both cannot be the mechanism. Requests ablations including supervised dimensionality reduction.

**Response:** Adopt the reviewer's second alternative explicitly — explained variance is unsupervised, discriminative power is supervised, and there is no contradiction in the highest-variance component not being the most discriminative. The dilemma dissolves once the claim that the ordering is physically meaningful is dropped (see R1-5).

| Requested ablation | Status |
| :-- | :-- |
| Whitening on/off | ☑ `test_whitening_necessity.py` |
| Removal of leading components | ☑ `ablate_pc_ranges.py` |
| Supervised DR (LDA / PLS) | ☑ `supervised_dr_comparison.py` |
| Alternative feature weighting | ☐ **outstanding** |

Three of his four requested ablations are now done.

#### Supervised dimensionality reduction — how to reproduce

```bash
python supervised_dr_comparison.py
```

Outputs `final_results/H2_supervised_dr.{txt,csv}`. Runtime is a few minutes; no GPU or TensorFlow needed.

Four approaches are compared at matched component counts (n = 2, 5, 10, 20, 50, 102), all projections fit on the training set only and applied unchanged to the five held-out test sets:

| Method | Projection | Classifier |
| :-- | :-- | :-- |
| PCA + XGBoost | unsupervised (variance) | nonlinear |
| PLS + XGBoost | supervised (label covariance) | nonlinear |
| PLS-DA | supervised | linear (logistic on PLS scores) |
| LDA | supervised | linear |

The `PLS + XGBoost` arm is the one that answers the question, because it holds the classifier fixed so that only the projection differs. Comparing linear PLS-DA against nonlinear PCA + XGBoost would confound the projection with the classifier and test linearity rather than the choice of basis.

#### Results

| n | supervised (PLS + XGB) | unsupervised (PCA + XGB) | difference |
| --: | --: | --: | --: |
| 2 | **60.70%** | 52.13% | **+8.57** |
| 5 | 72.23% | 72.72% | −0.49 |
| 10 | 79.62% | 80.00% | −0.37 |
| 20 | 83.24% | 81.92% | +1.32 |
| 50 | 87.25% | 85.33% | +1.92 |
| 102 | 88.59% | 88.92% | −0.33 |

Linear methods for comparison: PLS-DA plateaus at 73.93%, LDA reaches 68.22% on the 102 components and 63.58% on raw bins.

#### Three findings

**At two components the supervised projection wins by 8.6 points.** This shows that variance rank is not the right selection criterion: constrained to two directions, the two highest-variance ones give chance-level performance while two label-informed directions give 60.7%.

**Beyond roughly five components the two converge**, and at 102 they are indistinguishable. PCA is therefore not privileged — it is an adequate basis that retains the information once enough dimensions are kept, not a transformation that isolates anything.

**Both linear methods cap in the 63–74% range** against 88.9% for the nonlinear classifier on the same features.

**Still outstanding:** alternative feature weighting strategies, the fourth of his requested ablations. Not run. The other three ablations address the inconsistency he identified, so the response can rest on them.

**Where:** §4.2 (supervised-DR comparison), §3.2 (the reframing of what PCA is doing).

### R1-8 — Whitening may reduce robustness on real data

*Summary:* On real observations, low-variance components will also hold detector artifacts, calibration residuals, and contamination. Since whitening amplifies all of them equally, it may amplify noise rather than signal outside the simulator.

**Status:** ◐ PARTIAL — `domain_shift_sweep.py`, `domain_shift_mlp_restarts.py`. **His hypothesis is confirmed, but not by the method he requested.**

He asked for "experiments on independent simulation environments and, where possible, observational spectra to determine whether whitening genuinely improves out-of-distribution generalization." Neither exists. The evidence below comes from within-simulator perturbations, which is meaningful support for the hypothesis but is not the out-of-distribution test he specified. The response should state this rather than presenting the within-simulator evidence as a complete answer.

#### Experimental design — why this isolates whitening

The comparison must hold architecture fixed: a whitened MLP against an unwhitened XGBoost would differ in **both** architecture and whitening, leaving any difference uninterpretable. The sweep trains three models on identical frozen PCA features:

| Model | Components | Whitened | Role |
| :-- | :-- | :-- | :-- |
| XGBoost | 0–101 | no | recommended pipeline; scale-invariant control |
| MLP (whitened) | 0–101 | **yes** | the manuscript's neural pipeline |
| MLP (unwhitened) | 0–101 | **no** | identical in every way except whitening |

The two MLP rows share architecture, hyperparameters, component set and (within each restart) training seed, and differ **only** in whether the post-PCA `StandardScaler` is applied. Any difference in their degradation is therefore attributable to whitening itself.

#### Result 1 — the curves cross (requires no normalisation)

The cleanest evidence needs no statistical adjustment at all. Whitening *helps* on clean data and *hurts* under perturbation, so the two curves cross. Values are means ± σ over **five training restarts of each network** (`domain_shift_mlp_restarts.py` → `final_results/H2_whitening_restarts.{txt,csv}`; differences are paired within restarts):

| White noise | MLP whitened | MLP unwhitened | W − U (paired) |
| :-- | --: | --: | --: |
| SNR 15 (clean) | **79.3% ± 2.1** | 75.7% ± 2.5 | +3.6 ± 3.1 |
| SNR 12 | 69.1% ± 1.2 | **74.7% ± 2.7** ← crossover | −5.6 ± 2.1 |
| SNR 10 | 64.9% ± 1.1 | **73.8% ± 2.6** | −8.9 ± 1.8 |
| SNR 8 | 61.6% ± 0.7 | **72.3% ± 2.5** | −10.6 ± 2.0 |
| SNR 5 | 57.0% ± 0.3 | **68.0% ± 2.0** | −11.0 ± 1.7 |

Whitening starts 3.6 ± 3.1 points ahead — a clean-data benefit that is itself marginal against training variability — and ends 11.0 ± 1.7 points behind. The crossover lands by SNR 12 in four of five restarts; in the fifth the whitened network never led at all. The same crossover appears in the resolution-loss and stellar-contamination panels.

#### Result 2 — degradation as a fraction of headroom above chance

Models with a lower clean baseline have less room to fall, so raw point-drops are not directly comparable. Normalising by each model's headroom above 50%:

| Perturbation family | XGBoost | MLP whitened | MLP unwhitened | verdict |
| :-- | --: | --: | --: | :-- |
| White noise | 39% | **76% ± 1** | 30% ± 3 | whitening worse |
| Correlated noise | 57% | **77% ± 1** | 49% ± 5 | whitening worse |
| Resolution loss | 39% | **63% ± 3** | 39% ± 6 | whitening worse |
| Stellar contamination | 14% | **49% ± 8** | 24% ± 3 | whitening worse |
| Gain ramp | 7% | **14% ± 5** | 5% ± 2 | whitening worse |
| Baseline offset | 0% | 0% | 0% | no difference |

(MLP columns are restart means ± σ; the XGBoost column is deterministic and unchanged from the original sweep.)

Whitening reduces robustness in **5 of 6 families**. The sole exception is baseline offset, which is the one perturbation no model responds to at all (see R1-3), so it carries no information either way.

#### Interpretation

Reviewer 1 argued that whitening cannot distinguish chemically meaningful low-variance structure from detector artifacts and calibration residuals, and so would amplify both. A controlled experiment supports exactly that. Whitening rescales every low-variance component to unit variance regardless of content, which raises the weight of components dominated by noise and systematics as much as those carrying signal.

**The result supports his hypothesis and is reported as such.**

#### What it does for the paper

The trade-off is now quantified with error bars: **whitening buys 3.6 ± 3.1 accuracy points on clean data — a marginal benefit — and costs 1.6–2.8× more robustness headroom under perturbation across the five families with a measurable effect.**

It also strengthens the XGBoost recommendation rather than weakening the paper. XGBoost uses no whitening at all (demonstrated bit-identical with and without it, see R1-6), is the most accurate model, and is the most robust in absolute terms in every family tested. The recommendation now rests on three legs instead of two: accuracy, calibration, and robustness.

#### Caveat

The unwhitened MLP has a lower clean baseline (75.7% vs 79.3% restart means), so the headroom normalisation in Result 2 is a modelling choice a reader could question. Result 1 — the raw crossover — does not depend on it, and both point the same way. Both networks were retrained five times (`domain_shift_mlp_restarts.py`), so every quoted number is a restart mean ± σ with differences paired within restarts. The clean-data advantage of whitening is 3.6 ± 3.1 points — marginal against its own scatter, which strengthens the concession: whitening buys little even where it is supposed to help. Quote the crossover as "by SNR 12 in four of five restarts", not as a fixed SNR.

**Where:** §3.2 (whitening reframing), §4.1, new §4.5 Robustness subsection (the controlled whitened-vs-unwhitened result).

### R1-9 — Overall recommendation

*Summary:* Recommends major revision. The generalisation, PCA interpretation, and whitening questions all remain open, and are described as requiring experimental validation rather than rewriting.

**Where:** cover letter (summary of what was and was not addressed), §5 (limitations).

---

## Reviewer 2

### R2-1 — Verify reference links ◐

**Done:** The README in the repository linked from the Data Availability statement reports the same headline numbers and spectral resolution as the manuscript.

**Outstanding:** Confirm all 42 references resolve. Only four currently carry URLs.

### R2-2 — Acknowledgments and disclosure of assistance

**Outstanding.** The manuscript currently has **no Acknowledgments section at all**. Add one, disclosing all editorial, technical, analytical, and writing assistance, including any AI tools used.

### R2-3 — Reference quality

**Outstanding.** References 1 and 2 are bare undated NASA entries — replace or remove. Audit all 42 for direct support of the claims they are attached to. Add recent work alongside the foundational citations. Also delete the stray reference-manager artifact in §4.1 (`("Website," n.d.)` appears mid-sentence after the bootstrap p-value).

**Required additions — the opacity data is uncited.** The molecular opacities underlying every spectrum in this work are the Exo-Transmit tables distributed with MultiREx, whose user manual states that use of the opacity data requires citing three papers. None of the three currently appears in the manuscript:

- Freedman, R. S., Marley, M. S., & Lodders, K. (2008), *ApJS* **174**, 504
- Freedman, R. S., Lustig-Yaeger, J., Fortney, J. J., et al. (2014), *ApJS* **214**, 25
- Lupu, R. E., Zahnle, K., Marley, M. S., et al. (2014), *ApJ* **784**, 27



**Required additions — the alternative opacity data in §4.5.** The opacity-database experiment uses ExoMolOP cross sections, which carry their own citation requirements: the opacity product, and the line list behind each molecule. All four verified against DOI records:

- Chubb, K. L., Rocchetto, M., Yurchenko, S. N., et al. (2021), *A&A* **646**, A21 — the ExoMolOP database itself (`10.1051/0004-6361/202038350`)
- Polyansky, O. L., Kyuberis, A. A., Zobov, N. F., et al. (2018), *MNRAS* **480**, 2597 — H₂O POKAZATEL (`10.1093/mnras/sty1877`)
- Yurchenko, S. N., Amundsen, D. S., Tennyson, J., & Waldmann, I. P. (2017), *A&A* **605**, A95 — CH₄ YT34to10 (`10.1051/0004-6361/201731026`)
- Yurchenko, S. N., Mellor, T. M., Freedman, R. S., & Tennyson, J. (2020), *MNRAS* **496**, 5282 — CO₂ UCL-4000 (`10.1093/mnras/staa1874`)

The first two of those DOIs are recorded in the downloaded files; the CO₂ file's DOI field contains the placeholder string `qqq`, so its reference was identified from the line-list name and confirmed independently. Do not take provenance metadata in opacity products on trust — the same placeholder appears in petitRADTRANS's CO₂ table.

Cite Kempton et al. (2017) for Exo-Transmit itself alongside them, and state in §3 which opacity compilation was used — the choice is not incidental, since §4.5 shows that substituting an alternative compilation costs 16.1 accuracy points. A reviewer checking reproducibility would expect the opacity source named.

### R2-4 — State all assumptions explicitly

**Outstanding.** Add an Assumptions subsection to §3: 1D spherically symmetric atmospheres; Gaussian noise at fixed SNR = 15; uniform R = 200 across all channels where the real instrument is heterogeneous; fixed abundance thresholds; enforced 50/50 class balance; H₂-dominated composition. Each with justification and likely effect on results. Overlaps R1-2 and R1-3.

### R2-5 — Avoid overreaching conclusions

**Outstanding.** Same scope reduction as R1-1. This is the second reviewer independently flagging overreach.

### R2-6 — Tense and person

**Outstanding.** Consistency pass. The manuscript currently uses first-person plural throughout.

---

# Manuscript rewrite instructions

Every entry in the Manuscript column above is still open. This section says exactly where each change goes and what to write. Work top to bottom; later sections depend on decisions made in earlier ones.

**Before starting, decide one thing:** the paper's scope. Everything else follows from it. The recommendation, based on the evidence gathered, is to present the work as *a controlled benchmark of model families, calibration behaviour and robustness on synthetic Ariel-like spectra*, rather than as a validated triage tool for the mission. Two reviewers independently flagged overreach (R1-1, R2-5), so this is not optional.

## Priority order

1. §4.2 — largest rewrite, and the one carrying a retraction
2. §3.2 and §4.1 — whitening reframing, must be consistent with §4.2
3. Abstract, Title, §5 — scope reduction, written last so they reflect the body
4. New robustness subsection in §4
5. New Assumptions subsection in §3
6. Acknowledgments — currently absent entirely
7. References and tense pass

---

## Title — serves R1-1, R2-5

Current: *A Calibrated PCA–Machine Learning Pipeline for Biosignature Candidate Triage in Exoplanet Transmission Spectra*

Defensible **only if** the Abstract scopes it clearly. "Biosignature candidate triage" implies operational readiness the evidence does not support. Either add "Synthetic" before "Exoplanet Transmission Spectra", or leave it and carry the scoping in the first and last sentences of the Abstract. **Open decision — needs the mentor's agreement before the abstract is finalised**, since the two options put the scoping in different places.

## Abstract — serves R1-1, R1-5, R1-6, R2-5

**Delete outright** (both are retracted claims — R1-5, R1-6):

> "PCA was used to reduce dimensionality while a secondary standardization step was applied to equalize the variance between the dominant physical systematics of the first two components and the subtle, chemically relevant absorption features contained in higher-order components."

> "the scale-invariant, node-splitting architecture of XGBoost exploited the chemically informative low-variance components natively, isolating absorption features from the dominant physical baselines"

**Replace the first** with a description of what the preprocessing does without claiming what it selects for:

> "PCA reduced the 550-bin spectra to 102 components. A secondary standardisation of those components was applied for the gradient-based models, which are sensitive to the large disparity in component variance; the tree ensembles are invariant to it."

**Replace the second** with the measured result:

> "The tree ensembles exploited discriminative structure distributed across the low-variance components without rescaling, whereas the neural networks required explicit standardisation to remain competitive and still trailed."

**Add two sentences**, one on scope and one on robustness:

> "Training data derives from a single radiative transfer code, and class labels are a deterministic function of the abundances used to generate them, so this work characterises achievable performance under idealised conditions rather than demonstrating biosignature detection. Evaluating the trained pipeline on spectra recomputed for the same planets with an independent radiative transfer code reduces accuracy by 4.1 percentage points, and substituting an alternative molecular opacity compilation reduces it by 16.1."

> "Under injected observational systematics the pipeline tolerates calibration-type errors but degrades under correlated noise, losing 13 accuracy points at an effective SNR of 10. An untrained-for optically thick cloud deck or photochemical haze degrades the recommended pipeline's accuracy monotonically with cloud-top altitude and haze density, bounding the method's applicability to atmospheres whose aerosols leave most of the spectral feature amplitude intact."

## §1 Introduction — serves R1-1, R1-4

**Task framing.** Wherever the task is described as "distinguishing between biosignature and non-biosignature environments", restate it as recovering a predefined abundance-threshold labelling. The Introduction already concedes that biosignatures require geological, atmospheric and stellar context — align the description of the label with that concession rather than leaving them contradictory (R1-4). The full renaming pass is specified in "Task renaming" below; do it once, globally, before editing individual sections, so later edits are written in the new vocabulary rather than converted afterwards.

**Contributions.** The three stated contributions (R = 200 regime; controlled like-for-like comparison; calibration as a first-class criterion) are still accurate. Add a fourth: robustness characterisation under injected systematics. Do **not** list the PCA interpretation as a contribution — it never was one, and it is now withdrawn.

## Task renaming — serves R1-4, R1-1, R2-5

**Do this first, globally.** The single largest part of the R1-4 response, and it costs nothing but words. The task is the recovery of a predefined abundance-threshold labelling from synthetic spectra; the manuscript currently describes it as biosignature detection. Every later section should be written in the corrected vocabulary rather than converted afterwards.

Replace throughout — the left column is what to search for, the right what to say instead:

Occurrence counts are from `Final Paper Manuscript (9).pdf`, so each row is a
search that will actually hit:

| Current phrasing | Occurrences | Replace with |
| :-- | --: | :-- |
| "distinguishing between biosignature and non-biosignature environments" | 1 | "recovering a predefined abundance-threshold labelling" |
| "biosignature detection" | 3 | "biosignature-candidate labelling" / "threshold-label recovery" |

The other phrasings considered for this table — "detecting biosignatures in
spectra", "biosignature planets", "identifies biosignatures" — do not occur in
the manuscript and need no pass.

Related phrases that do occur and carry the retracted attribution, to be removed
along with the passages quoted under §3.2, §4.1, §4.2 and §5 below: "chemically
informative" (4), "chemically critical" (1), "chemically relevant" (1). The
final check requires all three at zero.

Keep "biosignature" only where it refers to the underlying *scientific concept* (the Introduction's motivation, the CH₄/O₃ disequilibrium rationale, the discussion of what real detection would require). Remove it wherever it describes *what the model does*.

Two supporting statements to add. In §3.1, after the labelling rule (R1-4):

> "Because the labels are computed directly from the abundances supplied to the forward model, the classification task is the recovery of a predefined labelling convention from spectra, not the inference of abundances from observations. Ariel will require the latter, via retrieval, with the attendant uncertainty, parameter degeneracy and abiotic alternatives."

And, once the threshold-sensitivity result is included (§3.1 or §4.3):

> "The particular cutoffs are a convention inherited from prior work rather than a physical boundary. Moving both cutoffs together by up to half a dex and relabelling changes overall accuracy by less than two points (86.99% to 88.91%), so the reported performance is not an artefact of the specific values chosen; the advantage over a majority-class baseline is largest at the original cutoffs, where the class balance enforced by the generation plan holds."

## §3.1 Dataset — serves R1-4, R1-2

Covered by the two statements in "Task renaming" above; add them after the labelling rule.

## §3.2 Preprocessing — reframe whitening — serves R1-6, R1-7

**Delete** (R1-6):

> "This whitening step prevented the neural network architectures from being dominated by the high-variance physical baselines (PC0 and PC1) at the expense of the lower-variance but chemically critical higher-order components."

**Replace with:**

> "This standardisation places all retained components on a common scale. It is an optimisation remedy for the gradient-based architectures, which are dominated by the two highest-variance components in its absence; it makes no distinction between components on the basis of their content, rescaling every low-variance component equally regardless of whether it carries molecular absorption, continuum structure or noise."

**Then add the two supporting results:**

> "The tree ensembles are unaffected: XGBoost returns identical metrics with and without this step at every component range tested, confirming the expected invariance of node splitting to monotonic per-feature rescaling. The step is also substitutable — discarding the two highest-variance components and omitting standardisation entirely gives the MLP 79.49% ± 2.18%, against 78.76% ± 2.17% for the standardised full-component pipeline."

Evidence: `test_whitening_necessity.py` → `final_results/H2_whitening_necessity.txt`.

## §3 — new Assumptions subsection (R2-4)

Add after §3.1 or at the end of §3. State each assumption, why it is reasonable, and its likely effect on results:

| Assumption | Justification | Effect on results |
| :-- | :-- | :-- |
| 1D spherically symmetric atmospheres | TauREx standard; tractable | Omits limb asymmetry and 3D structure; likely optimistic |
| Gaussian noise at fixed SNR = 15 | Follows Duque-Castaño et al. | Real noise is correlated; the sweep shows correlated noise costs ~5 more points at equal SNR |
| Uniform R = 200 across all channels | Approximates Ariel Tier 3 upper end | The real instrument is heterogeneous; optimistic |
| Fixed CH₄/O₃ abundance thresholds | Follows prior work | Labels near the threshold are arbitrary: within 0.25 dex the classifier does not beat a majority-class baseline (§4.3). Moving both cutoffs by ±0.5 dex changes accuracy by under 2 points, so the headline is not an artefact of the specific values |
| Abundances drawn uniformly over several decades of log abundance | Spans the physically plausible range; avoids concentrating the sample at one composition | **Sets the difficulty distribution**: 64.8% of test planets lie ≥1 dex from the labelling cutoff, where accuracy is 95.3%, so overall accuracy is weighted toward well-separated cases (§4.3). Not an estimate of performance on a realistic population |
| Enforced 50/50 class balance | Prevents majority-class bias | Not the expected occurrence rate; accuracy is not a mission yield estimate |
| H₂-dominated composition | Larger scale height, stronger signal | Optimistic relative to high-mean-molecular-weight atmospheres |
| A single set of molecular opacity tables (Exo-Transmit compilation, via MultiREx) | The tabulation shipped with the simulation package | **Load-bearing, now quantified** — replacing the H₂O/CH₄/CO₂ tables with ExoMol line lists costs 16.1 accuracy points; replacing ozone as well costs 22.9 (§4.5). The direction of the bias depends on which molecules move. Also note CO and NH₃ are present in every composition but have no opacity data in either compilation, so they act only through mean molecular weight |
| No clouds or hazes in training | Not exposed by MultiREx until this work; the fork now supports a grey deck and a Lee-Mie haze | **Quantified for two prescriptions** — an untrained-for deck at 10⁴ Pa costs 10.5 accuracy points and nearly doubles the Brier score; a haze degrades performance monotonically with density (§4.5). Degradations are measured on the committed planets themselves, re-rendered with the aerosol added, so the comparison is paired planet-by-planet — see Experiment 3 |

## §4.1 Overall Performance — serves R1-5, R2-3

**Delete** the clause claiming chemical attribution (R1-5):

> "As Section 4.2 shows, the first two components carry 98.4% of the variance yet are essentially uncorrelated with the biosignature label, while the chemically informative signal resides in the low-variance tail."

**Replace with:**

> "As Section 4.2 shows, the two highest-variance components carry 98.41% of the spectral variance but classify at chance when used alone, while label-discriminative structure is distributed across the low-variance components."

Keep the tabular-data explanation and the small-sample caveat — both are sound and unattacked.

**Also delete** the stray reference-manager artifact in this section (R2-3): `("Website," n.d.)` appears mid-sentence after the bootstrap p-value.

## §4.2 — the largest rewrite — serves R1-5, R1-7

This section currently argues that PC0/PC1 encode physics while PCs 2–101 encode chemistry. **That argument is withdrawn in full.** Reviewer 1 is correct that principal components are linear combinations of all wavelengths and do not map onto individual atmospheric processes.

All passages quoted in this section were checked against `Final Paper Manuscript (9).pdf` and match verbatim.

**Delete these four.** The attribution runs through the whole subsection, not only the two sentences that state it most plainly — deleting fewer leaves the retracted claim standing in the section that is supposed to carry the retraction.

> "Meanwhile, Principal Components 2 through 101 successfully capture these vital high-frequency chemical absorption features."

> "By analyzing the component loadings and reconstructing the spectra, we determined that PC0 acts as a nearly perfect proxy for the mean transit depth (r = 0.9998) and PC1 captures the overall spectral slope. Thus, they encode broad physical properties like planetary radius and stellar continuum level rather than specific, high-frequency chemical absorption features."

> "Quantitative correlation analysis confirms that the first two principal components (PC0 and PC1) are physically tied to broad systematics rather than chemical features. Although these components explain approximately 98.41% of the total spectral variance, they showed practically zero quantitative correlation with the target biosignature labels (Pearson correlation coefficients of r = 0.0075 and r = -0.0235 respectively)."

The third is the one the reviewer addressed directly: the Pearson coefficients are the evidence he rejected as insufficient, and the AUC analysis below replaces them. Removing the two conclusion sentences while leaving the correlation evidence in place would answer him only halfway.

Fourth, the opening sentence of the subsection, which promotes the feature-engineering story to a finding (this also serves R1-1 and R2-5, and matches the parallel removal in §5):

> "A central finding of this research is the undeniable importance of Principal Component Analysis (PCA) as both a dimensionality reduction and denoising step."

Replace it with a plain statement of what the subsection establishes:

> "Dimensionality reduction was necessary for the neural architectures, and the location of label-discriminative structure within the reduced space is characterised below."

The r = 0.9998 measurement may be **retained as a descriptive fact** — PC0 is a near-perfect proxy for mean transit depth — provided it is not used to infer that chemistry is absent from PC0. If it is kept, state it as a description of PC0 alone and do not pair it with a claim about PCs 2–101.

**Figure 3's caption carries the same attribution and must be rewritten**, otherwise the retracted claim survives in the figure legend after the body text is corrected. Current caption:

> "Figure 3: PCA Scree Plot illustrating the explained variance ratio for the first 105 principal components on a logarithmic scale. The first two components (PC0 and PC1) account for 98.408% of the total variance but capture broad physical systematics. The next 100 components (indices 2–101) capture approximately 1.592% of the variance, representing the high-frequency chemical absorption features required for biosignature classification."

Replacement:

> "Figure 3: PCA scree plot showing the explained variance ratio for the first 105 principal components on a logarithmic scale. The first two components account for 98.408% of the total variance and the next 100 (indices 2–101) for approximately 1.592%. Explained variance does not track discriminative value: the two highest-variance components classify at chance when used alone, while the low-variance tail carries the label-discriminative structure (Section 4.2)."

**Keep** the PCA-necessity ablation ladder (raw CNN 50.56% → PCA 64.51% → standardised 75.53%). Dimensionality reduction genuinely matters; what is withdrawn is the claim that the *variance ordering* is what makes it work.

**Replace the interpretation with the measured claim:**

> "No individual principal component is strongly discriminative: the maximum single-feature AUC across all 102 components is 0.663 on the training set (0.654 on held-out data), while the component carrying 97.1% of the variance is indistinguishable from chance (AUC 0.506) and the next (AUC 0.530) is no more informative than the average low-variance component (mean AUC 0.531 across components 2 to 101). Classification performance arises from aggregating many weakly informative components rather than from a few dominant ones. Fourteen of the twenty most discriminative components fall outside the twenty highest-variance components, and retaining only the two highest-variance components yields 52.13% accuracy — chance, on balanced classes — while discarding them costs nothing (88.58% → 88.25%). Variance rank and discriminative rank are therefore substantially decoupled in this feature space."

**Add the ablation table** from `final_results/H2_pc_range_ablation.txt` and **the per-component figure** `final_results/pc_discriminative_power.png` as a new figure.

**State the retraction explicitly:**

> "An earlier interpretation of this feature space attributed the leading components to physical structure and the remainder to chemical absorption. That attribution is not supported: principal components are orthogonal linear combinations of all wavelength channels and do not correspond to individual atmospheric processes. The analysis is therefore restricted to the location of label-discriminative information, which is directly measurable."

**Cite the precedent** (also serves R2-3's request for foundational references): Jolliffe (1982), *A note on the use of principal components in regression*, on low-variance components being important predictors.

**Add the supervised-projection comparison** (this is R1-7's requested ablation, and it belongs here rather than in a separate section because it bears directly on what PCA is and is not doing). From `final_results/H2_supervised_dr.txt`:

> "To test whether the variance ordering is an appropriate selection criterion, principal components were compared against a supervised projection at matched dimensionality, holding the classifier fixed. Restricted to two components, partial least squares reaches 60.70% against 52.13% for the two highest-variance principal components. Beyond approximately five components the two projections become indistinguishable, and at 102 components they differ by 0.33 percentage points. Principal component analysis is therefore an adequate basis that retains the discriminative information once sufficient dimensions are kept, rather than a transformation that isolates it."

The useful comparison is `PLS + XGBoost` against `PCA + XGBoost`, which holds the classifier fixed so only the projection differs. Comparing linear PLS-DA against nonlinear PCA + XGBoost confounds the projection with the classifier and measures linearity instead.

**Report the two analyses together**, and state why either alone misleads: univariately every component is weak (mean AUC 0.518 for the leading pair vs 0.531 for the tail), but multivariately the leading pair is inert while the tail is collectively strong. Single-feature AUC cannot see interactions; the ablation cannot resolve individual components.

## §4.3 Error Analysis — keep, and strengthen — serves R1-4

This section was not attacked. Keep it, and add the margin analysis (`final_results/H2_label_margin.txt`, figure `label_margin.png`) as its quantitative counterpart:

> "Classification accuracy depends strongly on how far a planet's abundances lie from the labelling cutoff. Binning the test set by that distance, accuracy rises from 61.9% within 0.25 dex of the cutoff to 95.3% at distances beyond one dex. Because class balance varies between bins, each bin is compared against its own majority-class baseline: in the nearest bin the classifier does not exceed that baseline at all, and its mean predicted probability is 0.515. Within a quarter-dex of the cutoff the labelling separates atmospheres that are physically near-identical, so no spectrum can recover the distinction; the errors concentrated there reflect the arbitrariness of a threshold convention rather than a limitation of the classifier. This regime accounts for 8.8% of the test set."

Immediately follow it with the counterpart concession:

> "The converse also follows. Because abundances are drawn uniformly across several decades, 64.8% of test planets lie at least one dex from the cutoff, where the deciding abundance differs by more than an order of magnitude between the classes. Overall accuracy is therefore weighted toward well-separated cases, and the headline figure should be read alongside this breakdown rather than in place of it: the method separates chemically distinct atmospheres reliably, and approaches the labelling convention's own resolution limit as the distinction narrows."

Pair both with the existing error-clustering discussion, which describes the same effect qualitatively. The two halves of the table do different work and both are needed: one identifies the part of the residual error attributable to the labelling convention, the other the part of the headline attributable to easy cases. Reporting either alone misstates the result.

## §4.4 Calibration — add one caveat — serves R1-3

The reasoning here is sound. Add, after the Brier discussion:

> "This calibration advantage is measured in-distribution. Under injected observational systematics it degrades faster than accuracy: at an effective SNR of 10 the Brier score rises from 0.080 to 0.139 while accuracy falls eight points. Calibration should therefore be re-established on data representative of the deployment distribution rather than assumed to transfer."

## §4 — new Robustness subsection (R1-3, R1-8)

Place after §4.4. Content from `final_results/H2_domain_shift_sweep.txt` and both `domain_shift_*.png` figures.

Cover, in order: the method (models trained once, whole preprocessing chain frozen, only test sets perturbed, strengths anchored to the measured noise floor); the degradation table; the finding that calibration-type systematics are largely tolerated while correlated noise is the dominant vulnerability; the out-of-envelope split with its sample-size control; and the whitening robustness result.

**On whitening, concede plainly** (R1-8):

> "A controlled comparison in which two otherwise identical networks differ only in whether this standardisation is applied — each trained five times to average over initialisation — shows that it reduces robustness. Averaged over restarts, the standardised network begins 3.6 ± 3.1 accuracy points ahead on clean data and ends 11.0 ± 1.7 points behind at an effective signal-to-noise ratio of 5, with the curves crossing by SNR 12 in four of five restarts; normalised by headroom above chance, it degrades 1.6–2.8 times faster in every perturbation family with a measurable effect. This supports the concern that rescaling every low-variance component equally raises the weight of components dominated by systematics as much as those carrying signal — and the clean-data benefit that motivated the standardisation is itself marginal once training variability is accounted for."

### Independent radiative transfer code — present this before the opacity swap

The two belong together and the order matters: the code result is the control
that makes the opacity result interpretable. Content from
`final_results/H2_exotransmit.txt`.

> "To separate the influence of the radiative transfer implementation from that
> of the underlying opacity data, transmission spectra were recomputed for the
> held-out test planets with Exo-Transmit, an independently developed code
> written in a different language and using a different solver. The molecular
> cross sections were held fixed: the opacity tables used by the training
> simulator originate from Exo-Transmit and were supplied to it unchanged, so
> the radiative transfer implementation is the only quantity that differs. The
> two codes agree closely on the spectra themselves, with a median per-planet
> correlation of 0.997 and mean transit depths within 0.7%. Classification
> accuracy falls from 88.9% to 84.8%, a reduction of 4.1 percentage points."

Then the caveat, which must be stated because it accounts for much of the 4.1
points and a reader comparing codes would want it:

> "The two implementations differ in one physical assumption that cannot be
> reconciled by configuration: Exo-Transmit treats gravity as constant through
> the atmosphere, whereas the training simulator integrates hydrostatically with
> gravity decreasing with altitude. Comparisons with all molecular opacity
> disabled isolate this effect, showing mean transit depths agreeing to 0.4%
> while spectral contrast differs by 4–12%, scaling with the atmospheric extent
> relative to the planetary radius. The quoted reduction is therefore an upper
> bound on the difference attributable to the numerical scheme alone."

And the sentence that makes the pair worth reporting, to be placed after the
opacity result below:

> "Comparing the two experiments, substituting the opacity data costs four
> times as much accuracy as substituting the radiative transfer code, and close
> to six times when the ozone tabulation is replaced as well. The
> transferability of this method is therefore limited principally by uncertainty
> in molecular line lists rather than by the choice of forward model, which is a
> constraint shared by any retrieval-based analysis of real observations rather
> than an artefact of the simulation framework used here."

### Opacity database — the single-variable result

Content from `final_results/H2_opacity_swap.{txt,csv}`. Place this **before** the aerosol results: it is the most tightly controlled experiment in the section (same code, same planets, only the opacity tables differ) and it establishes that the pipeline's failure mode here is bias rather than signal loss, which frames how the aerosol degradations should be read.

> "The molecular opacity data were replaced with an independent set of line lists — ExoMol POKAZATEL for H₂O, YT34to10 for CH₄ and UCL-4000 for CO₂ — and the forward model recomputed for the same test planets, holding the radiative transfer code, atmospheric structure, geometry and labelling rule fixed. Accuracy falls from 88.9% to 72.8% and the Brier score rises from 0.080 to 0.203, with individual predictions changing for 26.8% of planets. The alternative compilation provides no ozone opacity, so this arm leaves ozone on its original tabulation and is internally coherent in its spectroscopy. Replacing ozone as well, using HITRAN cross sections, extends the substitution to every spectrally active molecule and to both species defining the class label: accuracy then falls to 66.0% and the Brier score to 0.253, with a third of predictions changing. The two figures bracket the sensitivity of this benchmark to its opacity data."

> "The degradation does not arise from a loss of spectral signal: median feature amplitude is essentially unchanged. It is a shift in the decision boundary. Replacing the three non-ozone absorbers raises the predicted-positive rate from 49.5% to 61.8% against a true rate of 50.3%; replacing ozone as well lowers it to 44.0%."

### Clouds

Content from `final_results/H2_aerosol_paired.{txt,csv}` and `cloudy_generalisation.png`. The paired files are the source for every aerosol figure quoted here; `H2_cloudy_evaluation.txt` and `H2_hazy_evaluation.txt` are the unpaired runs and their degradations do not match, so do not quote from them. Present clouds **after** the injected systematics, so the two are read on a common scale.

Explain the setup first — the deck is added to the forward model itself rather than applied to finished spectra, so this is different physics rather than a perturbation, and the classifiers never saw a cloud in training:

> "Clouds were introduced into the forward model as an optically thick grey deck at fixed cloud-top pressures, with the classifiers trained only on cloud-free spectra. The held-out test planets were re-rendered with the deck added and nothing else altered, so each cloudy spectrum is paired with a cloud-free spectrum of the same planet. Performance degrades monotonically with cloud-top altitude: 86.6% for a deck at 10⁵ Pa, 78.4% at 10⁴ Pa, 71.6% at 10³ Pa and 63.7% at 10² Pa, against 88.9% cloud-free, while the Brier score rises from 0.080 to 0.271 across the same range."

**Note on the comparator.** These figures are measured on the committed test planets re-rendered with the aerosol, so each aerosol spectrum is paired with a cloud-free spectrum of the same planet and the clear-sky baseline is a within-planet comparator. Quote the 10² Pa deck, not the 10¹ Pa one, which is featureless.

Then the ceiling caveat, so the bottom row is not read as a failure of the classifier:

> "Below approximately 10² Pa the deck suppresses feature amplitude to under a third of its cloud-free value, and at 10¹ Pa to three per cent. Accuracy approaches chance in this regime because the spectra retain almost no diagnostic structure, which reflects the absence of retrievable signal rather than a specific failure of the classifier. The informative range for this comparison is 10⁵ to 10² Pa; at 10² Pa the classifier still scores 13.7 percentage points above chance, whereas at 10¹ Pa it scores 2.3."

Then the applicability statement, which summarises the measured degradations:

> "Taken together these results bound the conditions under which the method is usable. Performance is essentially unaffected by mild aerosols: a deck at 10⁵ Pa costs 2.3 accuracy points and a haze retaining 95% of the clear feature amplitude costs 0.04, which is within the noise of the measurement. Degradation becomes substantial once an aerosol removes roughly a fifth of the feature amplitude: a haze at 0.79 costs 7.0 points and a deck at 10⁴ Pa 10.5. At strong muting both prescriptions reach a loss of about 25 points — 25.3 for a deck at 10² Pa and 25.6 for the densest haze tested. Usability therefore depends primarily on how much feature amplitude the aerosol leaves intact."

**Quote the MLP only as a restart average, stated as such.** Single training runs vary by several points (cloud-free baseline 76.5–82.2% across runs); the restart-averaged values paired with these aerosol sets are in `final_results/H2_aerosol_paired.{txt,csv}`. If the manuscript reports the MLP, use wording of the form: "MLP figures are means ± one standard deviation over five training restarts; individual runs vary by several accuracy points." XGBoost is quoted as a single value and needs no such treatment.

**Describe XGBoost's reproducibility precisely.** Given fixed inputs it reproduces exactly, so its figures need no restart averaging and may be quoted as single values. Separately, `subsample=0.8` means the sampled rows depend on training-row ordering even at fixed `random_state`, so a different shuffle seed moves accuracy by roughly 0.4 points; the pipeline pins the ordering with `shuffle(..., random_state=42)`. Both statements hold, and neither licenses an unqualified "deterministic" without the fixed-input condition.

**Then the haze comparison** — the reviewer asked for prescriptions, plural, and named hazes. Content from `final_results/H2_aerosol_paired.{txt,csv}` and `hazy_generalisation.png`. Present the haze after the deck:

> "A second aerosol prescription was tested. Photochemical haze was modelled with the Lee et al. (2013) parameterisation of Mie extinction for 0.1 µm particles, at five particle densities. The haze degrades performance monotonically with density, from 88.9% cloud-free to 63.3% at the highest density tested."

**State the limits of the study in the same subsection**, not only in §5: the injected systematics are parametric models rather than instrument-derived; four of the seven requested validation axes are covered in full and three in part; and the injected-systematics and aerosol experiments perturb or reconfigure TauREx spectra rather than leaving the code — only the cross-code experiment does that.

## §5 Conclusion — serves R1-1, R1-2, R1-5, R2-5

**Delete** (R1-5):

> "Although PC0 and PC1 captured most of the broad physical variance, later components preserved chemically informative spectral structure"

**Remove** the promotion of the PCA feature-engineering story to "a central finding" — it is not among the stated contributions and is now withdrawn.

**Add** the robustness findings and an explicit limitations paragraph naming: labels derived from generation parameters rather than retrieval; no aerosols in the training data (their out-of-distribution cost is now quantified in §4.5, but training with aerosols was not attempted); 1D atmospheres; parametric rather than instrument-derived systematics; a single 1/λ stellar-contamination proxy rather than a spot model; and no transfer-learning experiment (the domain-shift half of that axis is covered).

**"Single radiative transfer code" must not appear in this list.** Training uses one code; evaluation does not. State it with the numbers:

> "The models are trained on spectra from a single radiative transfer code. Evaluated on spectra recomputed for the same planets with an independent code, accuracy falls from 88.9% to 84.8%; evaluated on spectra recomputed with an alternative molecular opacity compilation, it falls to 72.8%."

Name the required next steps: **retrieval-derived labels** (the primary one — see R1-4), and the three partial robustness axes — a physical stellar spot/faculae contamination model, instrumental systematics derived from Ariel's published noise model, and transfer-learning (H₂→N₂) as an optional extension to the domain-shift work. Cross-simulator *evaluation* is done and should not be listed as outstanding.

## Acknowledgments — new section (R2-2)

The manuscript has no Acknowledgments section. Add one before References, disclosing all assistance received — editorial, technical, analytical, writing — including any AI tools used in analysis or drafting.

## References (R2-1, R2-3)

- Replace refs 1 and 2 (bare undated NASA web pages) with citable sources or remove
- Check ref 9: the prose describes CO detection via transmission spectroscopy, which is Snellen et al. 2010 — the entry is correct and the repository file has been fixed, but confirm the prose and entry still agree after editing
- Fix ref 39: Pan & Yang is cited as 2009; *IEEE TKDE* 22(10) is 2010
- Verify all 42 links resolve; only four currently carry URLs
- Consider adding Krissansen-Totton et al., *Disequilibrium biosignatures over Earth history* — already in `reference papers/` but uncited, and it supports the CH₄/O₃ disequilibrium rationale in §1
- Add Jolliffe (1982) for §4.2, and Grinsztajn et al. (2022) / Shwartz-Ziv & Armon (2022) for the tabular-data argument in §4.1 — all three now in the repository

## Global pass (R2-6)

Tense and person consistency. The manuscript currently uses first-person plural throughout ("we simulated", "we used"). Reviewer 2 asked for this explicitly — "Verify that you have used past perfect tence and third person throughout the manuscript wherever applicable" — so it is not an assumption and needs no confirmation with the editor. Just do it. Note he asks for past perfect as well as third person; check both.

## Final check before resubmission

- Every Manuscript-column cell in the Status table filled with a section and paragraph reference
- No occurrence of "chemically informative", "chemically critical", or "isolates the chemical" remains
- The response document quotes each of the 15 comments verbatim and names where each was addressed
- README numbers still match the manuscript after any results change

---

## Working notes

**Reproduction environment.** Use `~/tfenv/bin/python` for everything. It is pinned
to the manuscript's versions — Python 3.10, TensorFlow 2.21.0, scikit-learn 1.7.2,
XGBoost 3.2.0 — and Random Forest reproduces at 86.51% ± 1.96%, matching Table 3
exactly. It also carries the cloud-capable MultiREx fork and PyMuPDF for reading the
review and manuscript PDFs, and has pip available.

This matters because the environment cannot be rebuilt casually: the system Python
is 3.14, which has no TensorFlow wheels, and there is no pip or sudo available
system-wide. `~/tfenv` was bootstrapped via `uv` with a standalone CPython 3.10.
Prefer patching it over recreating it. Temporary scratch directories are wiped
between sessions, so nothing durable should live there.

**Verbatim reviewer comments** are in two gitignored, local-only files, with
`Scholastica.pdf` as the source. `revision/scholastica_reviews_verbatim.txt` is a
clean extraction of the reviews as written, with an ID map at the top showing which
passage each comment ID refers to — read this one when you need the exact wording.
`revision/reviewer_response_tracking.md` interleaves the same text with the response
plan and evidence. Both stay out of
the repository because peer review correspondence is confidential; the response
document submitted to the editor quotes each comment in full from that file.

**Benchmarking newly generated data.** Any newly generated dataset must be
compared against a baseline generated in the same run, or produced by
re-rendering the committed test planets themselves — not against the committed
88.92% figure. MultiREx's parameter sampling is not guaranteed stable across
versions of the generation code, so a fresh draw and the committed sets are not
necessarily the same population. The aerosol and opacity experiments both
re-render the committed planets for this reason. It applies directly to the next
planned R1-3 experiment: transfer learning H2→N2 requires regenerating N2 at 550
bins, and that data needs its own matched baseline.

**Headline baseline — settled.** Both **88.91%** and **88.92%** appear as the clear-sky XGBoost baseline, and they are two summaries of one run rather than two runs. 88.92% is the mean of the five per-set accuracies (87.76 / 90.07 / 88.67 / 89.94 / 88.15, mean 88.9175%); 88.91% is the accuracy pooled over all 2697 planets (2398 correct, 88.9136%). They differ because the sets have unequal sizes, and both round to **88.9%**, which is the figure to quote — see "Reporting precision" below.

Which appears where is a property of the script, not an inconsistency: the domain-shift sweep and the supervised-DR comparison report the per-set mean, so 88.92% is correct where those files are the source; the aerosol, opacity-swap and Exo-Transmit files report pooled, so 88.91% is correct there. **Prefer pooled** where a choice exists — every planet counts once, and it is what the paired experiments report. Only mixing the two inside a single comparison would be an error.

**Reporting precision — one decimal, and why.** Quote accuracies to one decimal
place throughout: the headline is **88.9%**, degradations are −16.1, −22.9,
−25.3 and so on.

With 2697 held-out planets the binomial standard error on the headline is 0.60
percentage points, a 95% interval of roughly 87.7–90.1%. A second decimal
therefore claims about a hundred times more precision than the data supports.
Two figures circulate in the older results files — 88.92% is the mean of the
five per-set accuracies (87.76 / 90.07 / 88.67 / 89.94 / 88.15), 88.91% is the
accuracy pooled over all planets, and they differ because the sets have unequal
sizes. The gap is 0.004 points against a 0.6-point uncertainty, so it is not a
disagreement, and both round to 88.9%. Prefer pooled where a choice is needed:
every planet counts once, and it is what the paired experiments report.

**Absolute figures and differences have different precisions, and the
differences are the stronger claim.** Two absolute accuracies each carry ±0.6
points, so gaps between them below roughly 2 points cannot be argued from —
the same rule already stated for the MLP, and it applies to XGBoost too. Paired
degradations are tighter because they compare identical planets: the opacity
result is −16.1 with a paired standard error of 0.95 points (95%: −17.9 to
−14.2). That is the payoff of the paired design and the reason separately drawn
comparison populations were abandoned.

**XGBoost determinism, precisely.** Given fixed inputs XGBoost reproduces
exactly, which is why its figures need no restart averaging. Separately,
`subsample=0.8` means the sampled rows depend on training-row ordering even at
fixed `random_state`, so a different shuffle seed moves accuracy by roughly 0.4
points. Both statements are true and are not in tension: the pipeline pins the
ordering with `shuffle(..., random_state=42)`. The MLP is not deterministic even
with inputs fixed, and does need restart means ± σ.

**Resolution confirmed.** MultiREx `wavenumber_grid` uses `np.logspace`, giving constant resolving power. Measured from the data columns, R = λ/Δλ = 199 across the band. The manuscript's "550 bins at R = 200" is correct. Note that two earlier commit messages describe the pipeline as "R=550" — that is the bin count, not the resolving power.

**Unverified physics risk — resolved, and the answer is a data limitation (2026-07-20).** The strongest ozone infrared band (9.6 μm) lies outside Ariel's 0.5–7.8 μm window; in-band detection was assumed to rely on the weak Chappuis band near 0.6 μm and a feature near 4.74 μm. The octave-occlusion test (`analyze_o3_band_occlusion.py`) showed classification runs predominantly through 4.74 µm: masking 0.5–1 µm costs 6 points on the O₃-diagnostic subset while masking 3.9–7.8 µm collapses it below chance.

The reason is the opacity data, and it is not specific to this simulator. **MultiREx's Exo-Transmit O₃ table contains no data below 1.73 µm** — the 0.5–1 µm octave is 0% covered and 1–2 µm only 10.7%, the remainder filled with a 1e-60 sentinel. Checking the alternatives shows the same limitation everywhere: DACE's ozone (HITRAN2020) is tabulated only over 0–7000 cm⁻¹, i.e. λ ≥ 1.43 µm, and petitRADTRANS's HITRAN ozone, though nominally spanning 0.3–28 µm, carries values of order 1e-43 cm² across the Chappuis region — negligible against 1e-19 cm² in the infrared.

The common cause is physical rather than a defect in any one product: ozone's visible absorption is an electronic continuum, not a line spectrum, so it does not appear in line-list-derived opacity tables of the kind every exoplanet radiative transfer code uses. Obtaining it requires separate continuum cross-section data, which none of these products bundles.

This belongs in the manuscript as a stated limitation of the forward model (§3 assumptions, and R1-3): simulated ozone is spectrally invisible shortward of 1.73 µm, which real ozone is not.

**Figure 4 check.** `final_results/plots/corner_plot_errors_scatter_xgboost.png` was deleted from the working tree while the CNN, MLP, and Random Forest versions were regenerated. `final_results/figure 4.png` still exists — confirm it is current before resubmission.
