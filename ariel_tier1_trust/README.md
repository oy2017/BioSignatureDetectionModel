# Can a simulator-trained classifier for Ariel Tier-1 spectra be trusted?

Supplementary material for the technical note *"Can a simulator-trained classifier for Ariel Tier-1 spectra be
trusted?"* (manuscript source:
[`v3/jhss/note_src.md`](../v3/jhss/note_src.md); built files in [`v3/jhss/`](../v3/jhss/)).

## In brief

- **What was tested.** The molecular classifiers proposed in the Ariel consortium's Tier-1 population study
  (Mugnai et al. 2021, AJ 162, 288), whose code was not released, re-implemented from the published description.
- **Reproduction.** All 48 published accuracies reproduced within 3.7 points (mean difference +0.4, correlation 0.96).
- **Where it could be trusted.** An alternative opacity database (−2.9 points), Exo-Transmit's own radiative-transfer
  code (−3.5) and omitted HCN and C₂H₂ (−0.1).
- **Where it could not.** Haze at the weak end of the range retrieved for observed hot Jupiters (−17.7), star spots
  (−10.8 at 20 %), high cloud decks (−9.8). Under haze it reported CH₄ and H₂O absent on nearly every planet that
  contained them, became more confident, and its own ensemble gave no warning; 87 % of the loss was recoverable.
- **Why.** The three optical photometric inputs: removing them cut the haze loss from 17.7 to 5.6 points
  (0.5 with AIRS bins only), at 3.8 points of clean accuracy. Normalisation was not the cause.
- **Does the answer transfer?** No. A second classifier, for Ariel's C/O objective, had a nearly opposite profile:
  omitted HCN and C₂H₂ cost it 26–30 points on carbon-rich real targets, haze only 5 points at Tier-3 binning.
  Each classifier needs its own stress test.

## Why the re-implementation is faithful

1. **Every element traces to the publication** (table below, page numbers of the arXiv version, arXiv:2110.00503).
2. **The published numbers are reproduced**, including their patterns: accuracy rises with the abundance threshold
   (15 of 16 classifier–molecule pairs; 16 of 16 published) and H₂O is the hardest molecule at 10⁻⁵ and 10⁻⁴ in both.
3. **The reproduction test discriminates.** An earlier, mistaken reading (seven Tier-1 points, 1 Pa atmosphere top,
   noise fixed at the requirement) missed by 10 points on average and by up to 38; only 14 of 48 cells were within 5.
4. **Readings of unstated details are justified from the text and figures**, and the haze result was checked against
   the implementation choices it could depend on (photometric noise ×3 and ×10, haze particle size).
5. **Unavoidable deviations were measured**: ExoMol instead of Exo-Transmit opacities costs the classifier 2.9 points;
   collision-induced absorption changes spectra by ≤ 3 ppm; the payload noise model reproduces the candidate sample's
   Tier-3 transit counts within 0.07 dex and matches the published noise bands of the paper's Figure 1.

| Element | Published statement (Mugnai et al. 2021) | Re-implementation |
| :-- | :-- | :-- |
| Planet list | Ariel candidate list, 1,000 planets incl. predicted TESS discoveries (p. 4) | 965 known planets, 2026 Mission Candidate Sample |
| Populations | POP-III: each planet ×4 (training); POP-I: each planet once (test) (pp. 5–6) | Same |
| Temperature | 0.7–1.05 × equilibrium temperature, isothermal (p. 4) | Same |
| Atmosphere grid | 100 layers, 10⁻⁴–10⁶ Pa; H₂/He with He/H₂ = 0.17 (p. 4) | Same |
| Abundances | CH₄, H₂O, CO₂, NH₃ log-uniform: 10⁻⁷–10⁻² (POP-I), 10⁻⁹–10⁻² (POP-III) (pp. 4, 6) | Same |
| Clouds | Grey opaque deck, 5 × 10²–10⁶ Pa (p. 4) | Same |
| Opacities | ExoMol; H₂–H₂ and H₂–He CIA (p. 5, Table 2) | Exo-Transmit tables; no CIA (measured effect above) |
| Binning | Tier 3, R = 20/100/30 (pp. 2, 5) | Same, 104 points incl. the three photometric points |
| Noise | ArielRad Tier-3 noise per bin rescaled to the Tier-1 transit count (p. 5) | ExoSim 2 payload model calibrated on the catalogue, per bin, at the integer Tier-1 transit count |
| Normalisation | "Each example spectrum is normalised to zero mean and unit dispersion" (p. 11) | Same |
| Classifiers | scikit-learn defaults: KNN k = 5, MLP 100 units, RFC, SVC one-vs-one (p. 12) | Same (defaults unchanged since 0.22) |
| Evaluation | Trained on POP-III, tested on POP-I, three thresholds (p. 18) | Same |

## What is in this directory

| Path | Contents |
| :-- | :-- |
| `figures/` | Figures 1–4 of the note |
| `results/reproduction/` | The 48-cell reproduction |
| `results/consortium_stress_test/` | Every departure: accuracy, ceilings, randomized grid and held-out variants, decline rules, calibration and conformal coverage, host dependence, trade-off (`alfnoor_trust.txt` is the readable summary) |
| `results/haze_mechanism/` | Input variants (the causal test), photometric-noise and particle-size checks, the same test on the carbon-rich classifier |
| `results/carbon_rich_classifier/` | The second classifier: stress test at Tier 3 and Tier 1, omitted absorbers on the known Ariel targets, calibration, mechanism |
| `results/forward_model_checks/` | Effect of the TauREx pressure-unit correction |
| `alfnoor_trust_expectations.md` | Expectations for the consortium classifier, committed before the stress test ([commit eee6337](https://github.com/oy2017/BioSignatureDetectionModel/commit/eee6337), 2026-09-13 08:53 PDT) |
| `collect.py` | Copies these files from `v3/results/`; re-run after any recompute |

## Reproducing the results

The scripts live in [`v3/`](../v3/) and run from there with the environment in [`requirements.txt`](../requirements.txt).
The forward model is TauREx 3 through a MultiREx fork with two corrections, both recorded in
[`v3/MULTIREX_FORK.md`](../v3/MULTIREX_FORK.md): Exo-Transmit opacity pressures read in Pa (reported upstream as
[TauREx issue 172](https://github.com/ucl-exoplanets/taurex3/issues/172)) and a grey cloud deck kept alongside a haze.
[`v3/forward_model_guard.py`](../v3/forward_model_guard.py) refuses to render if either is missing.

| Result | Command (from `v3/`) |
| :-- | :-- |
| Carbon-rich grid, classifier and its stress test | `bash rerun_audit.sh` (full regeneration, about six hours) |
| Consortium populations | `python alfnoor_faithful.py --render --jobs 8` |
| Reproduction (Figure 1, Table A1) | `python alfnoor_faithful.py --fit --natives faithful --layout tier3_r20 --noise radiometric` |
| Consortium stress test (Figure 2, Table 3) | `python alfnoor_trust.py --render --jobs 8` then `python alfnoor_trust.py --fit` |
| Haze mechanism (Figure 3, Table 4) | `python alfnoor_haze_mechanism.py` |
| Robustness of the haze result; carbon-rich input test | `python haze_generality.py --parts carbonrich photnoise particles` |
| Omitted absorbers on the known targets (Figure 4a) | `python mcs_absorbers.py` |
| Figures | `python plots/note_fig1_reproduction.py` (and `note_fig2`–`note_fig4`) |
| Manuscript | in `v3/jhss/`: `python make_tables_note.py && python render_note.py && python build_manuscript.py both` |
