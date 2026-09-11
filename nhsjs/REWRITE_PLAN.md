# NHSJS rewrite plan, version 2 (full regeneration)

Written 2026-09-08, replacing the version-1 plan of the same day. Version 1 condensed
the existing R = 200 study; version 2 rebuilds the study, because the author has said
cost is not a constraint and the JHSS reviewer's objections are answerable only by new
data. Sources for the journal rules: `Submission Guidelines - NHSJS.pdf`,
`NHSJS Article Submission.pdf`, the templates in `templates/`, and the three pages the
author cleared (submission-guidelines, submission-types, peer-review-process).

**Scope split (decided 2026-09-08):** this is *paper 1*, for NHSJS, with no retrievals.
The retrieval study is *paper 2*, for JHSS, outlined in `../jhss_paper2/OUTLINE.md`; it
reuses paper 1's grid, noise model and frozen classifier and runs in the background while
paper 1 is under review. Paper 1 stays retrieval-free so that paper 2's central result
does not appear anywhere first.

Everything in this directory from the condensation attempt (`NHSJS_standard.md`,
`build_docx.py`, `citation_mapping.txt`, the two DOCX files) is superseded; only the
template-handling and reference-formatting code in `build_docx.py` will be reused.

---

## 0. Decisions, with recommendations

1. **Primary spectral grid: Ariel's delivered Tier 3 configuration, not uniform R = 100.**
   The author asked for R = 100. Channel-matched is better, for the same reason R = 100
   beats R = 200: it is the resolution the mission delivers. The real wavelength solutions
   are already in `ariel_noise_model/` (`nirspec_wlsol.csv`, `airs0_wlsol.csv`,
   `airs1_wlsol.csv`), so the bins can be the instrument's own: three photometric bands
   below 1.1 µm, NIRSpec R ≈ 15 over 1.1–1.95 µm, AIRS-CH0 R ≈ 100 over 1.95–3.9 µm,
   AIRS-CH1 R ≈ 30 over 3.9–7.8 µm, about 100 bins in total. Uniform R = 100 and R = 200
   are then reported as two idealized comparators on a "resolution ladder", one figure,
   using the same planets. The risk is that accuracy at the delivered configuration is
   low, because the 3.9–7.8 µm channel carrying the O₃ and CH₄ bands is coarsest. That is
   the honest number, and the ladder shows exactly what the idealization was buying.
2. **Regenerate the grid rather than bin the old one.** Regeneration fixes the coupled
   sampling the reviewer flagged, stores noise-free spectra so noise is injected
   explicitly, and lets the set be ten times larger. A forward model costs 0.04 s and the
   machine has 22 cores, so 30,000 spectra take minutes.
3. **No retrievals in this paper.** They are paper 2 (JHSS). Paper 1 answers the label
   objection with threshold sensitivity, the margin analysis, and a disclosed limitation.
4. **Optional, second population: an N₂-dominated terrestrial grid.** 1–4 R⊕, 250–800 K,
   generated with the same code. It answers "no habitable-zone analogues" directly, but
   at Ariel's noise these spectra may be near-featureless, and it doubles the shift work.
   Recommendation: generate it, run the in-domain comparison only, and
   report it as one subsection; do not run the seven shift axes on it.
5. **Models: XGBoost, Random Forest, MLP.** No CNN (decided). With about 100 bins, PCA is
   no longer a necessity but a preprocessing choice, so the preprocessing comparison
   (raw bins, per-channel standardization, PCA, per-spectrum normalization) becomes one
   table row each. Per-spectrum normalization was set aside as out of scope in the
   previous manuscript; in a rebuilt pipeline it belongs in the comparison. Strike it
   here if that is not wanted.

---

## 1. Hard constraints (from the files in this directory and the cleared pages)

| Requirement | Source | Plan |
| :-- | :-- | :-- |
| Sections in this order: Title, Authors and affiliations, Abstract, Introduction, Methods, Results, Discussion, Acknowledgments, References | Guidelines p.5 | Followed exactly. No "Background" section, no "Conclusion" section. Results and Discussion are separate. |
| Abstract 200–250 words, structured Background/Objective → Methods → Results → Conclusions, followed by Keywords | Guidelines p.2 | Target 235 words, 6 keywords. |
| Introduction covers: background/context, problem statement and rationale, significance and purpose, objectives (hypotheses), scope and limitations, methodology overview | Guidelines p.2–3 | Six labelled subheadings inside the Introduction, in that order. |
| Methods (research papers): research design, sample, data collection, variables and measurements, procedure, data analysis, ethical considerations | Guidelines p.3–4 | Seven subheadings. "Ethical considerations" is one sentence (simulation-only, no human or animal subjects, all data and code public). |
| Discussion covers: restatement of key findings, implications, connection to objectives, recommendations, limitations, closing thought; no new information | Guidelines p.4–5 | Six subheadings. Every number in the Discussion must already appear in Results. |
| ≤ 20 pages including figures, tables, appendix; 12 pt; single spaced | Guidelines p.5, form p.3 | Budgeted at 17.7 pages (Section 4.1) so a Word page-count surprise cannot push it over. |
| Two Word files: (1) Standard citations, superscript numbers before punctuation, blind (no name, affiliation, acknowledgments), **file named exactly as the title**; (2) Online citations, full reference in double parentheses at every use, space before `((`, superscript commas between adjacent citations | Guidelines p.5–8, form p.3 | Build both from one source with `build_docx.py` (adapted). Title must therefore be filesystem-safe: no colon, no question mark, no slash. |
| Reference format: `N. ` then all authors as initials + surname, sentence-case title, `Journal. Vol. X, pg. Y-Z, Year, DOI.`; websites give author, page title, full URL, year | Guidelines p.6–7 | Reuse the formatter in `build_docx.py`; re-verify the four references without a DOI (commit 450f9c6). |
| Research Article: novel findings; abstract, introduction, subheadings, **minimum 5 figures or tables**; supporting online material allowed | submission-types page | Plan has 3 tables + 5 figures = 8 exhibits (Section 4.1). |
| Review: editorial screen → 1–2 peer reviewers → advisory board; decisions Accept / Minor / Major / Reject; median 8 weeks, up to 12 in Jul–Dec | peer-review page | Expect a decision around early November if submitted mid-September. |
| Form asks: title, type, abstract, number of citations, prior submission elsewhere | form p.1–2 | Prior submission answer: rejected by JHSS (not a publication); ML4PS 2026 workshop submission is non-archival and the form explicitly permits conference submissions. Say both. |

### 1a. Word, not LaTeX

Decision: **Word (DOCX), built from the official templates.** Reasons:

- The Word route needs two files; the LaTeX route needs three, and its second file is
  *still a Word document* with the online double-parentheses citations, so LaTeX only adds
  a PDF and a `.tex` upload on top of the same Word work.
- The journal's templates are Word files with named paragraph styles (below); their
  editorial screen checks for "each part of your paper" via those styles.
- The paper has no display equations that Word cannot carry, and the existing
  `build_docx.py` already pours a Markdown source into this exact template, renders
  superscript citations, and formats the reference list.

### 1b. The official templates (downloaded 2026-09-08 into `nhsjs/templates/`)

The submission-guidelines web page carries the same text as the PDF in this directory,
plus links to the templates. Files now in `templates/`:
`NHSJS-Manuscript-Template-Standard-Citations.docx`,
`NHSJS-Manuscript-Template-Online-Citations.docx`, `nhsjs.csl`,
`nhsjs-online-citation.csl`. The Standard template is byte-identical to the one
`build_docx.py` was previously reading from an old scratchpad path; point the script at
`templates/` instead.

What the templates fix that the PDF does not spell out:

| Item | Template rule |
| :-- | :-- |
| Page | Letter, 1-inch margins all round |
| Body style `Normal` | Times New Roman 12 pt, single spacing, 6 pt after each paragraph |
| Title style `NHSJS Title` | 18 pt bold, centred |
| Section heading style `NHSJS Section` | 14 pt bold, 12 pt space before (Introduction, Methods, Results, Discussion, Acknowledgments, References) |
| Subheading style `NHSJS Subsection` | 12 pt bold italic (use for the Introduction and Methods subheadings in Section 4 of this plan) |
| Instructions pages | Two pages of instructions plus a guided example precede the template; **delete them**; the manuscript begins on the template's page 3 |
| Abstract and Keywords | Written as `Abstract:` and `Keywords:` run-in labels in Normal style, not as headed sections |
| Captions | `Figure 1 \| one-line description.` with the number, bar and one-line description in bold, panel letters bold; figure caption below the figure, table caption above the table; every figure and table cited in order and placed near first mention, never collected at the end |
| Blind file | No names, affiliations, emails, or acknowledgments |
| Online file author block | All authors in one paragraph, affiliations linked by superscript numbers, each affiliation its own paragraph ending in a period ("Institution; City, Country."), corresponding author marked `*` with email after the affiliations |
| Online citations | `((full reference))` at every use, a space before `((`, identical text every time; adjacent citations separated by a superscript comma (the template describes the Find-and-Replace; `build_docx.py --online` should emit it directly) |
| IRB | Only if human or animal subjects; state "not applicable" in Ethical considerations |
| Supplementary | The guidelines say to attach the images used for figures as supplementary information; the form has a third upload slot, so attach a zip of the figure PNGs |

The two CSL files are for reference managers; the build script does not need them, but
they are the authority if a reference-format question comes up.

---

---

## 2. What the JHSS reviewer said, and what version 2 does about each point

| Reviewer point | Version 1 response | Version 2 response |
| :-- | :-- | :-- |
| Does not detect biosignatures | wording | Same wording rule: triage, label recovery, "CH₄–O₃ co-abundance label" |
| Labels are arbitrary thresholds, not retrieval-derived | 50-planet pilot | Deferred to paper 2. Paper 1 keeps threshold sensitivity (±0.5 dex), the margin analysis with a figure, and a disclosed limitation |
| Not photochemically self-consistent | disclose | Still disclosed. A photochemical model is the one thing not attempted; say so in one sentence |
| No habitable-zone analogues | disclose | Optional N₂ terrestrial grid (decision 5) or, if skipped, disclosed |
| Not applicable to real Ariel observations | R = 100 retraining | The primary grid *is* the Ariel Tier 3 configuration, with ExoRad-shaped noise per stellar temperature; the resolution ladder quantifies the earlier idealization |
| Resolution more favourable than Ariel's relevant channels | R = 100 retraining | Same as above; the 3.9–7.8 µm channel is at its delivered R ≈ 30 |
| Parameter sampling unintentionally coupled | disclose | Fixed: parameters drawn independently with NumPy, not through MultiREx's `clone_shuffled`, and the independence verified and reported (a correlation table in the repository, one sentence in Methods) |
| CNN comparison not meaningful | drop | Dropped |
| Sensitivity testing under shift has been done repeatedly | the ordering result | The ordering result, now measured on the mission configuration with paired re-renders, and the resolution ladder |

---

## 3. Study design, research question, hypotheses

**One-paragraph statement of what is new.** Prior machine-learning screens of synthetic
exoplanet spectra report accuracy inside their own simulator, at idealized resolution,
against labels that are a deterministic function of the injected abundances. This study
builds the screen at the resolution and noise Ariel will deliver, on an independently
sampled grid, and measures which simulator choices its reliability depends on by
re-rendering identical planets under seven controlled changes. The result is a
mission-configured triage classifier with calibrated probabilities, a resolution ladder
that quantifies what earlier idealizations were worth, and a ranked fidelity budget for
anyone training on simulated spectra.

**Research question.** At Ariel's delivered resolution and noise, how reliably can a
classifier trained on synthetic transmission spectra rank CH₄–O₃ candidates, and which
simulator choices does that reliability depend on?

Hypotheses, numbered in the order the Results answer them:

- **H1.** Tree ensembles will out-perform and out-calibrate the MLP at the Ariel
  configuration, and the preprocessing choice will matter less than at R = 200 because
  the feature count is small.
- **H2.** Moving from R = 200 to R = 100 to the delivered configuration will cost
  accuracy monotonically, with the largest step at the last rung, because the
  band-carrying channel is the coarsest.
- **H3.** Errors will concentrate near the labeling thresholds and at low feature
  amplitude, and moving both thresholds by half a dex will change accuracy by only a
  few points, so the result is not an artifact of the cutoffs.
- **H4.** Under each of seven shift axes, accuracy will fall smoothly and each large
  loss will trace to a measurable mechanism (amplitude suppression, chromatic
  distortion, decision-threshold bias).
- **H5.** The axes will not cost equally: line-list data and time-correlated noise will
  cost more than the radiative-transfer implementation or noise colouring. (Version 1's
  ordering, to be re-measured; it may change at the new resolution, and that is a
  result either way.)
- **H6.** Calibration and operating thresholds will degrade under shift faster than
  ranking accuracy does.

---

## 4. Page budget, and what is added, kept, and deleted

### 4.1 The 20 pages, allocated

Template settings give about 600 words per page of text and 46 lines of references
per page; figures are set at 6.0 in width.

| Block | Content | Pages |
| :-- | :-- | --: |
| Front matter | title, blind author line, 240-word abstract, keywords | 0.7 |
| Introduction | 1,300 words, six subheadings | 2.2 |
| Methods | 1,800 words, seven subheadings | 3.0 |
| Results | 2,500 words, five or six subsections | 4.2 |
| Discussion | 1,100 words, six subheadings | 1.8 |
| References | 30 entries | 2.0 |
| Figure 1 | workflow schematic | 0.5 |
| Table 1 | parameter ranges, both grids if decision 5 is taken | 0.5 |
| Table 2 | model × preprocessing comparison at the Ariel configuration | 0.3 |
| Figure 2 | A) resolution ladder, B) reliability curves | 0.45 |
| Figure 3 | A) error rate vs margin to threshold, B) error rate vs feature amplitude | 0.45 |
| Table 3 | seven-axis shift map: Δ accuracy, Δ Brier, mechanism | 0.55 |
| Figure 4 | A) fidelity-budget bars, B) injected-systematics sweeps | 0.55 |
| Figure 5 | precision–recall and threshold transfer, clean vs cloud deck | 0.4 |
| **Total** | 6,800 body words, 5 figures, 3 tables | **17.2** |

If decision 4 (N₂ grid) is taken, it adds one table row and about 250 words and still
fits under 18 pages. Trim order if the build exceeds
18.5 pages: Figure 4 to 5.0 in, Table 1 to the planetary and stellar rows, then the
preprocessing rows of Table 2 to a sentence.

### 4.2 Added relative to the current manuscript

| # | Addition | Why |
| :-- | :-- | :-- |
| A1 | Regenerated grid: independent sampling, noise-free storage, Ariel bin edges, ExoRad-shaped noise, 20,000 training and 5 × 2,000 test spectra | Removes three reviewer objections at once and makes every later number mission-configured |
| A2 | Resolution ladder (delivered / R = 100 / R = 200) on the same planets, Figure 2A | Quantifies the idealization that the old paper was built on |
| A3 | Margin and amplitude error analysis as a figure, Figure 3 | The paper's answer to the label objection, made visible rather than buried in text |
| A4 | Preprocessing comparison at about 100 features, Table 2 | PCA stops being a necessity; the comparison replaces the old feature-space section |
| A5 | Workflow schematic, explicit hypotheses, template-mandated subheadings | Template requirements and readability |
| A6 (optional) | N₂ terrestrial grid, in-domain comparison only (decision 4) | Answers "no habitable-zone analogues" with data |

### 4.3 Kept from the current manuscript, re-run on the new grid

| # | Kept | Note |
| :-- | :-- | :-- |
| K1 | Ariel tiers and the triage bottleneck; Duque-Castaño as precedent | Introduction, condensed to two paragraphs |
| K2 | Label definition, four stratified profiles, threshold-sensitivity and margin analyses | Methods and Results 3; thresholds unchanged so paper 2 can reuse the labels |
| K3 | Three models, grid search, five-set evaluation, McNemar and bootstrap | Re-tuned on the new grid |
| K4 | Calibration: reliability curves, ECE with equal-count bins, Brier | Re-measured |
| K5 | Seven shift axes with paired re-renders and the three mechanisms | Re-rendered for the new test planets; the mechanism paragraphs are re-verified, not assumed |
| K6 | Whitening trade-off | One paragraph if it reproduces at the new resolution; a sentence if it does not |
| K7 | Operating threshold under a cloud deck | Figure 5 |
| K8 | Limitations, next steps, closing thought | Rewritten around what remains: no photochemistry, single forward-model family for training, synthetic only |

### 4.4 Deleted

| # | Deleted | Reason |
| :-- | :-- | :-- |
| D1 | R = 200 as the working grid; the "ceiling then degrade" framing | Replaced by the ladder |
| D2 | The 1D-CNN; the raw-spectrum CNN baseline | Decided |
| D3 | The PCA feature-space section and its three figures (discriminative power, imprint projection, loadings) | With about 100 features PCA is one row of Table 2 |
| D4 | Separate Background section; HRS, Exoplanet Archive, Sing et al. paragraphs | Off-topic context |
| D5 | Hyperparameter, assumptions, PC-ablation, band-occlusion and per-axis tables | Repository, or folded into Table 3 |
| D6 | Confusion matrix, error corner plot, CH₄/O₃ error scatter, PCA reconstruction figures | Carried by a sentence or by Figure 3B |
| D7 | Brier decomposition, Random Forest vote-compression, post-hoc calibration discussion | One sentence each |
| D8 | The coupled-sampling disclosure | No longer true; replaced by the independence check |

---

## 5. Section-by-section content spec

Tags: `[reuse §x.y]` = content exists in `revision/revised_manuscript.md` and can be
condensed from there (numbers will change); `[new]`; `[num]` = from the ledger.

### Title (filesystem-safe)

Working: **A machine-learning triage classifier for exoplanet transmission spectra at the resolution and noise Ariel will deliver**

Shorter alternative: **Which simulation choices matter for a machine-learning triage classifier of Ariel-like transmission spectra**

### Abstract (200–250 words) `[new]`

Background/objective 60 w (Ariel bottleneck; prior screens idealized; RQ). Methods 70 w
(grid, Ariel configuration, three models, ladder, seven paired shift axes). Results 80 w `[num]`. Conclusions 40 w. Keywords: exoplanet
atmospheres; transmission spectroscopy; machine learning; domain shift; probability calibration; Ariel.

### Introduction (1,300 words, six subheadings)

1. Background and context (250 w) `[reuse §1 ¶1–2]`.
2. Problem statement and rationale (250 w): three idealizations in prior screens
   (resolution, labels, single simulator) `[new]`.
3. Significance and purpose (150 w): the one-paragraph statement from Section 3 `[new]`.
4. Objectives (200 w): RQ and H1–H6 `[new]`.
5. Scope and limitations (300 w): label recovery not detection; H₂-dominated
   population, or both populations; no photochemistry; labels are a deterministic
   function of injected abundances, with retrieval-derived labels named as the companion
   study; synthetic only `[new, reuse §1 ¶6–7]`.
6. Methodology overview (150 w): walk Figure 1 `[new]`.

### Methods (1,800 words, seven subheadings; Figure 1, Table 1)

1. Research design (150 w) `[new]`.
2. Sample (250 w): both grids' sizes, cleaning, class balance, four profiles, the
   independence check `[reuse §3.1, new]`. Table 1.
3. Data collection (400 w): MultiREx/TauREx 3 forward model, opacities, no CIA, noise-free
   generation at fine resolution; binning to the Ariel wavelength solutions and to the
   two comparator grids; ExoRad NSR shape per stellar temperature scaled to the stated
   median SNR; noise realizations drawn at training and evaluation `[new]`.
4. Variables and measurements (300 w): label; feature sets (raw, standardized, PCA,
   per-spectrum normalized); metrics `[reuse §3.2, §4.4; new]`.
5. Procedure (250 w): tuning, five-set evaluation, significance tests `[reuse §3.3–3.4]`.
6. Data analysis (400 w): the shift protocol (which axes are paired re-renders, which
   are injected; strengths anchored to the noise floor); the ladder; the threshold and
   margin analyses `[new, reuse §3.6]`.
7. Ethical considerations (50 w) `[new]`.

### Results (2,500 words)

1. In-domain at the Ariel configuration [H1] (450 w): Table 2; Figure 2B; McNemar.
2. Resolution ladder [H2] (250 w): Figure 2A.
3. Label sensitivity and error structure [H3] (400 w): Figure 3; threshold shift
   ±0.5 dex; accuracy vs margin; error rate vs feature amplitude.
4. Domain-shift map [H4, H5] (900 w): Table 3; Figure 4; one paragraph per mechanism;
   the null results.
5. Transfer of calibration and thresholds [H6] (300 w): Figure 5.
6. N₂ population, if taken (250 w).

### Discussion (1,100 words, six subheadings)

Key findings; implications (a fidelity budget for simulator-trained screens; what the
ladder says about published R = 200 results); connection to objectives H1–H6;
recommendations (retrieval-derived labels, named as the companion study in progress;
photochemical grids; consortium simulated observations); limitations; closing thought
`[new, reuse §5 for limitations and closing]`.

### Acknowledgments (online version only)

Mentors `[reuse]`. AI-use disclosure naming tools and uses.

### References

About 28: as in version 1 plus ExoRad, the Ariel wavelength-solution source, and the
cloud and haze contribution references (Lee et al. Mie).

---

## 6. Computation plan

Machine: 22 cores, 31 GB. Forward model 0.04 s after opacity load. Times below are
wall-clock on this machine. Everything writes under `v2/` (new top-level directory) so
the R = 200 study stays intact for the ML4PS submission.

### Stage 0. Prerequisites (half a day)

- Re-download the BT-Settl/PHOENIX solar-metallicity atlas (`phoenixm00_*.fits`,
  STScI reference atlases) into `v2/phoenix/`; the old scratchpad copy is gone.
- Derive Ariel bin edges from `ariel_noise_model/*_wlsol.csv` at the delivered
  resolving powers, plus the three photometric bands; write `v2/ariel_bins.json`.
  Report the bin count.
- Confirm the Mie-capable MultiREx fork still generates a cloud deck (it imports; run
  one spectrum).
- Confirm Exo-Transmit builds and the ExoMol/HITRAN swap tables load.

### Stage 1. Grid generation (one hour)

`v2/generate_grid.py`: draw every parameter independently with NumPy (log-uniform for
gases, uniform otherwise, the version-1 ranges of Table 1), build each system
explicitly, generate the noise-free spectrum at R = 1000 on 0.5–7.8 µm, store it with
the parameters. Sizes: 20,000 training, 5 × 2,000 test, H₂-dominated. Clean NaN and
depth > 1 rows, report counts. Verify parameter independence (|r| < 0.05 for every
pair) and write the correlation table. Bin to the three configurations with
`v2/bin_spectra.py`. If decision 5: the N₂ grid, 5,000 training and 2 × 1,000 test.

### Stage 2. Noise model (one hour)

`v2/noise.py`: for a spectrum and stellar temperature, interpolate the ExoRad NSR
shape from `final_results/ariel_nsr_curves.npz` onto the bin grid, scale so the median
per-bin SNR equals the requested level (primary 15; sweep 10, 7, 5), draw Gaussian
noise. Training draws a fresh realization per epoch or per fit; evaluation uses a
fixed seed per test set.

### Stage 3. In-domain study (one day)

Re-run the tuning, five-set evaluation, calibration, McNemar, bootstrap, threshold
sensitivity, and margin analyses at the Ariel configuration, for four feature sets ×
three models. Then the ladder: the same pipeline retrained at R = 100 and R = 200 on the
same planets. Adapt the existing scripts; do not copy them wholesale, because the
feature pipeline changes.

### Stage 4. Retrievals: not in this paper

Paper 2. See `../jhss_paper2/OUTLINE.md`. Its runs can start as soon as Stage 3 has
frozen the classifier, and they do not block anything below.

### Stage 5. Shift re-renders and evaluation (one day of compute)

For the 10,000 test planets:

| Axis | Generator | Notes |
| :-- | :-- | :-- |
| Independent RT code | `exotransmit_harness.py` adapted | same planets, opacities held fixed |
| Alternative opacities | `generate_opacity_swap_testset.py` adapted | ExoMol for three non-label gases; then + HITRAN O₃ |
| Clouds and hazes | `generate_aerosol_paired.py` adapted | five deck pressures, five haze densities |
| Stellar contamination | `evaluate_spots_phoenix.py` adapted | TLSE with PHOENIX, spot coverage 2–20 % |
| Resolution / SNR sweep | `noise.py` levels; ladder | correlated noise family retained |
| Instrument noise colouring | white vs ExoRad shape at matched σ | now a null-check, since the primary noise is already ExoRad-shaped |
| Extrapolation | radius split | retrain by construction |

Each axis reports Δ accuracy and Δ Brier, paired at the planet level, with the
predicted-positive rate for the opacity axis and the surviving feature amplitude for the
aerosol and stellar axes, so each mechanism claim is re-measured on the new grid.

### Stage 6. Figures and tables

`v2/plots/`: `fig1_workflow.py`, `fig2_ladder_calibration.py`, `fig3_margin_amplitude.py`,
`fig4_fidelity_sweeps.py`, `fig5_threshold_transfer.py`; `tables.py` writes the three
tables as Markdown blocks for the manuscript source. Every figure 6.0 in wide, h/w
0.45–0.6, 300 dpi, no in-image titles, no names or paths.

---

## 7. Numbers ledger

Every number in `manuscript.md` traces to a file under `v2/results/`, all produced on
2026-09-09 from the regenerated grid:

| Quantity | File |
| :-- | :-- |
| Grid sizes, acceptance by mass decile, parameter independence | `generation.txt`, `independence.txt` |
| Table 2 (model × preprocessing), McNemar, bootstrap | `ariel_indomain.txt`, `ariel_summary.json` |
| Resolution ladder | `r100_indomain.txt`, `r200_indomain.txt`, `tables.md` |
| Absolute-noise sensitivity run | `ariel_abs50_indomain.txt`, `ariel_abs50_labels.txt`, `ariel_abs50_calibration.txt` |
| Threshold sensitivity, margin, amplitude | `ariel_labels.txt`, `ariel_labels.parquet` |
| Table 3 and Figure 4 (shift map, all sweeps, extrapolation) | `ariel_shifts.txt`, `ariel_shifts.csv` |
| Calibration, reliability curves, operating thresholds | `ariel_calibration.txt`, `ariel_reliability.csv`, `ariel_pr.parquet` |
| Figures 1–5 | `figures/fig1_workflow.png` … `fig5_threshold_transfer.png` |

`nhsjs/fill_numbers.py` fills the grid- and tuning-related placeholders; the Results
and Discussion numbers were transcribed from the files above and should be re-checked
against them after any recompute.

---

## 8. Build and submission checklist

1. Source: `nhsjs/manuscript.md` (new; replaces `NHSJS_standard.md` as the build input). Figures come from `v2/plots/`.
   Citations in the source as `[n]` in first-mention order; the builder converts them.
2. `build_docx.py`: keep the template, superscript-citation, and reference-format code;
   change `TEMPLATE` to `templates/NHSJS-Manuscript-Template-Standard-Citations.docx`
   (and the Online template for the second file), change the input path and figure list,
   use the template's own styles (`NHSJS Title`, `NHSJS Section`, `NHSJS Subsection`,
   `Normal`), write captions in the `Figure N | …` bold form, and add a `--online` mode
   that emits the double-parentheses form with a space before `((` and superscript commas
   between adjacent citations (the template's Find-and-Replace procedure, done in code).
3. Outputs: `<exact title>.docx` (blind, standard) and `<exact title> - online.docx`
   (author block in the template's format, acknowledgments, online citations).
   Supplementary: a zip of the figure PNGs in the form's third slot.
4. Open both in Word and confirm: page count ≤ 20 (target ≤ 16), 12 pt Times New Roman,
   single spacing, letter size, superscripts before punctuation, template instruction
   pages removed, no author name or "oy2017" in the blind file, file name matches the title
   character for character.
5. Form fields: title; type = Research Article; abstract pasted; number of citations;
   prior-submission statement (JHSS rejection; ML4PS non-archival workshop).
6. Commit on `main` without the Co-Authored-By trailer (repo rule).

---

---

## 9. Work order and schedule

| Step | What | Owner | Wall-clock |
| :-- | :-- | :-- | :-- |
| 1 | Confirm decisions 1, 4, 5 | author | — |
| 2 | Stage 0 prerequisites | Claude | half a day |
| 3 | Stages 1–2: grid and noise model, independence check committed | Claude | one day |
| 4 | Stage 3 in-domain study and ladder; first ledger entries | Claude | one day |
| 5 | Stage 5 shift re-renders and evaluations | Claude | one day |
| 6 | Stage 6 figures and tables | Claude | one day |
| 7 | Draft `nhsjs/manuscript.md` from Section 5 | Claude | one day |
| 8 | Author review and edit list; apply; number and page check | author, Claude | two days |
| 9 | Build both DOCX files; page count confirmed in Word; submit | Claude, author | one day |
| — | Paper 2 retrievals start in the background after step 4 | Claude | weeks, off the critical path |

Critical path: about seven days. Steps 2–4 need nothing from the author beyond step 1.
