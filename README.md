# Machine-learning triage of synthetic exoplanet transmission spectra

Can a classifier trained on simulated Ariel spectra rank CH₄–O₃ candidates, and which
of the simulator's choices does that reliability depend on? This repository holds two
studies of that question.

## Start here

| If you are | Go to |
| :-- | :-- |
| **Reviewing the current paper** | **[`v2/REPRODUCE.md`](v2/REPRODUCE.md)** — every claim mapped to the command that produces it and the committed file it is read from |
| Reading the manuscript source | [`nhsjs/manuscript.md`](nhsjs/manuscript.md) |
| Looking for the code | [`v2/`](v2/) — the whole current study, ~15 scripts |
| Looking for the earlier study | [`study1_R200/`](study1_R200/) — superseded, kept for provenance |

## Layout

```
v2/            Current study: grid generation, three observing configurations,
               classifiers, the domain-shift map, figures. Start at v2/REPRODUCE.md.
nhsjs/         Submission package for the current paper (manuscript source, the two
               DOCX builds, the journal's templates, the rewrite plan).
jhss_paper2/   Outline of the companion retrieval study, not yet run.
ml4ps2026/     Four-page workshop version of the earlier study (non-archival).
study1_R200/   The earlier study at a uniform resolving power of 200: its scripts,
               spectra, results and manuscript. Superseded; see below.
ariel_noise_model/  ExoRad 2 payload and the noise-to-signal curves both studies use.
reference papers/   Third-party PDFs cited by the manuscripts.
```

## The two studies

**Current (`v2/`).** Spectra are simulated once at native resolution without noise and
then binned to the layout Ariel actually delivers for Tier 3 — three photometric bands
below 1.1 µm, then R = 15, 100 and 30 across the infrared — with noise shaped by a
radiometric model of the payload. Every parameter is drawn independently. Three
classifier families are compared on four feature sets, and the best pipeline is frozen
and re-scored on the same test planets re-rendered under seven controlled changes to
the physics. The headline: per-spectrum normalization is worth 6 to 18 accuracy points
and matters more than the choice of model, the delivered resolution costs under two
points relative to R = 200, and the classifier is robust to the radiative-transfer
implementation while fragile to anything that changes spectral shape.

**Earlier (`study1_R200/`).** A uniform R = 200 grid with parameters drawn through the
simulator's sampler, which delivered several of them nearly collinear, and features from
PCA on raw transit depths. Rejected by the Journal of High School Science in September
2026 for insufficient novelty; the current study is the rebuild its referee described.
Several of its conclusions do not survive the rebuild — notably the cost of swapping
non-label opacity tables (16 points there, under 2 here) and the claim that tree
ensembles beat neural networks. **Do not quote its numbers for the current paper.**

## Environment

Python 3.10. `pip install -r requirements.txt` installs the pinned stack (MultiREx over
TauREx 3, scikit-learn, XGBoost, TensorFlow). Set `OMP_NUM_THREADS=8` before training:
XGBoost with 20 threads on a loaded machine ran about 100× slower in our tests.

Regenerating shifted spectra additionally needs Exo-Transmit, the ExoMolOP tables, a
converted HITRAN ozone table, a Mie-capable MultiREx fork and the BT-Settl/PHOENIX
stellar atlas; each script names its own requirement. None of them is needed to check a
number, because every result file is committed.
