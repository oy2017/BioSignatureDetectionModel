# When can a simulator-trained screen be trusted?

Ariel will deliver transmission spectra of roughly a thousand exoplanets. Machine-learning
screens trained on simulated spectra have been proposed to triage them — the consortium's
own Tier-1 strategy trains classifiers on simulated spectra to decide which planets deserve
deeper observation (Mugnai et al. 2021). Every such screen is validated the same way: on
held-out spectra from the simulator that trained it. There will be no Ariel data to validate
it on before the decisions are made.

This repository asks the question a mission actually faces: **on data whose physics differs
from the simulator — in ways we anticipate and in ways we deliberately hold out — which
planets can a screen still classify, does it know when it can't, and what fixes what?**

## The answer, in the form the paper gives it

A reliability envelope: one table, one row per mismatch, saying for each

| column | meaning |
|---|---|
| **loss** | accuracy the mismatch removes from the screen, on the same planets |
| **irreducible** | the part no retraining recovers (ceiling from a model trained at the test condition) |
| **absorb** | what a training grid that randomizes the uncertain ingredients buys back |
| **detect** | whether a decline rule ranks the screen's errors — by confidence, or by distance from the training set |
| **fix** | what removes the loss: training on it, a better observation, or a change to the forward model |

evaluated at the binning where the triage decision is made (Tier 1, seven points), on the
mission's actual target list, with thresholds fixed on clean data and nothing tuned on the
shifted sets. The map has three regions, and they are the result:

1. **Mismatch you modelled** (clouds, haze, stellar contamination, noise level, another
   radiative-transfer code, other opacity tables): randomized training plus a
   confidence-based decline rule recovers most of it; what remains is information the
   mismatch destroyed, and only a better observation — or training on the alternative code
   or tables — gets it back.
2. **Mismatch that helps**: quenched (disequilibrium) chemistry makes the carbon-rich label
   *easier*, at every eddy-diffusion strength tried. Train on equilibrium.
3. **Physics the simulator omitted**: nothing done to the training set helps, the screen is
   confidently wrong, only a distance-based novelty alarm sees it, and only fixing the
   forward model repairs it. The worked case is two molecules carbon-rich atmospheres are
   known to carry (HCN, C₂H₂) that the field's standard Ariel training grid omits — and that
   put a carbon-rich screen at chance on the planets it exists to find.

Numbers for every row are in [`v3/TRUST_IDEA.md`](v3/TRUST_IDEA.md) (consolidated table
§4e). **They are being regenerated** — see *Status* — and should not be quoted until the
re-run finishes.

## Two screens under test

The procedure is the product; the screens are the worked examples.

- **The consortium's Tier-1 molecular screen** (Mugnai et al. 2021, AJ 162, 288): flags
  H₂O, CH₄, CO₂ or NH₃ above an abundance threshold from seven Tier-1 points, with four
  default scikit-learn classifiers. Alfnoor is not public, so it is rebuilt from the paper's
  text in [`v3/alfnoor_screen.py`](v3/alfnoor_screen.py); every deviation is listed in its
  output.
- **A carbon-rich screen** (this repository): is C/O > 1? — a regime, not a molecule, and a
  stated Ariel science target. FastChem equilibrium chemistry, the Ariel layout, noise shaped
  by the consortium's simulator, tuned XGBoost on per-spectrum-normalized bins.

One is simple and theirs; one is tuned and ours. Where both break in the same places the
result is about the setup, not the model; where they differ, the difference is a finding.

## The procedure (reusable on any screen)

```
render the test planets under each mismatch      shift_*.py, compound.py
bound what any retraining could recover           oracle.py        (train at the test condition)
randomize the uncertain ingredients               randomize_train.py, trust_randomized.py
hold each axis out in turn                        trust_randomized.py
decline rules: confidence vs distance             trust_detect.py, trust_envelope.py
at the tier where the decision is made            tier_screen.py
on the mission's real target list                 mcs_testset.py, mcs_eval.py
per host type; per chemistry assumption           host_dependence.py, kzz_eval.py
```

Two habits run through all of it. Expectations are written down and committed before a run
([`v3/PREREGISTERED_PREDICTIONS.md`](v3/PREREGISTERED_PREDICTIONS.md), TRUST_IDEA §4f), and
misses are reported as misses — several of ours are. And every relative number names its
reference: recovery against the oracle, not against clean accuracy; selective accuracy
against the clean selective baseline at the same coverage. The first habit found an
earlier "repair rule" of ours to be an artefact of the wrong denominator; it is retired in
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) §4.2 and the retraction is kept on the record.

## Status (2026-09-12)

A full regeneration is running (`v3/rerun_all.sh`, ~10 h). Rebuilding the consortium's
screen exposed that our own forward model carried NH₃ through mean molecular weight only —
a documented simplification inherited from the earlier study, but indefensible in a paper
whose headline is an omitted species. The NH₃ opacity table is now in, and every v3 number
is being recomputed. The previous run is archived in `v3/results_noNH3/`. The qualitative
map is expected to hold; cool-planet numbers are expected to move.

## What is and is not new

No method here is new: domain randomization, selective prediction, distance-based
out-of-distribution scores, target-trained oracle bounds and held-out-axis evaluation are
all standard, and the paper cites their origins. What is new is the question asked of an
exoplanet screen at all, the physics-side mismatch axes (others shift instrument noise), the
absorb-or-detect-or-fix framing with held-out axes, the envelope as a mission deliverable,
and the finding about the field's standard training grid. Closest prior work: Ardévol
Martínez et al. 2022 (three mismatches priced on a retrieval CNN, no remedy or detection),
the Ariel Data Challenge datasets (instrument-shift only), and misspecification detection
for simulation-based inference in cosmology (2025). Details: TRUST_IDEA §1a–2b.

## Layout

```
v3/              The trust programme. Start at v3/TRUST_IDEA.md, then rerun_all.sh for the
                 order everything runs in. results/ holds every number the paper quotes.
v2/              The preceding study (fixed-threshold CH4/O3 label, seven-axis budget).
                 Superseded as a paper; its scripts are the ancestors of v3's.
study1_R200/     The first study, at uniform R = 200. Superseded; kept for provenance.
nhsjs/           Submission package of the v2 paper. Not under active work.
ariel_noise_model/  ExoRad payload and noise curves; v3 adds ExoSim2 curves in v3/results.
reference papers/   Third-party PDFs cited.
RESEARCH_PLAN.md Program-level plan, decisions and their dates, prior-art checks.
```

## Environment

Python 3.10; `pip install -r requirements.txt`. The forward model is MultiREx 0.3.1 over
TauREx 3, as a local fork whose changes are documented and justified in
[`v3/MULTIREX_FORK.md`](v3/MULTIREX_FORK.md) (a real seeding bug in upstream, plus the
aerosol feature). Opacity tables for CO, NH₃, HCN and C₂H₂ are added from Exo-Transmit.
Re-rendering the mismatch axes additionally needs Exo-Transmit, the ExoMolOP tables, the
PHOENIX atlas, FastChem and an ExoSim2 environment; each script names its requirement.
Set `OMP_NUM_THREADS=8` before training.

Results files are committed, so every number can be checked without regenerating spectra.
