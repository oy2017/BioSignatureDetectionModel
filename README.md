# When can a simulator-trained screen be trusted?

**TL;DR.** Machine-learning screens are proposed to decide which of Ariel's thousand
exoplanets get a closer look. They are trained and validated on simulated spectra, because
no real ones exist yet — so nobody knows what they do when the real atmospheres differ from
the simulator. We measure it. Breaking the simulator one assumption at a time (clouds, haze,
starspots, noise, other codes, other opacity tables, other chemistry, missing molecules) and
testing two screens — the consortium's own design and one aimed at a real science target —
we find the answer has three parts. **Mismatch you modelled** can be largely absorbed by
randomizing it into training, and a confidence-based decline rule handles the rest; what
remains is information the mismatch destroyed, so only a better observation recovers it.
**Some mismatch helps**: disequilibrium chemistry makes the carbon-rich label easier. **Physics
the simulator omitted cannot be trusted at all**: the screen is confidently wrong, no
retraining on other ingredients helps, only a distance-based novelty alarm sees it, and only
fixing the forward model repairs it — and the field's standard Ariel training grid omits two
molecules that put a carbon-rich screen at chance on the very planets it exists to find. The
deliverable is a reliability table a mission can act on, a scripted procedure anyone can run
on their own screen, and the general lesson: *a model validated only on the world that built
it has not been validated.* Numbers are being regenerated (see Status); the qualitative map is
the result.

## Why this matters, from the beginning

**What Ariel is.** Ariel is a European Space Agency telescope, launching in 2029, whose
whole job is to look at the atmospheres of about a thousand planets around other stars. It
does this by watching a planet pass in front of its star and measuring how much starlight
the planet's atmosphere absorbs at each wavelength. The result is a *transmission spectrum*:
a short curve, a few dozen numbers, whose bumps say which molecules are present — water,
methane, carbon dioxide — and in what rough proportions.

**The problem of a thousand planets.** Ariel cannot study every planet deeply. Its survey is
organised in tiers: a quick look at all thousand (Tier 1: a handful of coarse measurements
per planet), a deeper look at a few hundred (Tier 2), and a very deep look at a few dozen
(Tier 3). Somebody has to decide, from the quick look, which planets deserve the deep one.
Done by hand that is a thousand judgement calls; done with a model that fits the physics to
each spectrum it is slow and needs expert supervision. So a natural idea has been proposed,
including by the Ariel consortium itself: train a machine-learning *screen* — a classifier —
to read the quick-look spectrum and flag the planets worth more time (Mugnai et al. 2021).

**The catch: there is nothing real to train it on.** Ariel has not flown. No spectrum of
Ariel quality exists for any of these planets. So the screen is trained on *simulated*
spectra: a computer model of an atmosphere, a computer model of the telescope, tens of
thousands of imaginary planets. And it is tested the same way — on more spectra from the
same simulator, held back from training. That test gives a reassuring number, typically
well above 90 %.

**Why that number means less than it looks.** The simulator embodies hundreds of choices:
which molecules exist in the atmosphere, which database gives their absorption, whether
there are clouds or haze, how the star's spots contaminate the signal, how noisy the
detector is. The real universe will make all of those choices differently from the
simulator. A screen that is 95 % accurate on the simulator's imaginary planets may be 95 %
accurate on real ones, or it may not, and the standard test cannot tell the difference,
because both training and testing happen inside the same set of assumptions. The
consortium's own paper is explicit that its spectra were used only as "transmission
spectral shapes to test our methods against", with no claim about the atmospheric model's
realism. That is the state of the art: validated in-simulator, nowhere else.

**Why it is worth worrying about.** A screen that quietly fails does not look like a
failure. It returns confident answers, and the mission acts on them: deep observations go to
the wrong planets, and the planets that would have been the discoveries never get a second
look. Telescope time on a space mission cannot be refunded. We also have a concrete example
that the failure is not hypothetical: carbon-rich atmospheres — one of the things Ariel is
meant to find — contain two molecules (HCN and C₂H₂) that chemistry papers have described
since 2012 and that the standard Ariel machine-learning training grid does not include. A
screen trained without them, asked to find carbon-rich planets, performs at chance on
exactly those planets, while reporting high confidence.

**What this repository does.** It asks the question the mission actually faces — *when can
such a screen be trusted, and when can it not?* — and answers it with measurements rather
than assurances. We take two screens (the consortium's published design, rebuilt from their
paper, and a second one of our own aimed at a real Ariel science target), and we break the
simulator on purpose, one assumption at a time and several at once: add clouds, add haze,
put spots on the star, change the noise, swap the radiative-transfer code, swap the opacity
database, change the chemistry, add the molecules the grid forgot. For each break we measure
what the screen loses; how much of that loss *any* retraining could recover; how much a
training set that randomizes the uncertain ingredients buys back; whether the screen can be
made to *decline* the planets it is about to get wrong; and what finally removes the loss.
We do this at the coarse Tier-1 resolution where the triage decision would really be made,
and on Ariel's actual list of target planets rather than an idealised population.

**Who can use it.** An Ariel team deciding whether a screen may enter the ranking gets a
table, one row per way the simulator can be wrong, saying what happens and what to do about
it. Anyone training a machine-learning model on a simulator grid — which is now most of the
field — gets a scripted procedure they can run on their own model. Anyone building such a
grid gets two design rules with evidence behind them. And a reader with no background gets
a worked example of a general lesson: a model validated only on the world that built it has
not been validated.

**What is and isn't claimed.** No method here is new; the tools are standard in machine
learning and are cited. What is new is the question asked of an exoplanet screen, the
physics the simulator gets wrong being the thing varied (others vary only detector noise),
the framing of each failure as absorb-or-detect-or-fix, the answer as a table a mission can
act on, and the finding about the field's standard training grid. The numbers are specific
to our two screens and our simulator; the procedure is not.

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
