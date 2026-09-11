# Pre-registered predictions for the ExoSim2-derived axes

Committed 2026-09-11, **before** any ExoSim2-derived perturbation has been scored
against any classifier. The point of this file is its commit timestamp: the
predictions below are made from the repair rule alone (RESEARCH_PLAN.md §4.2),
by counting draws, without looking at the outcome. Changing them after the runs
would defeat the purpose; if a prediction is wrong, the wrong prediction stays
here and the paper reports it.

## The rule being tested

What augmentation can recover is set by how many independent values the shift
draws per spectrum. One-draw shifts (a re-rendered physics, a single tilt with a
sign) recovered 80–89 % of the loss on the v2 grid; per-bin redraws (white and
correlated noise, 102 draws) recovered 29–31 %.

## Axis: ExoSim2 radiometric noise (adopted)

Construction: per planet, ExoSim2 `focalplane` + `radiometric` with the planet's
own transmission spectrum and host star; σ(λ) per Ariel bin from its noise
budget (source photon + foreground photon + dark + read); one Gaussian draw per
bin per planet at that σ.

Draw count per spectrum: **102** (one per bin).

**Prediction: unrepairable. Augmentation recovers roughly one third of the
loss (20–40 %), in the same class as the v2 white- and correlated-noise axes.**

Secondary prediction: because the ExoSim2 σ(λ) is wavelength-structured
(photometer bins vs spectrometer bins, QE and transmission curves), the *cost*
will differ from the ExoRad-shaped noise of v2, but the *repairability* will
not. If augmentation recovers > 60 % here, the rule is wrong and this file says
so.

## Axis: ExoSim2 time-domain systematics (bounded pilot, ≤ 20 planets, if run)

Construction: full sub-exposure + NDR simulation with pointing jitter, then a
bespoke extraction to a per-bin transit depth. Only if run; see plan §7 item 5.

- Pointing jitter: one displacement time series per observation, shared by all
  bins → **few draws per spectrum → predicted repairable (> 60 %)**.
- Read noise and dark current in the frames: per-pixel, per-read → **many draws
  → predicted unrepairable (< 40 %)**.
- A flat-field / QE-map error, if injected: **one fixed map per instrument,
  zero draws per spectrum** → predicted **most repairable of all**. This is the
  prediction where the rule and intuition disagree ("detector systematics are
  bad") and is the one worth running first if the pilot happens.

## What would falsify the rule

Any ExoSim2-derived axis whose measured recovered fraction lands in the band
the rule forbids for its draw count: a per-bin-redraw axis recovering > 60 %, or
a one-draw axis recovering < 40 %. Either outcome is reported as-is.

## Axis 8: quenched composition (added 2026-09-11, before any augmentation on it)

Construction: the same test planets re-rendered with their carbon/oxygen partitioning
frozen at the quench level of an anchored Guillot profile with a convective adiabat
(`v3/generate_grid.py --mode quenched`; RESEARCH_PLAN.md section 3, Axis 8). Every planet's
shifted spectrum is a deterministic function of its parameters: no random draw enters.

Draw count per spectrum: **0** (a deterministic re-render, like stellar contamination
and haze in v2).

**Prediction: repairable. Augmenting the training set with quenched spectra recovers
more than 60 % of the loss — in the same class as the v2 physics axes (79-89 %).**

Why this is a real test: the rule was established on perturbations that change the
spectrum's *shape* through added or scaled features. Quenching changes *which
molecules are present* — it moves carbon between CH4 and CO — so the shifted spectra
sit in a different chemical regime, not a distorted version of the same one. If the
draw-count rule is about the perturbation's degrees of freedom and not about the kind
of physics, it should still hold here. If augmentation recovers < 40 %, the rule is
wrong for composition shifts and this file says so.

Secondary prediction, from the chemistry: the cost of this axis before repair will be
concentrated on planets cooler than ~1000 K, where quenching moves CH4 and CO by dex,
and near zero above 1500 K, where nothing changes.
