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
