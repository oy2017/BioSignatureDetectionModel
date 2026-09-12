# Can we trust it? — the idea, what is new, and the stress test

Written 2026-09-11, before the randomized grid is rendered. Purpose: find the holes now,
not after the compute. Every claim below has a named control and a kill criterion. If a
result comes back without its control, it is not a result.

## 1. The idea in one paragraph

Ariel will deliver ~1000 transmission spectra; machine-learning screens trained on
simulators are proposed to triage them; there will be no Ariel data to validate those
screens on before the decisions are made. We ask whether such a screen can nevertheless
be *deployed with a stated reliability*, and we make the question operational: on spectra
whose physics differs from the training simulator — in ways we anticipate and in ways we
deliberately hold out — the planets the screen **accepts** must be classified at a stated
accuracy, and it must **decline** the rest. Two capabilities (absorb the anticipated
mismatch; detect the unanticipated) and one product (a reliability envelope: accepted-set
accuracy and coverage per mismatch, at a decline rule fixed on clean data). The worked
example is a carbon-rich (C/O > 1) screen on the Ariel layout with FastChem chemistry and
consortium-simulator noise; the eight mismatch axes are the community's known ones.

## 1b. Who this is for, what decision it informs, and why now

**Who.** Three groups, in order of how directly they can act on it.
1. *Teams building Ariel's target-ranking and triage tools* (the consortium's Tier-2 selection
   and the Ariel Data Challenge community). Their decision: whether an ML screen may enter the
   ranking at all, and if so with what decline rule and under which forward-model ingredients
   it has been shown to hold. Today that decision is made on in-simulator accuracy alone.
2. *Anyone training ML retrievals on simulator grids* (NPE/FMPE/CNN retrievals on TauREx or
   petitRADTRANS). Their posteriors are as simulator-bound as our screen, and the same
   protocol — ceilings, held-out axes, selective accuracy against the clean selective
   baseline — applies unchanged. What transfers is the protocol, not our numbers.
3. *Grid builders.* Two design rules come out with evidence: randomize the ingredients you are
   unsure of (and here is what it costs in-domain), and train on equilibrium chemistry because
   quenching makes the C/O label easier, not harder.

**What the work gives them that they do not have.** A worked, reproducible answer to "what
happens when the simulator is wrong" — which mismatches a screen can absorb, which it can
flag, which it can do neither with (the expected hole: opacity tables) — and a table format a
mission can put requirements against. The numbers are specific to our screen, label and
simulator; they are an existence proof and a template, and the paper says so.

**Why it is worth doing now.** Ariel launches in 2029 and its triage pipelines are being
chosen now; mis-ranked Tier-2/3 time is irreversible mission time. The ML-retrieval literature
is growing fast with in-simulator validation only; nobody has put physics mismatch, ceilings
and held-out axes in one place. The marginal cost to us is low: the grid, eight rendered
axes, the ExoSim2 payload and the oracle machinery exist; the remaining work is one
randomized render and the held-out runs.

**What it does not do.** It does not validate a screen on real data (none exists), it does
not make a screen robust to mismatches outside the axis set, and its specific accuracies do
not transfer to another label or simulator. It makes the question answerable and answers it
once, with controls.

## 2. What is new — ranked, with what is NOT new beside each

| # | Claim | Not new | New |
|---|---|---|---|
| 1 | A reliability envelope for a simulator-trained exoplanet screen, measured under physics mismatch with held-out axes | Selective prediction, OOD detection, held-out-corruption protocols (all standard ML); misspecification detection for SBI in cosmology (2025) | Asked of exoplanet screens at all; physics-side axes (chemistry, aerosols, opacities, contamination) rather than instrument noise; the envelope as a mission deliverable |
| 2 | Shift detection ≠ error detection (pilot result, §4) | Known in ML folklore | Measured here: distance-based OOD scores flag every shifted spectrum (AUROC 0.9–1.0) yet do not rank the screen's errors (AUROC ≈ 0.5); they would reject real data wholesale |
| 3 | Joint randomization of simulator ingredients ("domain randomization") applied to spectra; whether it absorbs compound mismatch and what it costs in-domain | Domain randomization (robotics) | First use on astronomical spectra; the compound and in-domain-cost measurements |
| 4 | Loss decomposition: total = irreducible + reducible, with the oracle as ceiling | Target-only bound (domain adaptation) | Reported per simulator ingredient for a mission layout; replaces the retired repair rule |
| 5 | Quenched chemistry makes the C/O label easier; equilibrium training is the right grid | Moses 2013 chemistry | The ML consequence and the train-on-equilibrium design rule |

No method is new. The paper must say "we apply" and cite the origins. The novelty is the
question, the axes, and the product.

## 2b. Which results are findings, and how we can know (checked 2026-09-11)

A claimed "finding" has to pass three checks before the word is used: (a) a literature
search phrased as the claim's *negation* finds nothing; (b) the result survives the
implementation choices it rests on; (c) an independent mechanism predicts it.

- **Shift-vs-error detection split — NOT a finding.** It is an established ML result:
  Jaeger et al. (ICLR 2023, "A call to reflect on evaluation practices for failure detection")
  showed that advances in OOD detection do not improve failure detection, and the FD-Shifts
  benchmark formalises it under covariate shift. Our pilot *reproduces* it on spectra, which
  is good for the protocol (it behaves as the ML literature says it should) and must be
  cited as such, not claimed.
- **Quenching helps the C/O screen — a candidate finding, not yet earned.** (a) The retrieval
  literature says the opposite-sounding thing for a different task: assuming equilibrium
  *biases retrieved C/O* (Kawashima & Min 2021; Bédard/ten-hot-Jupiter re-analysis 2025;
  parameterised quench retrievals 2026). Nothing found on a C/O *classifier* becoming more
  accurate under quench. (b) Not yet checked: the result rests on one K_zz (1e9), one
  profile construction and a 100-bar search cap. Required before the word "finding":
  a K_zz sweep (1e7–1e11) on the test splits and a comparison of our quench levels against a
  published kinetics benchmark. (c) Mechanism is independent and known: deep C/O > 1
  chemistry depletes H2O by orders of magnitude (Madhusudhan 2012; Moses 2013) and quenching
  carries it up; the label separation we measure (0.3 → 3.3 dex in H2O below 1000 K) is
  that mechanism. So (c) passes, (a) passes narrowly, (b) is pending.

Rule going forward: nothing is called a finding in the plan or a draft until (a)–(c) are
recorded next to it.

## 3. Stress test — how each claim could be wrong, and the control that guards it

**S1. Simulator-in-the-loop.** Every "mismatch" is generated by our own simulators, so the
envelope is an envelope over *our* axes; genuine unknown unknowns are not in it.
*Cannot be fixed, only bounded.* Guards: (a) leave-one-axis-out makes each axis an unknown
to the screen at least once; (b) the axis set is the community's (Barstow 2020; Ardévol
Martínez 2022; ADC 2024), not ours; (c) Test 4 on real JWST spectra, if feasible, is the
only external check and is label-free. The paper states the envelope as *conditional on
the axis set*, never as a guarantee. Kill criterion: none — this is the stated limit.

**S2. The denominator trap (the one that bit us).** "Recovered %" against the clean
accuracy invented a repair rule. Every relative claim now names its reference:
- absorption → the per-case oracle (trained at the test strength), not clean accuracy;
- selective accuracy under shift → the *clean* selective accuracy at the same coverage
  (declining low-margin planets raises accuracy on clean data too: 98.85 % at 90 %
  coverage); the credit for a detector is the difference, not the level;
- LOAO transfer → both the clean-trained screen on that axis and the oracle.

**S3. Declining everything is not trust.** A detector that flags the whole shifted set
produces high accepted accuracy at zero coverage. Guard: success is defined jointly —
accepted accuracy ≥ target *and* coverage ≥ floor — and coverage is always printed. Pilot
(§4) shows the distance scores do exactly this failure: coverage 0–30 % under shift.

**S4. Test strengths inside vs outside the training range.** The v2 noise rows were
extrapolations (trained to SNR 8, tested at 5) and read as failures of augmentation.
Guard: randomized training ranges are continuous and logged; every test case is tagged
in-range or out-of-range; the envelope table carries the tag; at least one out-of-range
case per axis is included *on purpose* to show where the envelope ends.

**S5. Randomization that trivially matches the test.** If the randomized grid's levels
equal the test levels, Test 1 only tests interpolation. Guard: continuous draws (log-uniform
haze density, cloud pressure, spot coverage; SNR uniform), discrete test levels; and the
real test of generalization is LOAO, not the in-range cases.

**S6. Label realism.** The screen is only worth trusting if the label is one Ariel wants.
C/O > 1 from equilibrium chemistry is a Tier-2 science target; the cut-sensitivity result
shows the spectrum encodes the chemistry transition at C/O ≈ 1 (accuracy collapses if the
cut is moved). Remaining idealisations: abundances constant with altitude; one quench
model (one K_zz); no photochemistry. Stated, partly priced (Axis 8), not fixed.

**S7. Noise convention.** SNR relative to each planet's peak-to-peak amplitude makes faint
planets as easy as bright ones. Guard: the absolute-ppm noise axis is in the envelope, and
the "realistic subset" (depth < 3 %, amplitude < 300 ppm) is reported separately. A referee
will still ask for the whole envelope under absolute noise; budget for one such run.

**S8. Hard-coded numbers.** Three scripts carried v2 literals into v3. Guard: every summary
sentence is computed from the CSV it describes; a grep for digit literals in the
reporting code is part of the checklist before any table is quoted.

**S9. Statistical honesty.** Differences < 0.5 point are within training-draw variance
(±0.05 on the mean, but ±0.5–1 between test sets). Guard: the five test sets give a spread
for every number quoted; no claim rests on a difference smaller than that spread. Rival
"eliminations" with n = 5 cases (Spearman −0.30) were never evidence; do not write them.

**S10. "Trust" as a word.** Referees will bristle. Use it in the question; in claims use
"reliability envelope" and define it operationally. Never write "guarantee".

**S11. The detector may only see the noise level.** Distance scores respond to added noise
above all (AUROC(shift) = 1.0 at SNR 5). If real Ariel noise differs from the simulator's,
those scores decline everything regardless of physics. Guard: detectors are evaluated for
*error* ranking, and the noise axis is in the table as the case they fail.

**S12. Could the whole thing be done better with conformal prediction?** Conformal methods
give coverage guarantees in-distribution and fail under shift unless the shift is known;
none is published for exoplanet spectra. Guard: include split-conformal on the clean
calibration set as a baseline row, so the paper shows what the standard guarantee is
worth under mismatch rather than being asked why it was omitted.

## 4. Pilot results already in hand (frozen clean-trained screen, existing renders)

Compound mismatch (`compound.py`): spots × haze × low SNR costs *less* than the sum of the
parts on every compound tried (excess −2.4 to −7.5 points); the axes mask each other on
this screen. The pre-registered expectation of super-additivity is **wrong**; noted here
before Test 1 runs. Compounding is therefore not the interesting question; the randomized
grid's in-domain cost and LOAO transfer are.

Detectors (`trust_detect.py`, decline 10 % of clean):
- Distance scores (Mahalanobis, PCA reconstruction, k-NN): detect shift (AUROC 0.74–1.00)
  but not errors (AUROC 0.33–0.74, mostly ≈ 0.45); coverage collapses to 0–65 %. They
  answer "is the simulator off?" — useful as a *global* alarm — not "which planet is
  wrong?"
- Ensemble disagreement (XGB/RF/MLP): best accepted accuracy (mean 96.9 %, worst 93.5 %)
  at mean coverage 60 % (20–78 % on severe cases). Error AUROC 0.64–0.89.
- Probability margin: coverage 66–98 %, accepted 89–99 %, error AUROC 0.74–0.94; fails to
  restore clean-level accuracy under spots (89.8 % accepted at 86 % coverage vs ≈ 99 %
  for clean at that coverage).
- Quenched spectra are invisible to every detector (AUROC(shift) 0.44–0.58) and need no
  declining (98.2 %).
Conclusion for the design: the "detect" half must be framed as *two* instruments — a
global alarm (distance scores, per-population) and a per-planet decline rule (ensemble or
margin) — and judged against the clean selective baseline (S2). The pre-registered
expectation that "distance-based scores detect re-rendered physics" was right about
detection and wrong about usefulness.

## 5. Decision gates (cheap first; each has a kill criterion)

- **G1 — detect (done).** Kill if no score ranks errors above AUROC 0.7 under re-rendered
  shifts. Passed (ensemble 0.64–0.89, margin 0.74–0.94) — with the S2/S3 caveats.
- **G2 — compound (done).** Kill the compounding claim if losses are sub-additive. Killed;
  compounds stay as test cases, not as a finding.
- **G3 — randomized grid (~1 h render + training).** Kill the recipe if clean cost > 2
  points or < 80 % of each single-axis oracle reached. Either outcome is reported.
- **G4 — LOAO (reuses G3 machinery, ~1 h per held-out axis).** No kill; the pattern of
  transfer/no-transfer *is* the envelope's content. Opacity tables predicted not to
  transfer and not to be detected — the expected hole.
- **G5 — envelope table.** Kill the paper's headline if, at a clean-fixed decline rule,
  no detector yields accepted accuracy ≥ 93 % at ≥ 70 % coverage across in-range axes.
  Then the honest headline is "not yet deployable, and here is why".
- **G6 — real spectra (stretch).** Go only if G3–G5 produce a paper on their own.

## 6. Is it solid enough? (not bulletproof; better than what is out there)

What is out there for exoplanet ML screens is in-distribution validation (almost all
papers), one three-axis cost table without remedy or detection (Ardévol Martínez 2022), and
instrument-only shift datasets (ADC). Cosmology's SBI community has the better methods for
misspecification detection, on a different problem. This programme is better than the
exoplanet state of the art because it (i) asks the deployment question directly, (ii)
prices physics mismatch with ceilings, (iii) tests absorption *and* detection with held-out
axes, and (iv) hands over a table a mission can act on. It is not bulletproof because of S1,
and S1 cannot be removed by more simulation — only narrowed by Test 4. The paper is
honest if it says so in the abstract.
