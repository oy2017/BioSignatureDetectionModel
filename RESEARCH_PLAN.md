# Why this work exists, and what it has to deliver

Written 2026-09-11. Supersedes the framing in `nhsjs/REWRITE_PLAN.md` and
`jhss_paper2/OUTLINE.md` where they disagree; those remain accurate about the
mechanics of paper 1 and the retrieval study.

---

## 1. The bind that makes this necessary

Ariel launches in 2029 and will survey roughly a thousand planets, of which only
50 to 100 reach Tier 3, the deep characterisation. Something has to rank the
rest, and that ranking is a genuine mission bottleneck. Machine-learning screens
trained on synthetic spectra are the proposed instrument for it.

This creates a structural problem that is nobody's fault and cannot be dodged:

> **The screen must be built entirely inside a simulator and then deployed on
> real photons, and the gap between the two cannot be measured before launch.**

There are no Ariel observations to validate against and will not be for years.
Waiting is not an option either, because the screen has to exist before the data
does. Any group building one is in this position.

## 2. The move: sensitivity analysis is what you have when validation is impossible

You cannot measure the distance to the truth. You *can* measure how much the
answer moves when you change the things you were free to choose.

If a screen's accuracy barely shifts under every plausible variation of the
simulator's ingredients, that is evidence it learned something about spectra. If
it swings thirty points when one haze prescription is swapped, the headline
accuracy was a property of the simulator, not of the physics.

The same logic is standard in climate modelling, where validation against the
future is impossible and perturbed-physics ensembles are used instead.

**The design constraint that keeps this from being a closed loop:** the
alternatives must come from *other groups' independently developed code and
data*. Exo-Transmit for radiative transfer, ExoMolOP and HITRAN for opacities,
Lee's Mie haze, Rackham's transit light source effect, BT-Settl stellar spectra.
You must not choose both sides of any comparison. Where we do choose both sides
— the invented instrument systematics — the result is correspondingly weaker,
and §5 addresses that.

## 3. Making the screen a real screen

The v2 study's screen is a benchmark construct, not something anyone would
deploy. Three reasons, and all three are fixable:

| problem | why it is a problem |
| :-- | :-- |
| Label is a threshold on injected abundances | No real pipeline measures abundances and applies a cutoff. Ariel measures a spectrum; abundances come from retrieval with uncertainties. |
| Abundances are independent log-uniform draws | Gives 10 % ozone by volume in a hydrogen atmosphere — impossible on both oxidation state and quantity. Median positive-class O₃ is 100 ppm against Earth's ~0.3 ppm column mean. |
| The task is not the mission's task | Ariel's triage decision is which planets merit deep characterisation; its primary science is C/O ratio and metallicity. |

### The fix

**Composition from equilibrium chemistry.** FastChem (pip-installable,
milliseconds per atmosphere) computes abundances from temperature, pressure,
C/O ratio and metallicity. The recorded parameter draws are reusable, so the
grid design, test splits and independence properties carry over.

**Label = C/O ratio class.** Two-way, with the cut at one of two physically
meaningful values — solar (≈ 0.55) or the carbon-rich boundary (≈ 1) — chosen at
the feasibility step (below), not at an arbitrary abundance.

Why this and not the two alternatives that were considered and rejected:

- **Abundance thresholds (the v2 label)** — a cutoff nobody computes, on a
  chemistry that cannot exist.
- **H₂O detectability** — proposed on 2026-09-11 and **killed the same day by a
  check on the existing grid**: feature amplitude is 83 % explained by bulk
  parameters (radius, mass, temperature, stellar radius) and 1 % by log H₂O.
  A detectability label would have been "is this a puffy hot planet around a
  small star", which the target catalogue already answers without a spectrum.
  A second defect: under the paper's primary peak-to-peak noise convention
  (σ = peak-to-peak / 15) every planet has the same effective SNR by
  construction, so the label would have been nearly constant. **Do not revive
  this label.**

C/O passes the four tests the detectability label failed:

| requirement | C/O |
| :-- | :-- |
| a question a real pipeline answers | it is Ariel's primary science objective |
| composition, not bulk | drives the CH₄/CO/H₂O/CO₂ partitioning — a *shape* signature, which is what survives per-spectrum normalization; amplitude is bulk-driven and normalization discards it, which is exactly why normalization won by 6–18 points |
| invariant under every shift axis | it is an input to the chemistry, not a property of the observed spectrum |
| chemically consistent | by construction under FastChem |

**Cut point: C/O > 1.0 (carbon-rich), decided by the feasibility gate on
2026-09-11** (`v3/feasibility.py`, 596 planets, Ariel layout, peak-to-peak SNR 15,
C/O ~ U(0.2, 1.5) and [M/H] ~ U(−1, 1.5) sampled independently):

| cut | positive rate | bulk-only (no spectrum) | spectrum | margin |
| :-- | --: | --: | --: | --: |
| solar, 0.55 | 0.72 | 65.8 % | 85.7 % | +20.0 |
| **carbon-rich, 1.0** | 0.38 | 53.4 % | **92.3 %** | **+38.9** |

Per temperature band at the carbon-rich cut, spectrum accuracy against majority:
500–1000 K **80.5 vs 65.7**; 1000–1500 K 95.6 vs 62.7; 1500–2500 K 97.8 vs 59.5.
The label is learnable in every band and weakest in the coolest, exactly where
the chemistry table predicted (at 600 K, C/O barely moves H₂O; the signature is
carried by CH₄ alone). That degradation is real, quantified, and to be reported —
not a reason to drop the cool third.

Two honest notes on the test. Row "bulk-only" is a *leakage* check, not the
confound check: C/O and temperature are sampled independently, so temperature
*cannot* predict the label by construction (the bulk-only score equals the
majority rate, as it must). The confound check is the per-band row — does the
spectrum still separate C/O *within* a temperature band, where temperature is
held roughly fixed — and it passes. Second, the 0.38 positive rate is a sampling
artefact of the U(0.2, 1.5) range; for the grid, sample C/O so the cut lands at
the median (e.g. U(0.2, 1.8)). A population half carbon-rich is not realistic and
does not need to be — this is a stress-test grid, as §6 already says.

Metallicity must be sampled independently of C/O, or the classifier learns
metallicity as a proxy.

**Two design constraints the C/O label imposes, both mandatory:**

1. **CO must carry opacity.** The v2 grid gives opacity to H₂O, CH₄, CO₂ and O₃
   only; CO acts through mean molecular weight alone. At high temperature carbon
   sits in CO, so without CO opacity a C/O label measures an invisible molecule.
   ExoMolOP has CO. This is a configuration change for the regeneration, but it
   is not optional.
2. **The temperature confound must be checked, not assumed away.** Under
   equilibrium the CH₄/CO transition is driven by temperature as much as by C/O,
   so the spectral signature of C/O changes character across 500–2500 K. The risk
   is a classifier that learns temperature from the continuum and reports it as
   C/O — the same failure mode as the detectability label, one level up. Test:
   regress the classifier's score on temperature within each C/O class before
   trusting anything downstream. If temperature dominates, stratify the label by
   temperature band or narrow the grid.

The design-trap rule that killed the detectability label is general and worth
keeping in view for any future label:

> **The label must be invariant under every shift axis.** If the label depends on
> the observed spectrum, adding haze changes the answer key as well as the exam,
> and "accuracy under haze" conflates label drift with classifier degradation.

**The central finding does not depend on the label.** The repair rule is a claim
about perturbations, not about what is being classified, so relabelling should
leave it intact. That makes the regeneration a free fourth test of the rule: if
draw count still separates repairable from unrepairable under a completely
different label, the rule has survived a change of task as well as three rival
explanations.

### Axis 8: price the chemistry idealisation instead of apologising for it

Equilibrium chemistry is itself a simulator choice, so it belongs in the budget
rather than in the caveats. **Add equilibrium versus quenched composition as a
shift axis.**

Quenching is the cheap and dominant disequilibrium effect: compute equilibrium,
then freeze abundances above the level where the chemical timescale exceeds the
mixing timescale. It is a standard approximation in retrieval codes, not a
research project. Full photochemistry (VULCAN) on a subset can follow.

Severity is temperature-dependent and mildest for the species carrying the label.
Above ~1200 K reaction rates keep equilibrium a fair description at observable
pressures; below ~1000 K quenching freezes the CH₄/CO ratio and photochemistry
starts producing HCN, C₂H₂ and hazes. The grid spans 500–2500 K, so the cool
third is where it bites. Carbon-bearing species are the exposed ones — CH₄ is the
textbook casualty — while **water is the dominant oxygen reservoir across most of
the range and comparatively robust**, its abundance set mainly by metallicity and
C/O. Quenching still redistributes carbon and oxygen between CO, CH₄ and H₂O.

For a **C/O label this cuts the other way**: the carbon-bearing species that
carry the C/O signature are exactly the ones quenching disturbs. So Axis 8 is not
a courtesy measurement — it is the axis most likely to cost the most, and the
one a referee will ask about first. Its result decides whether the label is
usable across the cool third of the grid or only above ~1200 K.

If the measured cost is small, the objection is retired with a number instead of
a paragraph. That is strictly better than disclosing it.

## 4. What the contributions are

**The product is a criterion and a procedure. The numbers are the worked
example, not a reference table.** In descending order of how far each travels:

| | what it is | how far it transfers |
| :-- | :-- | :-- |
| **1. A criterion** | The repair rule: what augmentation can recover is set by how many values the shift draws per spectrum. | Everywhere. It is not a number. It also contradicts the standard ML account (§4.2). |
| **2. A procedure** | Freeze the screen, re-render the same planets with one ingredient changed, rank the costs. Runs at ~0.4 s/planet. | Everywhere. Any group gets their own ranking in an afternoon. |
| **3. A worked example** | The Ariel budget itself. | The *ordering* transfers (ρ = 0.944 across pipelines, 0.973 across spectral configurations). The *magnitudes* do not. |

What a reader buys with it: two decisions that otherwise get made by guesswork —
where to spend realism effort, and whether to model something or train against
it — each worth months of work.

Table 3 of the paper should be presented as a demonstration that the procedure
yields an interpretable ranking, **not** as a lookup table other groups can cite
numbers from. Presenting it as the latter invites the obvious objection and
deserves it.

### 4.1 The audit method (useful, not surprising)

Freeze one screen, re-render the same planets with one simulator ingredient
changed at a time, rank the costs. It runs at ~0.4 s per planet. It answers
three questions a team otherwise settles by guesswork: where to spend realism
effort, whether to model something or train against it, and how much of a
published accuracy is simulator-dependent.

Findings of this kind — implementation choices are nearly free until they touch
the label-bearing molecule, anything that recolours the spectrum is expensive —
are **confirmatory**. Any spectroscopist would have bet that way. They are worth
having as a budget, not as a discovery.

### 4.2 The repair rule (the finding that resolves a real question)

What augmentation can recover is set by **how many values the shift draws per
spectrum**, not by whether it came out of the forward model.

| axis | power above ⅛ cycle/bin | draws per spectrum | gap recovered |
| :-- | --: | --: | --: |
| stellar spots, 20 % | 27 % | 1 | 80 % |
| haze, 3 × 10⁷ m⁻³ | 11 % | 1 | 89 % |
| gain ramp, 2 × noise | 5 % | 1 | 84 % |
| correlated noise | 2 % | 102 | 31 % |
| white noise | 76 % | 102 | 29 % |

This matters because it **contradicts the standard ML account**. The Fourier
perspective on robustness (Yin et al., NeurIPS 2019) predicts that
high-frequency corruptions are the augmentable ones. On physically motivated
spectral corruptions it gets the sign wrong: Spearman(high-frequency power,
repairability) = **−0.30**. Draw count separates the groups completely,
80–89 % against 29–31 %, no overlap, ρ = **−0.87**.

A competent ML researcher would have bet on frequency. That is the test of a
finding worth having, and it is a result about **machine learning**, established
in a domain where the corruptions are haze and starspots rather than synthetic
blur.

Two rival explanations were eliminated by designed experiments rather than
argument:

- **Determinism vs provenance.** The four original axes confound
  deterministic/stochastic with physics/instrument. The gain ramp breaks it: it
  is injected exactly like the noise but its whole effect is one tilt and a sign.
  It repairs at 84 %, landing with the physics.
- **The Fourier account**, as above, measured rather than dismissed.
- **Invertibility.** The obvious refinement — "draw count is just a proxy for
  whether the perturbation can be undone from the data" — was tested in
  `v2/invertibility.py` and fails, in the wrong direction. A ridge map recovers
  the clean spectrum *better* from correlated-noise spectra (R² = 0.899) than from
  hazed ones (0.734), yet augmentation repairs haze and not noise.
  Spearman(invertibility, repairability) = −0.20. **This result is computed,
  committed, and absent from the manuscript.** It belongs in the paper as the
  third eliminated rival.

**Known weakness: n = 5 axes.** The separation is perfect and the mechanism is
tested, but five points is five points. §5 is how that gets fixed.

### 4.3 Transferability (useful, mildly surprising)

The ordering of costs is carried by the **feature representation**, not the model
family: pipelines sharing a representation agree at ρ = 0.944, pipelines sharing
only a model family agree at 0.846 — the same figure, to three decimals, as
pipelines sharing nothing at all. The ordering also holds across spectral
configurations (ρ = 0.973 between the delivered Ariel layout and uniform R = 200).
Magnitudes do not transfer, differing severalfold between pipelines.

Practical consequence: a published fidelity budget is reusable as a ranking of
what to model carefully, never as a set of numbers. The magnitude spread between
pipelines is a median 8.1×, up to 32×, and still 4.9× between the two strongest
pipelines — so "run the audit on your own pipeline" is the honest advice, and the
procedure is cheap enough to make that reasonable.

### 4.4 Who uses this, and for what

Concrete users and the decision each one makes with it. If a claimed use does
not name a decision, it is not a use.

**Inside the mission.**

| user | decision | what they take from the work |
| :-- | :-- | :-- |
| A group building an Ariel triage screen | Where to spend realism effort | The ranking: aerosols and stellar contamination first; RT code and non-label opacities last. |
| The same group | Model it, or train against it? | The repair rule: count draws per spectrum. One-draw shifts are learnable; per-bin redraws need a better observation. |
| The same group, with a different pipeline | Can I reuse the published numbers? | No — but the procedure runs in an afternoon. Run it. |
| Ariel data-challenge participants | Which corruptions to augment against | The rule, plus the measured evidence that the Fourier heuristic misleads here. |
| A referee or mission planner | Whether to trust a screen with observing time | A checklist of what the screen has and has not been audited against, and the lower-bound caveat (§6). |

**Outside the mission.** The repair rule is a claim about machine learning, not
about Ariel. It says the standard Fourier account of augmentation robustness
makes the wrong prediction when corruptions are generated by physics rather than
drawn from a corruption library, and it names what predicts better. That
transfers to any sim-to-real problem with physically generated shifts —
remote sensing, medical-imaging simulators, robotics — and it is the reason the
ML4PS thread exists separately from the astronomy thread.

**What is not a use.** "Understanding simulator dependence" in the abstract is
not a use. Neither is "informing future work". Every sentence in a Discussion
that claims utility should be traceable to a row in the table above.

## 5. The weakest link, and the experiment that fixes it

Everything above measures sensitivity to disagreements **the field has already
named**, so it is blind wherever the alternatives agree. That limit is real and
must be stated.

But one axis is weaker than the rest: the instrument systematics — gain ramp,
offsets, smoothed-Gaussian correlated noise — are toy models **we invented**.
That is the one place the circularity objection fully lands.

**ExoSim2** (pip: `exosim` 2.0.1) is the Ariel consortium's own end-to-end
instrument simulator: detector effects, pointing jitter, correlated noise in
both time and wavelength, built on the current payload design by the group
behind ExoRad 2. Pushing the same planets through it turns the weakest axis into
the strongest, and generates two results whose answers are genuinely uncertain:

1. **Does a cheap in-simulation audit predict the loss against a full instrument
   chain?** The budget says systematics cost ~8 points and correlated noise ~11.
   That prediction could be accurate or off by a factor of three. Either answer
   is publishable, and one is a warning the field needs.
2. **Does the repair rule survive out of sample?** Classify ExoSim2's systematics
   by draw count *a priori* — jitter is one draw per exposure, read noise is
   per-pixel — **commit the prediction to the repository before running it**,
   then measure. That is a pre-registered test of the rule against corruptions
   we did not design, and it takes n from 5 to something defensible.

## 6. Caveats, sorted by whether we can price them

State these on the first page, not in a late Limitations paragraph. The useful
distinction is not "limitation or not" but **whether the limitation can be turned
into a measurement**. Where it can, do that instead of disclosing it.

### Priced — converted into an axis, so the paper reports a cost

- **Equilibrium is not photochemistry.** Becomes Axis 8 (§3). Quenched versus
  equilibrium composition, measured like any other ingredient.
- **Instrument systematics were invented by us.** Becomes the ExoSim2 validation
  (§5), where the alternative is the consortium's own instrument chain.
- **Spectral configuration.** Already priced: the ordering holds at ρ = 0.973
  between the delivered layout and uniform R = 200.
- **Pipeline choice.** Already priced: ordering stable at ρ = 0.944 across six
  pipelines, magnitudes not.

### Unpriced — genuine limits, to be stated plainly and not argued around

- **No real photons.** Ariel has not launched. ExoSim2 is the realism ceiling
  available before 2029; it is a simulator, not an observation. Nothing in this
  programme establishes performance on real Ariel data, and it should never be
  written as though it does.
- **Blind where simulators agree.** The budget measures sensitivity to
  disagreements the field has already named. If TauREx and Exo-Transmit are wrong
  in the same way, the audit cannot see it. **The budget is therefore a lower
  bound on simulator dependence, not a distance from reality.** This is the same
  structural fact as the retrieval result: two models sharing an assumption
  cannot reveal that assumption.
- **Magnitudes do not transfer between pipelines.** The *list* of choices is
  general to transmission spectroscopy; the *ordering* is measured stable across
  pipelines and spectral configurations; the *magnitudes* are specific to the
  pipeline and configuration that produced them. Advice to a reader is to run the
  procedure, not to cite the table.
- **n = 5 axes for the repair rule.** Fixed only if the ExoSim2 out-of-sample
  test (§5) is run and the pre-registered prediction holds.
- **Ariel-shaped, not Ariel.** Even with the fixes, this is a study of a screen
  built the way one would be built, not of a screen anyone has deployed.

## 7. Execution order

1. **Label: C/O class (§3). Decided 2026-09-11.**
2. **FastChem feasibility — DONE 2026-09-11** (`v3/feasibility.py`,
   `v3/results/feasibility.txt`). FastChem 4.0.3 installed; CO opacity added to
   MultiREx from `~/exotransmit_src/Opac/opacCO.dat` and verified live; the
   carbon-rich cut separates at 92.3 % against a 53.4 % bulk floor and holds in
   every temperature band. Gate passed.
3. **Write the pre-registered prediction** for the ExoSim2 axes — draw-count
   classification and expected repairability — and commit it before any of it
   runs.
4. **Regenerate** the grid and re-run the analysis chain, including Axis 8
   (quenched versus equilibrium composition) so the chemistry idealisation is
   priced rather than disclosed.
5. **ExoSim2 spike.** Install, run one target end-to-end, find out what it emits
   and how long it takes. If it yields focal-plane time series needing a bespoke
   extraction pipeline, the plan changes.
6. **Run the validation** and compare against the committed predictions.

## 8. Decisions already taken, not to be re-litigated

- Paper 1 (NHSJS) carries no retrieval results; the retrieval work is paper 2.
  The retrieval section was cut from paper 1 on 2026-09-11 and the version that
  contained it is preserved at `nhsjs/manuscript.with_retrieval.md`.
- The biosignature framing goes. The task is C/O-ratio classification, and the
  paper should not cite the biosignature literature as motivation.
- The H₂O-detectability label is dead (§3) and is not to be revived. Amplitude
  is bulk-driven; the label would have measured planet size.
- CO opacity is mandatory in the regenerated grid. A C/O label without CO
  opacity is not a C/O label. (Added 2026-09-11 from Exo-Transmit's own
  `opacCO.dat`, so the primary grid stays single-source; the ExoMolOP shift axis
  still needs its CO table fetched.)
- The cut is C/O > 1.0. Decided by the feasibility numbers above, not by taste.
- The workshop branch `ml4ps-benchmark` and its anonymous mirror are frozen for
  the duration of ML4PS review. Nothing lands there.
