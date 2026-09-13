# Paper outline — selective, one spine, everything else cut or appendixed

Working title: *When can a simulator-trained screen be trusted? A reliability map for
machine-learning triage of Ariel transmission spectra*

**Value in one sentence.** Screens proposed to triage Ariel's thousand spectra are validated only
inside the simulator that trained them; we show, for the consortium's own screen and a
science-target screen, exactly which simulator errors such a screen survives, which it can be
made to survive, which it can flag, and which — including one present in the field's standard
training grid — break it silently and are repaired only in the forward model.

**The reasoning chain the paper follows** (every section is one link; nothing else gets in):
1. Triage screens exist and are validated in-simulator → no one knows what they do off-simulator.
2. Off-simulator cannot be tested on real data → so test against deliberate, physically motivated
   breaks, with the right references (a ceiling, a clean selective baseline).
3. Two screens, one procedure → results about the setup, not a model.
4. The map has three regions → each with a rule for what to do.
5. The headline: omitted species sit on the class-defining feature → silent failure, forward-model fix.
6. Tier 1, where triage happens → most loss is irreducible; abstention does not work.
7. What a mission should therefore do.

Target length: ~7,000 words main text, 4 figures, 2 tables; everything else in appendices.

---

## 1. Introduction (≈900 words)
- Ariel, the tiers, the triage decision; the consortium's Tier-1 ML screen (Mugnai 2021) and
  the Data Challenge grids; validated in-simulator, quoted.
- Why in-simulator accuracy does not measure deployment; why no real data can fix that before
  launch; the stakes in one sentence (as reasoning, not citation).
- The question, operationally: accepted planets classified at a stated accuracy, the rest
  declined, under physics the simulator got wrong — anticipated and held out.
- Contributions, three bullets: the map (absorb / detect / fix with ceilings and held-out
  axes) for two screens; the omitted-species result and its label-specificity; the Tier-1
  result. One sentence: no method is new, and what is.

## 2. Two screens and one procedure (≈1,400 words)
- The consortium screen rebuilt: their recipe in one paragraph, deviations in one sentence,
  reproduction outcome in one sentence (NH3/CH4 near Table 6 at reduced noise; CO2 not;
  details → Appendix A).
- The carbon-rich screen: label from equilibrium chemistry (why C/O = 1 is the learnable cut,
  one sentence), grid, Ariel layout, ExoSim2-shaped noise, normalized XGBoost. One sentence on
  the fork and the added opacity tables (NH3 episode → Appendix D, one paragraph).
- The mismatch axes: one table row each, with what is varied and why it is a real concern.
- The measurements: loss; ceiling (oracle); randomized grid; held-out axes; decline rules
  (confidence vs distance) against the clean selective baseline at equal coverage; all at Tier 3
  and Tier 1; real targets (MCS). One sentence each. Pre-registration in one sentence.

## 3. The map (≈1,800 words) — Table 1 (Tier 3), Figure 1 (the three regions)
- Region 1, modelled mismatch: training reaches 95–100 % of the ceiling; the randomized grid
  reaches 31–83 % at a one-point cost (a miss against our expectation, said so); the residual is
  irreducible (noise 4–5 points at SNR 5) or needs the alternative code/tables (85–88 %
  reducible). Confidence-based declining works; credit is negative everywhere (the rule buys
  back part, never all). Contamination is the one host-dependent axis (one sentence).
- Region 2, mismatch that helps: quenching +6 on cool planets at every K_zz; train on
  equilibrium; reverse direction costs 10. Four sentences.
- Region 3 is §4.
- The trade-off: randomizing an axis blinds the alarm to it (numbers); a fragile canary is the
  design consequence. One paragraph.

## 4. The omitted species (≈1,200 words) — Figure 2 (spectra with/without HCN+C2H2 and the
   shift in predicted probability), Figure 3 (confidence vs distance on this axis)
- Chemistry: HCN and C2H2 are the carbon-rich markers (Madhusudhan 2012, Moses 2013);
  abundances here; the standard grid omits them.
- Result: 96.5 → 71.9; carbon-rich planets 97.6 → 47.6; quenched variant worse; robust across
  models, binning, test sets; null control.
- Mechanism, honestly: band overlap plus broadband reshaping of the normalized spectrum.
- Not absorbable (randomized 75.8); confidently wrong (calibration, conformal coverage 64 %
  with empty sets); distance scores catch it (AUROC 0.92, positive credit); fully repaired by
  adding the species (96.4).
- Label-specific: the molecule-presence screen loses < 1 point → the general statement.

## 5. Tier 1 (≈800 words) — Table 2 (Tier-1 envelope with ceilings), Figure 4 (Tier 3 vs Tier 1
   for the same mismatches, plus real targets)
- Frozen 88.7; randomized helps (9–14 points on spots/haze/compounds); no decline rule works
  (AUROC 0.57–0.70, credits −7 to −28); ceilings low (haze 75, tables 79, species 82): most
  of every serious loss is not in the data.
- Real targets: 57 % inside the box; 79 % at Tier 1 under the mission's noise; M-dwarf hosts 60 %
  for atmospheric-temperature reasons.
- Consequence stated plainly.

## 6. What a mission should do (≈600 words)
- Audit the species list against the class's chemistry; include the regime's markers.
- Randomize what you are unsure of; keep a separate distance-based alarm for what you did not
  model; do not expect confidence-based abstention at Tier 1.
- Treat opacity tables as the largest reducible risk and noise as the largest irreducible one.
- Run this procedure (scripts) on any screen before it enters a ranking; report against
  ceilings, at the decision tier, on the real target list.

## 7. Limitations and what was not found (≈400 words)
- Simulator-in-the-loop: the map is conditional on the axis set; LOAO narrows, real data would
  test. Constant-with-altitude abundances, one quench scheme, no photochemistry, one spot
  contrast, relative-SNR convention (absolute-noise rows → appendix).
- Consortium screen partly reproduced; authors to be consulted.
- Failed expectations listed in one paragraph: randomized ceiling share, compounding,
  absorber effect on the other screen, cool-host noise-shape cost, and the retired repair rule
  (one sentence, as a lesson about denominators).

---

## Cut entirely (in the repository, not in the paper)
- The repair-rule history, the Fourier and invertibility rivals, the gain-ramp axis.
- The resolution ladder (R100/R200), the six-pipeline transfer study, the prevalence analysis,
  the physics band-index baseline, the realism-subset check, the per-channel offset axis.
- The ExoSim2-vs-ExoRad comparison beyond one methods sentence; the exosim "noise colouring" rows.
- Label-margin tables beyond the cut-sensitivity sentence; amplitude-quintile error analysis.
- Compound sub-additivity beyond one clause; absolute-noise rows (appendix table only).
- Everything from v2 and study1_R200.

## Appendices
A. Consortium screen rebuild: recipe, deviations, full Table-6 comparison, noise sensitivity.
B. Full envelope tables (Tier 3 and Tier 1, all rules, all cases), ceilings, held-out transfer.
C. Conformal and calibration tables.
D. Forward model: fork changes, added opacity tables, the NH3 episode, K_zz sweep, chemistry.
E. Pre-registered expectations and their outcomes.

## Figures
1. The three-region map: one panel per region, loss vs what recovers it (bar per mismatch:
   irreducible / recovered by training / recovered by randomization / residual).
2. HCN + C2H2: a carbon-rich planet's spectrum with and without, at Ariel binning; the
   screen's predicted probability for carbon-rich planets before and after (histogram).
3. Detection inversion: error-ranking AUROC of confidence vs distance across mismatches,
   the absorber axis highlighted; credit vs clean selective baseline.
4. Tier 3 vs Tier 1: same mismatches, frozen / randomized / ceiling; inset: real targets.
