# Pre-registered predictions: outcome

Written 2026-09-11 after the v3 analysis chain ran (`run_chain.sh`, logs in this directory).
`PREREGISTERED_PREDICTIONS.md` is left exactly as committed; this file is the scorecard.
All numbers: frozen `norm_xgb`, pooled five test sets (n = 9016), clean accuracy 95.84 %.

## 1. Radiometric noise axis (102 draws per spectrum): predicted unrepairable, 20–40 % recovered

The training noise is ExoRad-shaped at SNR 15; the ExoSim2-shaped noise costs 0.21 points
at the same SNR (`evaluate_shifts.log`, "noise colouring"), so the *shape* is not the test —
the per-bin redraw is. Measured on the cases that cost more than 5 points:

| axis | case | cost (pts) | recovered |
|---|---|---|---|
| correlated noise | SNR 8 | 6.8 | 48 % |
| correlated noise | SNR 5 | 12.1 | 50 % |
| white noise | SNR 5 | 7.9 | 40 % |

**Verdict: direction right, band wrong.** All three land below the 60 % line that the file
named as falsification, so the rule survives; but correlated noise recovers 48–50 %, above the
20–40 % band predicted from the v2 grid. The per-bin axes repair better on this grid than on v2.

## 2. Deterministic (≤ 1 draw) axes, for comparison: predicted > 60 %

| axis | case | cost (pts) | recovered |
|---|---|---|---|
| stellar spots | 10 % | 6.5 | 76 % |
| stellar spots | 20 % | 9.7 | 69 % |
| gain ramp | ×2 | 5.2 | 77 % |

Haze never clears the 5-point criterion on this grid (worst case 2.4e8: −4.6 pts, 52 % recovered).
Deterministic 69–77 % vs stochastic 40–50 %: **non-overlapping, as the rule requires, but the
separation is ~25 points, half the ~50 points seen on v2 (79–89 vs 29–35).** The paper must
report the v3 bands, not the v2 ones; the three scripts that had the v2 numbers typed in
(`augment_ramp.py`, `frequency.py`) now read them from the CSVs.

## 3. Axis 8, quenched composition (0 draws): predicted repairable > 60 %, cost concentrated < 1000 K

**Outcome: untestable in the pre-registered direction.** The frozen pipeline does *better* on
quenched spectra: 95.84 → 98.24 % (all four pipelines agree: norm_mlp +2.6, norm_rf +2.6,
pca_xgb +5.9). There is no loss to recover, so the prediction is neither confirmed nor falsified.

The secondary prediction is **wrong in sign**: the change is concentrated below 1000 K exactly
as the chemistry said (88.61 → 97.44 % on 2345 cool planets; ≤ 0.3 pts elsewhere), but it is a
gain, not a cost. Why (from `test*_params[_quenched].parquet`, 500–1000 K): quenching freezes
the deep, hot partition and carries it up, and at depth the C/O > 1 water depletion is enormous.
The label separation in mean log10 abundance between C/O > 1 and C/O < 1 planets goes from
H2O −0.31 dex, CH4 +0.72, CO2 −0.03 (equilibrium) to H2O −3.30, CH4 +3.62, CO2 −3.03 (quenched).
Quenching makes the label *more* visible in cool atmospheres, so a screen trained on
equilibrium chemistry is helped, not hurt, by a disequilibrium it never saw. This is a real
result about the label, not about the repair rule.

**Post hoc reverse direction** (labelled as such in `augment_quenched.py`, not pre-registered):
a screen trained on quenched spectra only, deployed on equilibrium spectra, loses 8.9 points
(98.56 → 89.67 %); mixing equilibrium spectra into training recovers 65 % of that loss, inside
the deterministic band. Consistent with the rule, but it carries the weight of a post-hoc test.

## 4. Time-domain pilot

Not run. No prediction scored.

## What changes in the paper

- Report v3 bands (69–77 / 40–50), not v2's. The qualitative claim (ordering by draw count, not
  by physics vs instrument and not by frequency content: `frequency.log` Spearman −0.30 over five cases, the
  *wrong* way for the Fourier account) stands; the quantitative gap is half as wide.
- The one-draw/per-bin "20–40 %" number is retired; say "below 50 %" or give the measured range.
- Axis 8 is reported as a finding about the C/O label under disequilibrium, with the repair test
  marked untestable forward and 65 % post hoc in reverse.

## Addendum (same day): the ceiling correction retires the repair rule

`oracle.py` trains the pipeline only at each test strength and uses that accuracy as the
ceiling instead of the clean accuracy. Against the ceiling, mixed-strength augmentation
recovers 83–99 % (deterministic) and 76–107 % (stochastic) of the recoverable loss on v3, and
90–99 % / 55–104 % on v2, the low stochastic values being the cases whose test SNR lay outside
the training levels. **There is no difference in what augmentation can absorb.** The
deterministic/stochastic split in every Table-4-style result is a split in *irreducible* loss
(v3: 0.4–2.8 points for re-rendered or injected deterministic shifts, 3.0–4.8 points for
noise), not in augmentation's reach. The draw-count rule, the gain-ramp test and the Fourier
and invertibility rival eliminations were explaining an effect that the denominator created.
Sections 1–2 above stand as measurements; their interpretation does not. Full tables:
`results/ariel_oracle.txt` (v3) and `../v2/results/ariel_oracle.txt` (v2).
