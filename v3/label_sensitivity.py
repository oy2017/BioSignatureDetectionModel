"""The panel's scores under the label audit of 2026-10-30 (results/jwst_panel/label_sources.csv), which read all 104 cells against their sources after the verdicts were known.

The frozen labels stay the scored ones, since changing them after the verdicts would be post hoc. This script reports what every design scores under two corrected sets:
  corrected  the three scored labels that depart from the rule leave (TOI-270 d NH3 in the Felix reduction, K2-18 b H2O in the Madhusudhan and Schmidt reductions) and the two unscored cells the rule scores enter (HD 209458 b CO2 and WASP-52 b NH3, both absent);
  strict     also every judgement call: TOI-270 d (Holmberg) NH3, HAT-P-18 b CO2, L 98-59 d H2O, CH4 and NH3, WASP-17 b H2O and WASP-39 b H2O leave, and WASP-52 b CO2 enters as absent.
Writes results/jwst_panel/label_sensitivity.txt.
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "jwst_panel")
DES = ["published", "no-optical", "AIRS-only", "shape flag", "haze-retrained", "haze+deck-retrained", "the method", "census, same haze design"]
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
T = pd.read_csv(os.path.join(OUT, "panel_tier1.csv"))
frozen = {(r.planet, r.molecule): r.label for r in T[(T.design == "published") & (T.precision == "JWST") & T.label.notna()].itertuples()}
OUT_REAL = [("TOI-270d", "NH3"), ("K2-18b", "H2O"), ("K2-18b-alt", "H2O")]; IN_REAL = {("HD209458b", "CO2"): "absent", ("WASP-52b", "NH3"): "absent"}
OUT_JUDGE = [("TOI-270d-hm", "NH3"), ("HAT-P-18b", "CO2"), ("L98-59d", "H2O"), ("L98-59d", "CH4"), ("L98-59d", "NH3"), ("WASP-17b", "H2O"), ("WASP-39b", "H2O")]; IN_JUDGE = {("WASP-52b", "CO2"): "absent"}
assert all(k in frozen for k in OUT_REAL + OUT_JUDGE); assert not any(k in frozen for k in list(IN_REAL) + list(IN_JUDGE))
corrected = {k: v for k, v in frozen.items() if k not in OUT_REAL}; corrected.update(IN_REAL)
strict = {k: v for k, v in corrected.items() if k not in OUT_JUDGE}; strict.update(IN_JUDGE)
rng = np.random.default_rng(405); L = ["Scores under the label audit; right 1, split or withheld 0.5, wrong 0; difference from the re-implemented classifier with a 95 % bootstrap over planets", ""]
for tag, labs in (("frozen", frozen), ("corrected", corrected), ("strict", strict)):
    n = len(labs); nabs = sum(v == "absent" for v in labs.values()); L.append(f"== {tag}: {n} labels ({nabs} absent, {n - nabs} present; calling everything absent scores {nabs})")
    for prec in ("JWST", "Tier-1"):
        Q = T[T.precision == prec]; rows = []
        for (pl, m), lab in labs.items():
            q = Q[(Q.planet == pl) & (Q.molecule == m)].set_index("design")
            rows.append(dict(planet=pl, molecule=m, **{d: float((q.present[d] if lab == "present" else q.absent[d]) + 0.5 * q.split[d]) for d in DES}))
        C = pd.DataFrame(rows); plan = C.planet.map(BASE).to_numpy(); units = sorted(set(plan)); per = C.groupby(plan)[DES].sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy(); ix = rng.integers(0, len(units), (20000, len(units))); den = cnt[ix].sum(1)
        cells = []
        for d in DES:
            dif = (per[d].to_numpy()[ix].sum(1) - per["published"].to_numpy()[ix].sum(1)) / den * n
            cells.append(f"{d} {C[d].sum():.1f}" + ("" if d == "published" else f" ({C[d].sum() - C['published'].sum():+.1f}; {np.percentile(dif, 2.5):+.1f},{np.percentile(dif, 97.5):+.1f})"))
        L.append(f"  {prec}: " + "; ".join(cells))
    L.append("")
txt = "\n".join(L); open(os.path.join(OUT, "label_sensitivity.txt"), "w").write(txt + "\n"); print(txt)
