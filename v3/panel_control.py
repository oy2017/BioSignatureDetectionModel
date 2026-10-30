"""How much of the panel's score is the coverage mask? (review of 2026-10-30)

A label passes the control when the all-input design, reading the planet's clear twin through the reduction's own mask and scattered with the real per-bin errors (panel_mechanism.py), gets
it right in at least half of the draws: it then cannot be said that the mask alone hides or fakes the label. Every design is scored on the labels that pass, on those that fail and on the
labels with no twin, at JWST precision and at Tier-1 noise (panel_tier1.py), and on the reductions with measured data both below 1.1 um and above 1.95 um.
Also gives the trivial baseline that calls every label absent. Writes results/jwst_panel/panel_control.{txt,csv}.
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "jwst_panel")
DES = ["published", "no-optical", "AIRS-only", "shape flag", "haze-retrained", "haze+deck-retrained", "the method", "census, same haze design", "amplitude flag, AIRS-only elsewhere"]
BASE = lambda s: s.replace("-alt", "").replace("-hu", "").replace("-hm", "")
T = pd.read_csv(os.path.join(OUT, "panel_tier1.csv")); S = pd.read_csv(os.path.join(OUT, "panel_mechanism_scaled.csv")); Tr = pd.read_csv(os.path.join(OUT, "panel_mechanism_transplants.csv"))
tw = S[(S.test == "twin") & (S.variant == "all") & (S.precision == "JWST")].set_index(["planet", "molecule"]); tws = S[(S.test == "twin") & (S.variant == "airs_only") & (S.precision == "JWST")].set_index(["planet", "molecule"])
lab = T[(T.design == "published") & (T.precision == "JWST") & T.label.notna()][["planet", "molecule", "label"]].copy()
def ctl(r):
    k = (r.planet, r.molecule)
    if k not in tw.index: return "no twin"
    f = tw.loc[k].present_freq
    return "passes" if (f >= 0.5) == (r.label == "present") else "fails"
lab["control"] = lab.apply(ctl, axis=1)
lab["airs_control"] = lab.apply(lambda r: "no twin" if (r.planet, r.molecule) not in tws.index else ("passes" if (tws.loc[(r.planet, r.molecule)].present_freq >= 0.5) == (r.label == "present") else "fails"), axis=1)
both = set(Tr[(Tr.measured_optical > 0) & (Tr.measured_airs > 0)].planet)
lab.to_csv(os.path.join(OUT, "panel_control.csv"), index=False)
L = ["Control through the mask: labels whose clear twin the all-input design gets right through the reduction's own mask at the real errors, and the designs' scores on them", ""]
L.append(lab.groupby(["control", "label"]).size().unstack(fill_value=0).to_string()); L.append("")
L.append("Fails (the mask alone misleads the all-input design): " + ", ".join(f"{r.planet} {r.molecule} ({r.label})" for r in lab[lab.control == "fails"].itertuples())); L.append("")
piv = {p: T[(T.precision == p) & T.label.notna()].pivot_table(index=["planet", "molecule"], columns="design", values="credit")[DES] for p in ("JWST", "Tier-1")}
rng = np.random.default_rng(405)
def row(mask_df, tag):
    idx = pd.MultiIndex.from_frame(mask_df[["planet", "molecule"]]); n = len(idx); nabs = int((mask_df.label == "absent").sum())
    L.append(f"== {tag}: {n} labels ({nabs} absent, {n - nabs} present; calling everything absent scores {nabs})")
    for p in ("JWST", "Tier-1"):
        q = piv[p].loc[idx]; plan = np.array([BASE(a) for a in q.index.get_level_values(0)]); units = sorted(set(plan)); per = q.groupby(plan).sum().loc[units]; cnt = pd.Series(plan).value_counts().loc[units].to_numpy(); ix = rng.integers(0, len(units), (20000, len(units))); den = cnt[ix].sum(1)
        cells = []
        for d in DES:
            dif = (per[d].to_numpy()[ix].sum(1) - per["published"].to_numpy()[ix].sum(1)) / den * n
            cells.append(f"{d} {q[d].sum():.1f}" + ("" if d == "published" else f" ({q[d].sum() - q['published'].sum():+.1f}; {np.percentile(dif, 2.5):+.1f},{np.percentile(dif, 97.5):+.1f})"))
        L.append(f"  {p}: " + "; ".join(cells))
    L.append("")
row(lab, "All labels"); row(lab[lab.control == "passes"], "Labels passing the control"); row(lab[lab.control == "fails"], "Labels failing the control"); row(lab[lab.control == "no twin"], "Labels with no twin")
row(lab[lab.planet.isin(both)], f"Reductions with measured data below 1.1 um and above 1.95 um ({len(both)})")
row(lab[(lab.planet.isin(both)) & (lab.control == "passes")], "Those reductions, labels passing the control")
row(lab[lab.label == "present"], "Present labels only"); row(lab[(lab.label == "present") & (lab.control == "passes")], "Present labels passing the control")
txt = "\n".join(L); open(os.path.join(OUT, "panel_control.txt"), "w").write(txt + "\n"); print(txt)
