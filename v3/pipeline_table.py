"""One table for the detector pipelines: the consortium classifier, its AIRS-only input, the input-disagreement routing,
the design retrained on the haze-and-deck population, that design with its own routing, and the census-first pipeline
(muting beyond the prior -> retrained design, else either flag -> AIRS-only), each scored on clean simulation, on each
departure, and on the real JWST panel.

Simulation columns: mean accuracy over the four learners and four molecules on POP-I from the remedy block of
alfnoor_diagnostic.py (results/diagnostic_remedy.csv for the consortium learners, diagnostic_remedy_hazedeck.csv for the
retrained ones, both from the same test cases and noise draw); 'frozen' is the design on all inputs, 'fallback_everywhere'
its AIRS-only input, 'remedied' the either-flag routing between the two. The scale-height rows are measured by
pipeline_sim.py on the same test cases and noise draw. The census row needs a published composition and is not simulated;
its simulated values repeat the either row and are bracketed in the manuscript. Real-panel columns: results/jwst_panel/muting_router.csv and router_real.csv (26 reductions, 22
planets, labels fixed before any verdict).

Writes results/pipeline_table.{txt,csv}.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS  # noqa: E402

CASES = [("clean", "clean"), ("haze3e7", "haze 3e7"), ("cloud1e2", "deck 1e2 Pa"), ("spots20", "spots 20 %"), ("white_x2", "noise x2")]


def sim(df, rule, col):
    g = df[df.rule == rule].groupby("case")[col].mean()
    return {c: 100 * g.get(c, np.nan) for c, _ in CASES}


def panel(df, router):
    q = df[df.router == router]; c = q.correct.dropna().astype(bool)
    ch4 = q[(q.molecule == "CH4") & (q.label == "present")].verdict.value_counts()
    return f"{int(c.sum())}/{len(c)}", f"{ch4.get('present', 0)}/{ch4.get('split', 0)}/{ch4.get('absent', 0)}"


def measured(ps, name):
    g = ps[ps.pipeline == name].groupby("case").accuracy.mean()
    return {c: g[c] for c, _ in CASES}


def scored(df, router):
    q = df[df.router == router].set_index(["planet", "molecule"]); return q.correct.dropna().astype(bool)


def main():
    rem = pd.read_csv(os.path.join(RESULTS, "diagnostic_remedy.csv"))
    rem_h = os.path.join(RESULTS, "diagnostic_remedy_hazedeck.csv"); rem_h = pd.read_csv(rem_h) if os.path.exists(rem_h) else None
    mr = pd.read_csv(os.path.join(RESULTS, "jwst_panel", "muting_router.csv")); rr = pd.read_csv(os.path.join(RESULTS, "jwst_panel", "router_real.csv"))
    ps = pd.read_csv(os.path.join(RESULTS, "pipeline_sim.csv"))
    routers = [(mr, "none"), (rr, "AIRS-only everywhere"), (mr, "either -> AIRS-only"), (mr, "retrained hazedeck everywhere"), (mr, "retrained hazedeck everywhere, either -> its AIRS-only"),
               (mr, "muting -> retrained hazedeck, else either -> AIRS-only"), (mr, "proxy -> retrained haze, else either -> AIRS-only"), (mr, "proxy (two-sided) -> retrained hazedeck, amplified -> withheld, else either -> AIRS-only")]
    common = None
    for d_, r_ in routers:
        i_ = set(scored(d_, r_).index); common = i_ if common is None else common & i_
    common = sorted(common); on_common = [int(scored(d_, r_).loc[common].sum()) for d_, r_ in routers]
    rows = [
        ("consortium classifier, all inputs", sim(rem, "either", "frozen"), panel(mr, "none")),
        ("consortium classifier, AIRS-only input", sim(rem, "either", "fallback_everywhere"), panel(rr, "AIRS-only everywhere")),
        ("either flag -> AIRS-only", sim(rem, "either", "remedied"), panel(mr, "either -> AIRS-only")),
        ("retrained (haze + deck), all inputs", sim(rem_h, "either", "frozen") if rem_h is not None else {}, panel(mr, "retrained hazedeck everywhere")),
        ("retrained, either flag -> its AIRS-only", sim(rem_h, "either", "remedied") if rem_h is not None else {}, panel(mr, "retrained hazedeck everywhere, either -> its AIRS-only")),
        ("census-first: muting -> retrained, else either -> AIRS-only", sim(rem, "either", "remedied"), panel(mr, "muting -> retrained hazedeck, else either -> AIRS-only")),
        ("scale-height flag (no composition) -> retrained haze, else either", measured(ps, "pipeline (haze, flag observed)"), panel(mr, "proxy -> retrained haze, else either -> AIRS-only")),
        ("scale-height flag -> retrained haze + deck, two-sided (amplified withheld)", measured(ps, "pipeline (hazedeck, flag observed), two-sided"), panel(mr, "proxy (two-sided) -> retrained hazedeck, amplified -> withheld, else either -> AIRS-only")),
    ]
    out = []
    for (name, s, (sc, ch4)), oc in zip(rows, on_common):
        r = dict(pipeline=name); r.update({lab: s.get(c, np.nan) for c, lab in CASES}); r["panel scored"] = sc; r[f"right of {len(common)} common labels"] = oc; r["panel CH4 present/split/absent"] = ch4; out.append(r)
    df = pd.DataFrame(out); df.to_csv(os.path.join(RESULTS, "pipeline_table.csv"), index=False)
    L = [f"{len(common)} labels are scored by every pipeline.", "Detector pipelines side by side (pipeline_table.py): simulated accuracy (%, mean over four learners and four molecules, POP-I at Tier-1 noise)",
         "and the real JWST panel (26 reductions; scored labels correct; CH4 verdicts on the eight methane reductions). Census row simulated as the either row; scale-height rows measured by pipeline_sim.py.", "",
         f"  {'pipeline':<60} " + " ".join(f"{lab:>12}" for _, lab in CASES) + f" {'panel':>8} {'common':>7} {'CH4 p/s/a':>10}"]
    for _, r in df.iterrows():
        L.append(f"  {r.pipeline:<60} " + " ".join(f"{r[lab]:12.1f}" if np.isfinite(r[lab]) else f"{'-':>12}" for _, lab in CASES) + f" {r['panel scored']:>8} {r[f'right of {len(common)} common labels']:>7} {r['panel CH4 present/split/absent']:>10}")
    txt = "\n".join(L); print(txt); open(os.path.join(RESULTS, "pipeline_table.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
